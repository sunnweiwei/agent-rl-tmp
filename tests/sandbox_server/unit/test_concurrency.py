"""Concurrency tests for SandboxManager.

Verifies that:
- Multiple trials can execute actions in parallel (across trials)
- Actions within a single trial are serialized (action_lock)
- The per-profile semaphore bounds max_running_actions
- Queued actions wait and then proceed when slots free
- Concurrent trial creation/destruction doesn't corrupt counters
- Lease reaper doesn't destroy trials with pending actions
"""

import asyncio
import time

import pytest
import pytest_asyncio

from sandbox_server.isolation.base import ExecutionResult, IsolationBackend, TrialHandle
from sandbox_server.manager import SandboxManager
from sandbox_server.profiles import SandboxProfile


class SlowMockBackend(IsolationBackend):
    """Backend with configurable execution delay for concurrency testing."""

    name = "mock"

    def __init__(self, delay: float = 0.1):
        self.delay = delay
        self.concurrent_count = 0
        self.peak_concurrent = 0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return True

    async def create_trial(self, trial_id, profile, workspace, task_config=None):
        return TrialHandle(trial_id=trial_id, workspace=workspace)

    async def execute(self, handle, profile, argv, timeout, env=None):
        async with self._lock:
            self.concurrent_count += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent_count)
        try:
            await asyncio.sleep(self.delay)
        finally:
            async with self._lock:
                self.concurrent_count -= 1
        return ExecutionResult(0, "ok", "")

    async def destroy(self, handle):
        pass


def _profile(max_actions: int = 4, max_trials: int = 100, lease: float = 300.0) -> SandboxProfile:
    return SandboxProfile(
        name="test",
        backend_type="mock",
        max_running_actions=max_actions,
        max_live_trials=max_trials,
        lease_seconds=lease,
        enabled_tools=frozenset({"bash", "return_answer"}),
    )


@pytest_asyncio.fixture
async def slow_backend():
    return SlowMockBackend(delay=0.05)


@pytest_asyncio.fixture
async def manager_and_session(tmp_path, slow_backend):
    profile = _profile(max_actions=4, max_trials=100)
    mgr = SandboxManager(
        {"test": profile},
        {"mock": slow_backend},
        tmp_path / "workspaces",
    )
    await mgr.start()
    session = await mgr.create_session("test", {})
    yield mgr, session, slow_backend
    await mgr.close()


# ---------------------------------------------------------------------------
# Cross-trial parallelism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_actions_across_trials_run_in_parallel(manager_and_session):
    """Actions on different trials should run concurrently, not sequentially."""
    mgr, session, backend = manager_and_session

    trials = []
    for i in range(4):
        trial = await mgr.create_trial(session.session_id, "", {})
        trials.append(trial)

    t0 = time.monotonic()
    results = await asyncio.gather(
        *(mgr.execute_action(t.trial_id, "bash", {"command": "echo"}, timeout=5.0) for t in trials)
    )
    elapsed = time.monotonic() - t0

    assert all(r.success for r in results)
    # 4 actions at 50ms each: parallel should take ~50-100ms, sequential would take ~200ms+
    assert elapsed < 0.2, f"4 parallel actions took {elapsed:.3f}s — should be <0.2s if parallel"
    assert backend.peak_concurrent >= 2, f"peak concurrent was {backend.peak_concurrent}, expected >=2"


@pytest.mark.asyncio
async def test_peak_concurrency_matches_trial_count(manager_and_session):
    """With enough semaphore slots, all trials should run simultaneously."""
    mgr, session, backend = manager_and_session

    trials = []
    for i in range(4):
        trials.append(await mgr.create_trial(session.session_id, "", {}))

    await asyncio.gather(*(mgr.execute_action(t.trial_id, "bash", {"command": "echo"}, timeout=5.0) for t in trials))

    assert backend.peak_concurrent == 4


# ---------------------------------------------------------------------------
# Within-trial serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_actions_within_same_trial_are_serialized(tmp_path):
    """Multiple actions on the same trial should execute one at a time (action_lock)."""
    backend = SlowMockBackend(delay=0.05)
    profile = _profile(max_actions=10)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})

    t0 = time.monotonic()
    results = await asyncio.gather(
        *(mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=5.0) for _ in range(3))
    )
    elapsed = time.monotonic() - t0

    assert all(r.success for r in results)
    # 3 serialized actions at 50ms each: should take ~150ms+
    assert elapsed >= 0.12, f"3 serialized actions took {elapsed:.3f}s — should be >=0.12s"
    # peak concurrent within this trial should be 1
    assert backend.peak_concurrent == 1

    await mgr.close()


@pytest.mark.asyncio
async def test_same_trial_waiters_do_not_consume_running_capacity(tmp_path):
    """Actions waiting on the same trial lock should not consume global slots."""
    backend = SlowMockBackend(delay=0.2)
    profile = _profile(max_actions=2)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})

    tasks = [
        asyncio.create_task(mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=5.0))
        for _ in range(5)
    ]
    await asyncio.sleep(0.05)

    caps = mgr.capacity()
    assert caps[0].running_actions == 1
    assert caps[0].queued_actions == 0
    assert backend.peak_concurrent == 1

    results = await asyncio.gather(*tasks)
    assert all(r.success for r in results)
    assert mgr.runtimes["test"].running_actions == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_action_timeout_releases_trial_lock_and_capacity(tmp_path):
    """A timed-out action must not block later actions, reward, or cleanup."""
    backend = SlowMockBackend(delay=0.2)
    profile = _profile(max_actions=2)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})

    timed_out = await mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=0.05)
    assert timed_out.success is False
    assert "TimeoutError" in timed_out.error
    assert mgr.runtimes["test"].running_actions == 0

    ok = await mgr.execute_action(trial.trial_id, "return_answer", {"answer": "ok"}, timeout=1.0)
    assert ok.success is True

    await mgr.destroy_trial(trial.trial_id)
    assert trial.trial_id not in mgr.trials
    assert mgr.runtimes["test"].running_actions == 0

    await mgr.close()


# ---------------------------------------------------------------------------
# Semaphore bounding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semaphore_bounds_max_running_actions(tmp_path):
    """When more actions are submitted than max_running_actions, excess should queue."""
    backend = SlowMockBackend(delay=0.1)
    profile = _profile(max_actions=2, max_trials=10)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trials = []
    for i in range(4):
        trials.append(await mgr.create_trial(session.session_id, "", {}))

    t0 = time.monotonic()
    results = await asyncio.gather(
        *(mgr.execute_action(t.trial_id, "bash", {"command": "echo"}, timeout=5.0) for t in trials)
    )
    elapsed = time.monotonic() - t0

    assert all(r.success for r in results)
    # With semaphore=2 and 4 tasks at 100ms: should take ~200ms (2 waves)
    assert elapsed >= 0.15, f"Expected >=0.15s for 2 waves, got {elapsed:.3f}s"
    assert backend.peak_concurrent <= 2, f"peak concurrent was {backend.peak_concurrent}, should be <=2"

    await mgr.close()


@pytest.mark.asyncio
async def test_queued_actions_report_wait_time(tmp_path):
    """Queued actions should report non-zero queue_wait_seconds."""
    backend = SlowMockBackend(delay=0.1)
    profile = _profile(max_actions=1, max_trials=10)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    t1 = await mgr.create_trial(session.session_id, "", {})
    t2 = await mgr.create_trial(session.session_id, "", {})

    r1, r2 = await asyncio.gather(
        mgr.execute_action(t1.trial_id, "bash", {"command": "echo"}, timeout=5.0),
        mgr.execute_action(t2.trial_id, "bash", {"command": "echo"}, timeout=5.0),
    )

    assert r1.success and r2.success
    # One of them should have waited (queue_wait > 0)
    max_wait = max(r1.queue_wait_seconds, r2.queue_wait_seconds)
    assert max_wait > 0.05, f"Expected queued wait >50ms, got {max_wait:.3f}s"

    await mgr.close()


# ---------------------------------------------------------------------------
# Concurrent trial creation/destruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_trial_creation(manager_and_session):
    """Creating many trials concurrently should not corrupt live_trials counter."""
    mgr, session, _ = manager_and_session

    trials = await asyncio.gather(*(mgr.create_trial(session.session_id, "", {}) for i in range(20)))

    assert len(trials) == 20
    assert mgr.runtimes["test"].live_trials == 20


@pytest.mark.asyncio
async def test_concurrent_trial_destruction(manager_and_session):
    """Destroying many trials concurrently should leave live_trials at 0."""
    mgr, session, _ = manager_and_session

    trials = await asyncio.gather(*(mgr.create_trial(session.session_id, "", {}) for i in range(10)))

    assert mgr.runtimes["test"].live_trials == 10

    await asyncio.gather(*(mgr.destroy_trial(t.trial_id) for t in trials))

    assert mgr.runtimes["test"].live_trials == 0
    assert len(mgr.trials) == 0


@pytest.mark.asyncio
async def test_create_and_destroy_interleaved(manager_and_session):
    """Interleaved create/destroy should maintain correct counter."""
    mgr, session, _ = manager_and_session

    trials = []
    for i in range(5):
        t = await mgr.create_trial(session.session_id, "", {})
        trials.append(t)

    # Destroy first 3 while creating 3 more
    await asyncio.gather(
        mgr.destroy_trial(trials[0].trial_id),
        mgr.destroy_trial(trials[1].trial_id),
        mgr.destroy_trial(trials[2].trial_id),
        mgr.create_trial(session.session_id, "", {}),
        mgr.create_trial(session.session_id, "", {}),
        mgr.create_trial(session.session_id, "", {}),
    )

    assert mgr.runtimes["test"].live_trials == 5  # 5 - 3 + 3


# ---------------------------------------------------------------------------
# Lease reaper safety
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lease_reaper_skips_trials_with_pending_actions(tmp_path):
    """Reaper should not destroy a trial that has an action in flight."""
    backend = SlowMockBackend(delay=0.3)
    profile = _profile(max_actions=4, lease=0.05)  # very short lease
    mgr = SandboxManager(
        {"test": profile},
        {"mock": backend},
        tmp_path / "ws",
        lease_check_interval=0.05,
    )
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})

    # Start a slow action — lease will expire during execution
    action_task = asyncio.create_task(mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=5.0))

    # Wait for lease to expire + reaper to run
    await asyncio.sleep(0.15)

    # Trial should still exist (pending_actions > 0 protects it)
    assert trial.trial_id in mgr.trials

    result = await action_task
    assert result.success

    # Now that action is done and lease is expired, next reaper cycle should clean it
    await asyncio.sleep(0.15)
    assert trial.trial_id not in mgr.trials

    await mgr.close()


# ---------------------------------------------------------------------------
# Capacity counters under concurrent load
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capacity_counters_during_concurrent_actions(tmp_path):
    """running_actions counter should reflect actual concurrent execution."""
    backend = SlowMockBackend(delay=0.1)
    profile = _profile(max_actions=4, max_trials=10)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trials = []
    for i in range(3):
        trials.append(await mgr.create_trial(session.session_id, "", {}))

    # Start 3 concurrent actions
    tasks = [
        asyncio.create_task(mgr.execute_action(t.trial_id, "bash", {"command": "echo"}, timeout=5.0)) for t in trials
    ]

    # Check mid-flight
    await asyncio.sleep(0.02)
    caps = mgr.capacity()
    assert caps[0].running_actions >= 2, f"expected >=2 running, got {caps[0].running_actions}"

    results = await asyncio.gather(*tasks)
    assert all(r.success for r in results)

    # After all done
    caps = mgr.capacity()
    assert caps[0].running_actions == 0

    await mgr.close()


# ---------------------------------------------------------------------------
# Cleanup race conditions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_destroy_trial_while_action_in_flight(tmp_path):
    """Destroying a trial while an action runs: action should complete or
    error cleanly, counters must be correct afterward."""
    backend = SlowMockBackend(delay=0.2)
    profile = _profile(max_actions=4, lease=300.0)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})

    action_task = asyncio.create_task(mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=5.0))

    # Let action start, then destroy
    await asyncio.sleep(0.05)
    destroy_task = asyncio.create_task(mgr.destroy_trial(trial.trial_id))

    # Both should complete without crashing
    action_result = await action_task
    await destroy_task

    # Action should have succeeded (it acquired action_lock first, destroy waited)
    assert action_result.success

    # Trial should be gone, counters clean
    assert trial.trial_id not in mgr.trials
    assert mgr.runtimes["test"].live_trials == 0
    assert mgr.runtimes["test"].running_actions == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_action_after_destroy_errors_cleanly(tmp_path):
    """An action submitted after destroy starts should fail with a clear error."""
    backend = SlowMockBackend(delay=0.01)
    profile = _profile(max_actions=4, lease=300.0)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})
    await mgr.destroy_trial(trial.trial_id)

    # Action on destroyed trial should raise KeyError (trial not found)
    with pytest.raises(KeyError, match="unknown trial"):
        await mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=5.0)

    await mgr.close()


@pytest.mark.asyncio
async def test_destroy_session_while_actions_in_flight(tmp_path):
    """Destroying a session while its trials have actions running: all actions
    should complete or error, all trials should be cleaned up, counters correct."""
    backend = SlowMockBackend(delay=0.15)
    profile = _profile(max_actions=10, max_trials=10, lease=300.0)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trials = []
    for i in range(3):
        trials.append(await mgr.create_trial(session.session_id, "", {}))

    # Start actions on all trials
    action_tasks = [
        asyncio.create_task(mgr.execute_action(t.trial_id, "bash", {"command": "echo"}, timeout=5.0)) for t in trials
    ]

    # Let actions start, then destroy session
    await asyncio.sleep(0.03)
    await mgr.destroy_session(session.session_id)

    # Gather action results — they may succeed or fail but must not crash
    results = await asyncio.gather(*action_tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, Exception):
            assert isinstance(r, KeyError | RuntimeError)
        else:
            # ActionResponse — could be success or error
            pass

    # Everything should be cleaned up
    assert session.session_id not in mgr.sessions
    assert mgr.runtimes["test"].live_trials == 0
    assert mgr.runtimes["test"].running_actions == 0
    assert len(mgr.trials) == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_double_destroy_trial_concurrent(tmp_path):
    """Two concurrent destroy calls on the same trial should both succeed
    without corrupting the counter (idempotent)."""
    backend = SlowMockBackend(delay=0.01)
    profile = _profile(max_actions=4)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})
    assert mgr.runtimes["test"].live_trials == 1

    # Two concurrent destroys
    await asyncio.gather(
        mgr.destroy_trial(trial.trial_id),
        mgr.destroy_trial(trial.trial_id),
    )

    assert trial.trial_id not in mgr.trials
    assert mgr.runtimes["test"].live_trials == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_double_destroy_trial_sequential(tmp_path):
    """Sequential double destroy should also be safe (idempotent)."""
    backend = SlowMockBackend(delay=0.01)
    profile = _profile(max_actions=4)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trial = await mgr.create_trial(session.session_id, "", {})

    await mgr.destroy_trial(trial.trial_id)
    await mgr.destroy_trial(trial.trial_id)  # should not raise

    assert mgr.runtimes["test"].live_trials == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_many_concurrent_destroys_counter_correct(tmp_path):
    """Destroy 20 trials concurrently — counter must end at 0, not go negative."""
    backend = SlowMockBackend(delay=0.01)
    profile = _profile(max_actions=10, max_trials=50)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trials = []
    for i in range(20):
        trials.append(await mgr.create_trial(session.session_id, "", {}))

    assert mgr.runtimes["test"].live_trials == 20

    await asyncio.gather(*(mgr.destroy_trial(t.trial_id) for t in trials))

    assert mgr.runtimes["test"].live_trials == 0
    assert len(mgr.trials) == 0

    await mgr.close()


# ---------------------------------------------------------------------------
# 1024-scale tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_1024_trial_creation_and_destruction(tmp_path):
    """Create 1024 trials, verify counter, destroy all, verify cleanup."""
    backend = SlowMockBackend(delay=0.001)
    profile = _profile(max_actions=128, max_trials=1100)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})

    trials = []
    for i in range(1024):
        t = await mgr.create_trial(session.session_id, "", {})
        trials.append(t)

    assert mgr.runtimes["test"].live_trials == 1024

    await asyncio.gather(*(mgr.destroy_trial(t.trial_id) for t in trials))

    assert mgr.runtimes["test"].live_trials == 0
    assert len(mgr.trials) == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_1024_trials_with_concurrent_actions(tmp_path):
    """1024 trials each execute one action. Semaphore bounds concurrency."""
    backend = SlowMockBackend(delay=0.01)
    max_actions = 64
    profile = _profile(max_actions=max_actions, max_trials=1100)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})
    trials = []
    for i in range(1024):
        trials.append(await mgr.create_trial(session.session_id, "", {}))

    t0 = time.monotonic()
    results = await asyncio.gather(
        *(mgr.execute_action(t.trial_id, "bash", {"command": "echo"}, timeout=5.0) for t in trials)
    )
    elapsed = time.monotonic() - t0

    assert all(r.success for r in results)
    assert backend.peak_concurrent <= max_actions
    # 1024 actions at 10ms each, 64 slots = ~16 waves = ~160ms minimum
    assert elapsed < 5.0, f"1024 actions took {elapsed:.2f}s, expected <5s"

    caps = mgr.capacity()
    assert caps[0].running_actions == 0

    await asyncio.gather(*(mgr.destroy_trial(t.trial_id) for t in trials))
    assert mgr.runtimes["test"].live_trials == 0

    await mgr.close()


@pytest.mark.asyncio
async def test_1024_trials_create_action_destroy_cycle(tmp_path):
    """Full lifecycle at scale: create, execute, destroy for 1024 trials."""
    backend = SlowMockBackend(delay=0.005)
    profile = _profile(max_actions=128, max_trials=1100)
    mgr = SandboxManager({"test": profile}, {"mock": backend}, tmp_path / "ws")
    await mgr.start()

    session = await mgr.create_session("test", {})

    async def lifecycle(i):
        trial = await mgr.create_trial(session.session_id, "", {})
        result = await mgr.execute_action(trial.trial_id, "bash", {"command": "echo"}, timeout=5.0)
        await mgr.destroy_trial(trial.trial_id)
        return result.success

    results = await asyncio.gather(*(lifecycle(i) for i in range(1024)))

    assert all(results)
    assert mgr.runtimes["test"].live_trials == 0
    assert len(mgr.trials) == 0

    await mgr.close()
