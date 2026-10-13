import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sandbox_server import memory_guard
from sandbox_server.actions import ActionDispatcher
from sandbox_server.config import get_config
from sandbox_server.isolation.base import IsolationBackend, TrialHandle
from sandbox_server.profiles import SandboxProfile
from sandbox_server.reward import RewardResult, compute_task_reward
from sandbox_server.schemas import ActionResponse, CapacityView, TrialState
from sandbox_server.tasks import TaskSpec, resolve_task
from sandbox_server.workspace import WorkspaceManager

log = logging.getLogger(__name__)


@dataclass
class SessionRecord:
    """One per training job. Binds to a profile and owns trials."""

    session_id: str
    profile: SandboxProfile
    metadata: dict[str, Any]
    created_at: float = field(default_factory=time.time)
    trial_ids: set[str] = field(default_factory=set)


@dataclass
class TrialRecord:
    """One per rollout episode. Holds workspace, backend handle, and task."""

    trial_id: str
    session_id: str
    task: TaskSpec
    workspace: Path
    metadata: dict[str, Any]
    lease_seconds: float
    lease_deadline: float
    handle: TrialHandle | None = None
    profile_name: str = ""
    state: TrialState = TrialState.READY
    # Serializes actions within one trial — only one tool call executes at a time.
    action_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Tracks in-flight actions so the lease reaper won't destroy a trial mid-execution.
    pending_actions: int = 0

    @property
    def task_id(self) -> str:
        return self.task.task_id


@dataclass
class ProfileRuntime:
    """Live counters and concurrency semaphore for one profile."""

    profile: SandboxProfile
    # Bounds how many actions (nsjail jails / docker exec calls) run simultaneously
    # across all trials of this profile. Excess actions queue on acquire().
    semaphore: asyncio.Semaphore
    # Optional narrower limits for expensive tools (for example execute_python).
    # Tools not listed here only use the profile semaphore.
    tool_semaphores: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    live_trials: int = 0
    running_actions: int = 0
    queued_actions: int = 0
    running_actions_by_tool: dict[str, int] = field(default_factory=dict)
    queued_actions_by_tool: dict[str, int] = field(default_factory=dict)


class SandboxManager:
    """Core state machine. Manages session → trial → action lifecycle,
    concurrency limits, lease reaping, and reward computation."""

    def __init__(
        self,
        profiles: dict[str, SandboxProfile],
        backends: dict[str, IsolationBackend],
        workspace_root: Path,
        lease_check_interval: float = 5.0,
    ):
        self.profiles = profiles
        self.backends = backends
        self.workspace_manager = WorkspaceManager(workspace_root)
        self.dispatcher = ActionDispatcher(self.workspace_manager)
        self.sessions: dict[str, SessionRecord] = {}
        self.trials: dict[str, TrialRecord] = {}
        self.runtimes = {
            name: ProfileRuntime(
                profile,
                asyncio.Semaphore(profile.max_running_actions),
                {
                    tool: asyncio.Semaphore(limit)
                    for tool, limit in profile.tool_max_running_actions.items()
                    if limit > 0
                },
            )
            for name, profile in profiles.items()
        }
        self.lease_check_interval = lease_check_interval
        # Guards mutations to sessions, trials dicts, and live_trials counters.
        # Not held during long operations (backend calls, action execution).
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None

    def _backend(self, profile: SandboxProfile) -> IsolationBackend:
        backend = self.backends.get(profile.backend_type)
        if backend is None:
            raise RuntimeError(f"no backend of type {profile.backend_type!r} for profile {profile.name!r}")
        return backend

    async def start(self) -> None:
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._lease_reaper())

    async def close(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            await asyncio.gather(self._reaper_task, return_exceptions=True)
            self._reaper_task = None
        for trial_id in list(self.trials):
            await self.destroy_trial(trial_id)

    async def create_session(self, profile_name: str, metadata: dict[str, Any]) -> SessionRecord:
        if profile_name not in self.profiles:
            raise KeyError(f"unknown profile {profile_name!r}")
        profile = self.profiles[profile_name]
        backend = self._backend(profile)
        if not backend.available:
            raise RuntimeError(f"backend {profile.backend_type!r} for profile {profile_name!r} is not available")
        memory_error = self._memory_pressure_error()
        if memory_error is not None:
            raise RuntimeError(memory_error)
        session = SessionRecord(uuid.uuid4().hex, profile, metadata)
        async with self._lock:
            self.sessions[session.session_id] = session
        return session

    async def destroy_session(self, session_id: str) -> None:
        session = self._session(session_id)
        for trial_id in list(session.trial_ids):
            await self.destroy_trial(trial_id)
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def create_trial(
        self,
        session_id: str,
        task_id: str,
        metadata: dict[str, Any],
        lease_seconds: float | None = None,
    ) -> TrialRecord:
        session = self._session(session_id)
        memory_error = self._memory_pressure_error()
        if memory_error is not None:
            raise RuntimeError(memory_error)

        # The daemon resolves task_id to a directory under its own task root;
        # clients never supply paths, images, or reward configuration.
        task = resolve_task(session.profile.task_root, task_id)

        # Atomic admission: increment live_trials under lock, then do the
        # (potentially slow) workspace + backend setup outside the lock.
        # If setup fails, we roll back the counter.
        runtime = self.runtimes[session.profile.name]
        async with self._lock:
            if runtime.live_trials >= session.profile.max_live_trials:
                raise RuntimeError(f"profile {session.profile.name!r} live trial capacity exhausted")
            runtime.live_trials += 1

        # trial_id is a UUID — multiple trials can share the same task_id
        # (e.g., GRPO creates N rollouts for one prompt, each gets its own trial).
        trial_id = uuid.uuid4().hex
        workspace = None
        handle = None
        backend = self._backend(session.profile)
        try:
            workspace = self.workspace_manager.create(session_id, trial_id)
            if task.workspace_dir is not None:
                await asyncio.to_thread(shutil.copytree, task.workspace_dir, workspace, dirs_exist_ok=True)
            handle = await backend.create_trial(trial_id, session.profile, workspace, task.config)
            for command in task.setup_commands:
                result = await backend.execute(
                    handle, session.profile, ["/bin/bash", "-lc", command], session.profile.setup_timeout_seconds
                )
                if result.returncode != 0:
                    raise ValueError(
                        f"setup command failed for task {task_id!r} (exit {result.returncode}): "
                        f"{(result.stderr or result.stdout).strip()[-2000:]}"
                    )
        except Exception:
            # Rollback: clean up backend state, workspace and counter.
            if handle is not None:
                try:
                    await backend.destroy(handle)
                except Exception:
                    log.exception("error destroying trial %s after failed setup", trial_id)
            if workspace is not None:
                self.workspace_manager.destroy(workspace)
            async with self._lock:
                runtime.live_trials -= 1
            raise

        lease = lease_seconds or session.profile.lease_seconds
        trial = TrialRecord(
            trial_id=trial_id,
            session_id=session_id,
            task=task,
            workspace=workspace,
            metadata=metadata,
            lease_seconds=lease,
            lease_deadline=time.monotonic() + lease,
            handle=handle,
            profile_name=session.profile.name,
        )
        async with self._lock:
            self.trials[trial_id] = trial
            session.trial_ids.add(trial_id)
        return trial

    async def heartbeat(self, trial_id: str) -> TrialRecord:
        trial = self._trial(trial_id)
        trial.lease_deadline = time.monotonic() + trial.lease_seconds
        return trial

    async def execute_action(
        self,
        trial_id: str,
        tool: str,
        args: dict[str, Any],
        timeout: float,
    ) -> ActionResponse:
        trial = self._trial(trial_id)
        session = self._session(trial.session_id)
        runtime = self.runtimes[session.profile.name]
        backend = self._backend(session.profile)

        # Under memory pressure, still allow return_answer so active episodes
        # can finish and the client can clean up. Block heavier tools.
        if tool != "return_answer":
            memory_error = self._memory_pressure_error()
            if memory_error is not None:
                return ActionResponse(
                    success=False,
                    error=memory_error,
                    queue_wait_seconds=0.0,
                    execution_seconds=0.0,
                )

        # Extend lease on each action — an active trial shouldn't be reaped.
        trial.lease_deadline = time.monotonic() + trial.lease_seconds

        # pending_actions protects against lease reaper: the reaper skips trials
        # with pending_actions > 0 even if the lease has expired.
        trial.pending_actions += 1

        try:
            # Serialize each trial before taking global capacity. Later actions
            # on the same trial should not consume profile/tool slots while
            # they are only waiting for an earlier action to finish.
            async with trial.action_lock:
                if trial.state in {
                    TrialState.CLEANING,
                    TrialState.DESTROYED,
                    TrialState.FAILED,
                }:
                    return ActionResponse(
                        success=False,
                        error=f"RuntimeError: trial is not executable in state {trial.state.value}",
                        queue_wait_seconds=0.0,
                        execution_seconds=0.0,
                    )

                # Queue phase: wait for a semaphore slot. This bounds how many
                # nsjail jails / docker exec calls run simultaneously across
                # all trials. It starts after action_lock acquisition, so
                # capacity reflects runnable work instead of per-trial waiters.
                queued_at = time.monotonic()
                if (
                    session.profile.max_queued_actions is not None
                    and runtime.queued_actions >= session.profile.max_queued_actions
                ):
                    return ActionResponse(
                        success=False,
                        error=(f"profile {session.profile.name!r} queued action capacity exhausted"),
                        queue_wait_seconds=0.0,
                        execution_seconds=0.0,
                    )
                runtime.queued_actions += 1
                runtime.queued_actions_by_tool[tool] = runtime.queued_actions_by_tool.get(tool, 0) + 1
                tool_semaphore = runtime.tool_semaphores.get(tool)
                acquired_tool = False
                acquired_profile = False
                try:
                    if tool_semaphore is not None:
                        await tool_semaphore.acquire()
                        acquired_tool = True
                    await runtime.semaphore.acquire()
                    acquired_profile = True
                except BaseException:
                    if acquired_tool and tool_semaphore is not None:
                        tool_semaphore.release()
                    raise
                finally:
                    runtime.queued_actions -= 1
                    runtime.queued_actions_by_tool[tool] = max(0, runtime.queued_actions_by_tool.get(tool, 0) - 1)
                queue_wait = time.monotonic() - queued_at

                # Execution phase: one action at a time per trial
                # (action_lock), but multiple trials execute in parallel
                # (semaphore has N slots).
                runtime.running_actions += 1
                runtime.running_actions_by_tool[tool] = runtime.running_actions_by_tool.get(tool, 0) + 1
                started_at = time.monotonic()
                try:
                    if timeout <= 0:
                        raise TimeoutError(f"tool {tool} timed out after {timeout:g}s")
                    trial.state = TrialState.ACTIVE
                    trial.lease_deadline = time.monotonic() + trial.lease_seconds
                    result = await asyncio.wait_for(
                        self.dispatcher.execute(
                            backend,
                            session.profile,
                            trial.handle,
                            tool,
                            args,
                            timeout,
                        ),
                        timeout=timeout,
                    )
                    trial.state = TrialState.READY
                    return ActionResponse(
                        success=True,
                        result=result,
                        queue_wait_seconds=queue_wait,
                        execution_seconds=time.monotonic() - started_at,
                    )
                except asyncio.TimeoutError:
                    trial.state = TrialState.READY
                    return ActionResponse(
                        success=False,
                        error=f"TimeoutError: tool {tool} timed out after {timeout:g}s",
                        queue_wait_seconds=queue_wait,
                        execution_seconds=time.monotonic() - started_at,
                    )
                except asyncio.CancelledError:
                    trial.state = TrialState.FAILED
                    raise
                except Exception as exc:
                    trial.state = TrialState.READY
                    return ActionResponse(
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        queue_wait_seconds=queue_wait,
                        execution_seconds=time.monotonic() - started_at,
                    )
                finally:
                    runtime.running_actions -= 1
                    runtime.running_actions_by_tool[tool] = max(0, runtime.running_actions_by_tool.get(tool, 0) - 1)
                    if acquired_profile:
                        runtime.semaphore.release()
                    if acquired_tool and tool_semaphore is not None:
                        tool_semaphore.release()
        except asyncio.CancelledError:
            trial.state = TrialState.FAILED
            raise
        finally:
            trial.pending_actions -= 1
            self._maybe_trim_memory()

    async def compute_reward(
        self,
        trial_id: str,
        answer: str | None = None,
        timeout: float | None = None,
    ) -> RewardResult:
        trial = self._trial(trial_id)
        session = self._session(trial.session_id)
        if trial.task.reward_config["type"] == "none":
            raise NotImplementedError(f"task {trial.task_id!r} has no server-side reward")
        backend = self._backend(session.profile)
        # Reward runs under action_lock — no concurrent tool execution during scoring.
        effective_timeout = timeout or session.profile.reward_timeout_seconds
        if effective_timeout <= 0:
            raise ValueError("reward timeout must be positive")
        async with trial.action_lock:
            trial.state = TrialState.EVALUATING
            try:
                memory_error = self._memory_pressure_error()
                if memory_error is not None:
                    return RewardResult(
                        reward=0.0,
                        passed=False,
                        details={
                            "error": "sandbox_memory_pressure",
                            "reason": memory_error,
                            "final_score": 0.0,
                        },
                    )
                # The deadline starts after action_lock acquisition so
                # unrelated lock wait does not consume reward time.
                result = await asyncio.wait_for(
                    compute_task_reward(backend, trial.handle, session.profile, trial.task, answer),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                result = RewardResult(
                    reward=0.0,
                    passed=False,
                    details={
                        "error": "reward_timeout",
                        "reason": f"reward timed out after {effective_timeout}s",
                        "timeout_seconds": effective_timeout,
                        "evaluation_level": "reward_timeout",
                        "final_score": 0.0,
                    },
                )
            finally:
                trial.state = TrialState.READY
                self._maybe_trim_memory()
        return result

    async def destroy_trial(self, trial_id: str) -> None:
        trial = self.trials.get(trial_id)
        if trial is None:
            return  # idempotent — already destroyed or never existed
        # Acquire action_lock so destroy waits for any in-flight action to finish.
        async with trial.action_lock:
            trial.state = TrialState.CLEANING
            if trial.handle is not None:
                try:
                    backend = self.backends.get(self.profiles[trial.profile_name].backend_type)
                    if backend is not None:
                        await backend.destroy(trial.handle)
                except Exception:
                    log.exception("error destroying trial %s backend state", trial_id)
            self.workspace_manager.destroy(trial.workspace)
            trial.state = TrialState.DESTROYED

        # Remove from global state and decrement counter under lock.
        async with self._lock:
            self.trials.pop(trial_id, None)
            session = self.sessions.get(trial.session_id)
            if session is not None:
                session.trial_ids.discard(trial_id)
            if trial.profile_name in self.runtimes:
                self.runtimes[trial.profile_name].live_trials -= 1
        self._maybe_trim_memory()

    def capacity(self) -> list[CapacityView]:
        result = []
        for name, runtime in self.runtimes.items():
            result.append(
                CapacityView(
                    profile=name,
                    live_trials=runtime.live_trials,
                    max_live_trials=runtime.profile.max_live_trials,
                    running_actions=runtime.running_actions,
                    max_running_actions=runtime.profile.max_running_actions,
                    queued_actions=runtime.queued_actions,
                    max_queued_actions=runtime.profile.max_queued_actions,
                    running_actions_by_tool={
                        tool: count for tool, count in runtime.running_actions_by_tool.items() if count
                    },
                    queued_actions_by_tool={
                        tool: count for tool, count in runtime.queued_actions_by_tool.items() if count
                    },
                    max_running_actions_by_tool=runtime.profile.tool_max_running_actions,
                )
            )
        return result

    def _session(self, session_id: str) -> SessionRecord:
        if session_id not in self.sessions:
            raise KeyError(f"unknown session {session_id!r}")
        return self.sessions[session_id]

    def _trial(self, trial_id: str) -> TrialRecord:
        if trial_id not in self.trials:
            raise KeyError(f"unknown trial {trial_id!r}")
        return self.trials[trial_id]

    @staticmethod
    def _memory_pressure_error() -> str | None:
        cfg = get_config()
        over_limit, snap = memory_guard.over_soft_limit(cfg.sandbox_memory_soft_limit_bytes)
        if not over_limit:
            return None
        return memory_guard.pressure_error(cfg.sandbox_memory_soft_limit_bytes, snap)

    @staticmethod
    def _maybe_trim_memory() -> None:
        cfg = get_config()
        memory_guard.maybe_trim(cfg.sandbox_memory_trim_threshold_bytes)

    async def _lease_reaper(self) -> None:
        """Background task: periodically destroys expired trials.
        Skips trials with pending_actions > 0 — those are still executing
        and will renew their lease when the action completes."""
        while True:
            await asyncio.sleep(self.lease_check_interval)
            now = time.monotonic()
            expired = [
                trial_id
                for trial_id, trial in self.trials.items()
                if trial.lease_deadline <= now and trial.pending_actions == 0
            ]
            for trial_id in expired:
                await self.destroy_trial(trial_id)
