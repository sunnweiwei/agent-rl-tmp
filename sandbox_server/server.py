import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sandbox_server.config import get_config
from sandbox_server.isolation import (
    DockerBackend,
    IsolationBackend,
    NsjailBackend,
    SubprocessBackend,
)
from sandbox_server.manager import SandboxManager
from sandbox_server.profiles import default_profiles
from sandbox_server.schemas import (
    ActionRequest,
    ActionResponse,
    CapacityView,
    HealthView,
    RewardRequest,
    RewardView,
    SessionCreate,
    SessionView,
    TrialCreate,
    TrialView,
)


def _build_backends(cfg) -> dict[str, IsolationBackend]:
    """All backends are instantiated; only the one matching the profile's
    backend_type is used at runtime. Unused backends are inert."""
    backends: dict[str, IsolationBackend] = {}
    backends["nsjail"] = NsjailBackend(cfg.nsjail_executable)
    backends["docker"] = DockerBackend(
        docker_executable=cfg.docker_executable,
        default_image=cfg.docker_default_image,
        mem_limit=cfg.docker_mem_limit,
        cpu_limit=cfg.docker_cpu_limit,
        pids_limit=cfg.docker_pids_limit,
        network=cfg.docker_network,
    )
    if cfg.allow_unsafe_subprocess:
        backends["subprocess"] = SubprocessBackend()
    return backends


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at server startup (before yield) and shutdown (after yield).
    Creates the SandboxManager and stores it on app.state for handlers to access."""
    cfg = get_config()
    profiles = default_profiles(cfg)

    if cfg.enabled_profiles:
        unknown = sorted(set(cfg.enabled_profiles) - set(profiles))
        if unknown:
            raise ValueError(f"unknown SANDBOX_ENABLED_PROFILES: {unknown}")
        profiles = {name: profiles[name] for name in cfg.enabled_profiles}

    # Dev mode: swap all backends to subprocess (no nsjail/Docker needed).
    if cfg.allow_unsafe_subprocess:
        profiles = {
            name: replace(profile, trusted_subprocess=True, backend_type="subprocess", python_executable="python3")
            for name, profile in profiles.items()
        }

    executor = ThreadPoolExecutor(
        max_workers=cfg.thread_pool_workers,
        thread_name_prefix="sandbox-io",
    )
    asyncio.get_running_loop().set_default_executor(executor)

    backends = _build_backends(cfg)
    manager = SandboxManager(
        profiles,
        backends,
        cfg.workspace_root,
        lease_check_interval=cfg.lease_check_interval,
    )
    # Stash on app.state so request handlers can retrieve it via _manager(request).
    app.state.sandbox_manager = manager
    app.state.sandbox_executor = executor
    await manager.start()
    try:
        yield
    finally:
        await manager.close()
        executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)


# Bearer-token auth. If SANDBOX_API_TOKEN is set, every endpoint except
# /v1/health requires `Authorization: Bearer <token>`. If unset, no auth
# (preserves localhost dev). /v1/health stays open so monitors can probe.
@app.middleware("http")
async def require_api_token(request: Request, call_next):
    token = os.getenv("SANDBOX_API_TOKEN", "")
    if token and request.url.path != "/v1/health":
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or header[7:] != token:
            return JSONResponse(status_code=401, content={"detail": "missing or invalid bearer token"})
    return await call_next(request)


def _manager(request: Request) -> SandboxManager:
    """Retrieve the manager from app.state. Defensive getattr because
    app.state is untyped — a typo would be a silent None, not a compile error."""
    manager = getattr(request.app.state, "sandbox_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="sandbox manager is not started")
    return manager


def _not_found(exc: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# v1 API — daemon lifecycle
#
# Path parameters (e.g., {trial_id}) are matched to function args by name.
# Pydantic-typed args (e.g., req: SessionCreate) are parsed from JSON body.
# request: Request is the raw FastAPI request (used to access app.state).
# response_model validates and serializes the return value.
# ---------------------------------------------------------------------------


@app.get("/v1/health", response_model=HealthView)
async def v1_health(request: Request):
    manager = _manager(request)
    available = [name for name, b in manager.backends.items() if b.available]
    return HealthView(
        status="ok" if available else "degraded",
        backends=available,
        profiles=sorted(manager.profiles),
    )


@app.get("/v1/capacity", response_model=list[CapacityView])
async def capacity(request: Request):
    return _manager(request).capacity()


@app.post("/v1/sessions", response_model=SessionView)
async def create_session(req: SessionCreate, request: Request):
    try:
        session = await _manager(request).create_session(req.profile, req.metadata)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SessionView(
        session_id=session.session_id,
        profile=session.profile.name,
        profile_version=session.profile.version,
        created_at=session.created_at,
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    try:
        await _manager(request).destroy_session(session_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    return {"ok": True}


@app.post("/v1/sessions/{session_id}/trials", response_model=TrialView)
async def create_trial(session_id: str, req: TrialCreate, request: Request):
    # 429 on capacity exhaustion — client can retry or back off.
    try:
        trial = await _manager(request).create_trial(
            session_id,
            req.task_id,
            req.metadata,
            req.lease_seconds,
        )
    except KeyError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return TrialView(
        trial_id=trial.trial_id,
        session_id=trial.session_id,
        task_id=trial.task_id,
        state=trial.state,
        lease_deadline=trial.lease_deadline,
    )


@app.post("/v1/trials/{trial_id}/heartbeat", response_model=TrialView)
async def heartbeat(trial_id: str, request: Request):
    try:
        trial = await _manager(request).heartbeat(trial_id)
    except KeyError as exc:
        raise _not_found(exc) from exc
    return TrialView(
        trial_id=trial.trial_id,
        session_id=trial.session_id,
        task_id=trial.task_id,
        state=trial.state,
        lease_deadline=trial.lease_deadline,
    )


@app.post("/v1/trials/{trial_id}/actions", response_model=ActionResponse)
async def execute_action(trial_id: str, req: ActionRequest, request: Request):
    try:
        return await _manager(request).execute_action(trial_id, req.tool, req.args, req.timeout)
    except KeyError as exc:
        raise _not_found(exc) from exc


@app.post("/v1/trials/{trial_id}/reward", response_model=RewardView)
async def compute_reward(trial_id: str, req: RewardRequest, request: Request):
    manager = _manager(request)
    try:
        result = await manager.compute_reward(
            trial_id,
            answer=req.answer,
            timeout=req.timeout,
        )
    except KeyError as exc:
        raise _not_found(exc) from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return RewardView(reward=result.reward, passed=result.passed, details=result.details)


@app.delete("/v1/trials/{trial_id}")
async def delete_trial(trial_id: str, request: Request):
    await _manager(request).destroy_trial(trial_id)
    return {"ok": True}
