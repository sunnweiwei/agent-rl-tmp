"""Async HTTP client for the sandbox server.

Lives under ``agents`` so rollout workers do not import the server package.
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp

DEFAULT_REWARD_HTTP_TIMEOUT_SECONDS = 1210.0
REWARD_HTTP_TIMEOUT_BUFFER_SECONDS = 10.0


class SandboxError(Exception):
    pass


class SandboxClient:
    def __init__(self, base_url: str, timeout: float = 60.0, api_token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_token = api_token if api_token is not None else os.getenv("SANDBOX_API_TOKEN", "")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"} if self.api_token else {}

    async def _request(self, method: str, endpoint: str, payload: dict | None = None, timeout: float | None = None):
        client_timeout = aiohttp.ClientTimeout(total=timeout or self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout, headers=self._headers()) as session:
            async with session.request(method, f"{self.base_url}/{endpoint}", json=payload) as response:
                response.raise_for_status()
                return await response.json()

    async def open_session(self, profile: str, metadata: dict | None = None) -> SandboxSession:
        result = await self._request("POST", "v1/sessions", {"profile": profile, "metadata": metadata or {}})
        return SandboxSession(self, result["session_id"], result["profile"])


class SandboxSession:
    def __init__(self, client: SandboxClient, session_id: str, profile: str):
        self.client = client
        self.session_id = session_id
        self.profile = profile

    async def create_trial(
        self, task_id: str = "", metadata: dict | None = None, lease_seconds: float | None = None
    ) -> SandboxTrial:
        result = await self.client._request(
            "POST",
            f"v1/sessions/{self.session_id}/trials",
            {"task_id": task_id, "metadata": metadata or {}, "lease_seconds": lease_seconds},
        )
        return SandboxTrial(self.client, result["trial_id"])

    async def close(self) -> None:
        await self.client._request("DELETE", f"v1/sessions/{self.session_id}")


class SandboxTrial:
    def __init__(self, client: SandboxClient, trial_id: str):
        self.client = client
        self.trial_id = trial_id

    async def heartbeat(self) -> dict:
        return await self.client._request("POST", f"v1/trials/{self.trial_id}/heartbeat", {})

    async def execute(self, tool: str, args: dict, timeout: float = 120.0) -> Any:
        result = await self.client._request(
            "POST",
            f"v1/trials/{self.trial_id}/actions",
            {"tool": tool, "args": args, "timeout": timeout},
            timeout=timeout + 10,
        )
        if not result.get("success"):
            raise SandboxError(result.get("error", "unknown sandbox action error"))
        return result.get("result")

    async def reward(self, answer: str | None = None, timeout: float | None = None) -> dict:
        payload: dict[str, Any] = {}
        if answer is not None:
            payload["answer"] = answer
        if timeout is not None:
            payload["timeout"] = timeout
            http_timeout = timeout + REWARD_HTTP_TIMEOUT_BUFFER_SECONDS
        else:
            http_timeout = max(self.client.timeout, DEFAULT_REWARD_HTTP_TIMEOUT_SECONDS)
        return await self.client._request("POST", f"v1/trials/{self.trial_id}/reward", payload, timeout=http_timeout)

    async def destroy(self) -> None:
        await self.client._request("DELETE", f"v1/trials/{self.trial_id}")
