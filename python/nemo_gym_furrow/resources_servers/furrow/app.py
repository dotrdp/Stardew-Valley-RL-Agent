"""NeMo Gym resources server for Furrow segment tasks (tools: observe, list_actions, act, status).

Written against nemo-gym 0.7.0 (``nemo_gym.base_resources_server.SimpleResourcesServer``). Per-rollout
state follows ``resources_servers/example_session_state_mgmt``: the session middleware puts a uuid in a
signed cookie (``request.session[SESSION_ID_KEY]``); ``/seed_session`` builds one ``furrow.Env`` for
that id from the dataset row, the tools act on it, and ``/verify`` scores and drops it. The typed
Environment-Server seed/close protocol (``ResourcesSeedSessionRequest``) is supported too: the
session is then keyed by ``resources_session_id``.

Gym starts this file as ``python app.py`` from this directory, so the repo's ``python/`` directory is
put on ``sys.path`` to import ``nemo_gym_furrow``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

_PYTHON_ROOT = str(Path(__file__).resolve().parents[3])
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from nemo_gym.base_resources_server import (  # noqa: E402
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY  # noqa: E402

from nemo_gym_furrow.core import (  # noqa: E402
    DEFAULT_HORIZON_DAYS,
    DEFAULT_MAX_PLAN_LENGTH,
    DEFAULT_MAX_TOOL_CALLS,
    FurrowSession,
    SessionClosedError,
    SessionStore,
    TaskSpec,
)


class FurrowResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED  # reward needs the live session

    session_ttl_seconds: float = 3600.0
    max_sessions: int = 4096
    default_horizon_days: int = DEFAULT_HORIZON_DAYS
    default_max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_plan_length: int = DEFAULT_MAX_PLAN_LENGTH


class FurrowSeedSessionRequest(BaseSeedSessionRequest):
    """Legacy ``/run`` seed: the agent forwards the whole dataset row."""

    model_config = ConfigDict(extra="allow")

    seed: int
    start_day: int
    horizon_days: Optional[int] = None
    max_tool_calls: Optional[int] = None
    env_config: Dict[str, Any] = Field(default_factory=dict)


class ActRequest(BaseModel):
    actions: Union[List[str], str] = Field(
        description="One macro name, or a day plan: several names run in order until the day ends."
    )


class FurrowVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class FurrowVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    success: bool = False
    end_reason: str = "session_lost"
    days_to_success: Optional[float] = None
    expert_days: Optional[float] = None
    expert_success: Optional[bool] = None
    speed_ratio: Optional[float] = None
    progress: float = 0.0
    days_played: float = 0.0
    requests_gained: int = 0
    slots_gained: int = 0
    tool_calls: int = 0
    invalid_actions: int = 0
    macros_executed: int = 0


class FurrowResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: FurrowResourcesServerConfig
    _store: SessionStore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._store = SessionStore(ttl_seconds=self.config.session_ttl_seconds, max_sessions=self.config.max_sessions)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/observe")(self.observe)
        app.post("/list_actions")(self.list_actions)
        app.post("/act")(self.act)
        app.post("/status")(self.status)
        return app

    # ---- sessions ----

    def _spec(self, row: dict) -> TaskSpec:
        try:
            return TaskSpec.from_mapping(
                row,
                horizon_days=self.config.default_horizon_days,
                max_tool_calls=self.config.default_max_tool_calls,
                max_plan_length=self.config.max_plan_length,
            )
        except (TypeError, ValueError) as e:
            raise HTTPException(status_code=422, detail=f"bad Furrow task: {e}") from e

    async def _create(self, session_id: str, spec: TaskSpec) -> None:
        try:
            await run_in_threadpool(self._store.create, session_id, spec)
        except SessionClosedError as e:
            raise HTTPException(status_code=409, detail=f"session {session_id!r} is already closed") from e
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

    async def seed_session(
        self, request: Request, body: Union[ResourcesSeedSessionRequest, FurrowSeedSessionRequest]
    ) -> Union[ResourcesSeedSessionResponse, BaseSeedSessionResponse]:
        if isinstance(body, ResourcesSeedSessionRequest):
            sid = body.resources_session_id
            await self._create(sid, self._spec(dict(body.task_data)))
            request.session[SESSION_ID_KEY] = sid  # tool calls carrying this cookie reach the same session
            return ResourcesSeedSessionResponse(resources_session_id=sid)
        await self._create(request.session[SESSION_ID_KEY], self._spec(body.model_dump()))
        return BaseSeedSessionResponse()

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        self._store.close(body.resources_session_id)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    def _session(self, request: Request) -> FurrowSession:
        session = self._store.get(request.session.get(SESSION_ID_KEY, ""))
        if session is None:
            raise HTTPException(status_code=404, detail="no Furrow session for this rollout (seed_session first)")
        return session

    # ---- tools ----

    async def observe(self, request: Request) -> dict:
        return await run_in_threadpool(self._session(request).observe)

    async def list_actions(self, request: Request) -> dict:
        return await run_in_threadpool(self._session(request).list_actions)

    async def act(self, request: Request, body: ActRequest) -> dict:
        return await run_in_threadpool(self._session(request).act, body.actions)

    async def status(self, request: Request) -> dict:
        return await run_in_threadpool(self._session(request).status)

    # ---- reward ----

    async def verify(self, request: Request, body: FurrowVerifyRequest) -> FurrowVerifyResponse:
        result = await run_in_threadpool(self._store.finish, request.session.get(SESSION_ID_KEY, ""))
        if result is None:
            return FurrowVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                mask_sample=True,
                failure_kind="session_lost",
                failure_reason="no Furrow session for this rollout at verify time",
            )
        return FurrowVerifyResponse(**{**body.model_dump(), **result})

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        rows = [r for task in tasks for r in task]
        wins = [r for r in rows if r.get("success")]
        ratios = [r["speed_ratio"] for r in wins if isinstance(r.get("speed_ratio"), (int, float))]
        return {
            "success_rate": len(wins) / len(rows) if rows else 0.0,
            "mean_speed_ratio_on_success": sum(ratios) / len(ratios) if ratios else None,
            "beats_expert_rate": sum(1 for x in ratios if x > 1.0) / len(rows) if rows else 0.0,
        }

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        keys = ("mean/reward", "success_rate", "mean_speed_ratio_on_success", "mean/invalid_actions")
        return {k: agent_metrics[k] for k in keys if k in agent_metrics}


if __name__ == "__main__":
    FurrowResourcesServer.run_webserver()
