"""End to end through NeMo Gym's real harness, offline.

Starts the fake OpenAI-compatible model, then ``gym env start`` (Furrow resources server, simple_agent,
legacy_agent environment server, built-in ``openai_model`` model server pointed at the fake), then
``gym eval run --no-serve`` on 3 generated tasks, and checks the rollouts and aggregate metrics.
Every Gym server runs from this interpreter's venv via ``skip_venv_if_present`` (no network).
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

pytest.importorskip("furrow")
pytest.importorskip("nemo_gym")

from nemo_gym_furrow import tasks  # noqa: E402
from nemo_gym_furrow.launch import CONFIG_PATH, SEARCH_ROOT, prepare_shared_venvs  # noqa: E402

GYM = Path(sys.executable).parent / "gym"
READY = re.compile(r"All (\d+) / \1 servers ready")
N_TASKS = 3


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def tail(path: Path, n: int = 60) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


def wait_for(predicate, timeout: float, proc: subprocess.Popen, what: str, log: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if proc.poll() is not None:
            pytest.fail(f"{what} exited with {proc.returncode} before it was ready:\n{tail(log)}")
        time.sleep(0.5)
    pytest.fail(f"{what} not ready after {timeout:.0f}s:\n{tail(log)}")


def http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def stop(proc: subprocess.Popen, group: bool) -> None:
    if proc.poll() is not None:
        return
    try:
        if group:
            os.killpg(proc.pid, signal.SIGINT)  # Gym shuts its servers and its Ray cluster down on SIGINT
        else:
            proc.terminate()
        proc.wait(timeout=60)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(proc.pid, signal.SIGKILL) if group else proc.kill()
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


@pytest.mark.skipif(not GYM.exists(), reason=f"NeMo Gym CLI not found at {GYM}")
def test_rollouts_through_the_nemo_gym_harness(tmp_path: Path) -> None:
    tasks.generate(
        tmp_path / "tasks", {"val": N_TASKS}, days_per_seed=1, horizon_days=1, max_tool_calls=20, min_expert_days=0.25
    )
    overrides = prepare_shared_venvs(tmp_path / "venvs")
    fake_port, head_port = free_port(), free_port()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAY_ADDRESS", "NEMO_GYM_"))}
    env.update(
        PYTHONPATH=os.pathsep.join([str(SEARCH_ROOT.parent), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
        CUDA_VISIBLE_DEVICES="",
    )
    fake_log, gym_log, collect_log = tmp_path / "fake.log", tmp_path / "gym.log", tmp_path / "collect.log"
    fake = servers = None
    try:
        with fake_log.open("w") as f:
            fake = subprocess.Popen(
                [sys.executable, "-m", "nemo_gym_furrow.fake_openai", "--port", str(fake_port)],
                cwd=tmp_path, env=env, stdout=f, stderr=subprocess.STDOUT,
            )  # fmt: skip
        wait_for(lambda: http_ok(f"http://127.0.0.1:{fake_port}/v1/models"), 60, fake, "fake model", fake_log)

        with gym_log.open("w") as f:
            servers = subprocess.Popen(
                [
                    str(GYM), "env", "start",
                    "--search-dir", str(SEARCH_ROOT),
                    "--config", CONFIG_PATH,
                    "--model-type", "openai_model",
                    "--model-url", f"http://127.0.0.1:{fake_port}/v1",
                    "--model-api-key", "dummy",
                    "--model", "furrow-fake-random",
                    *overrides,
                    f"+head_server.port={head_port}",
                ],
                cwd=tmp_path, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
            )  # fmt: skip
        wait_for(lambda: READY.search(gym_log.read_text(errors="replace")), 300, servers, "gym env start", gym_log)

        out = tmp_path / "results" / "rollouts.jsonl"
        with collect_log.open("w") as f:
            done = subprocess.run(
                [
                    str(GYM), "eval", "run", "--no-serve",
                    "--agent", "furrow_simple_agent",
                    "--input", str(tmp_path / "tasks" / "val.jsonl"),
                    "--output", str(out),
                    "--num-repeats", "1",
                    f"+head_server.port={head_port}",
                ],
                cwd=tmp_path, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600,
            )  # fmt: skip
        assert done.returncode == 0, f"gym eval run failed:\n{tail(collect_log)}\n--- servers ---\n{tail(gym_log)}"

        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        assert len(rows) == N_TASKS
        for r in rows:
            assert r["mask_sample"] is False and r["failure_kind"] is None, r.get("failure_reason")
            assert 0.0 <= r["reward"] < 1.0
            assert (r["reward"] > 0.5) == r["success"]
            assert r["end_reason"] in {"success", "horizon", "tool_budget", "env_finished"}
            assert r["tool_calls"] > 0 and r["macros_executed"] > 0
            calls = [o["name"] for o in r["response"]["output"] if o["type"] == "function_call"]
            assert calls[:3] == ["observe", "list_actions", "act"]
            assert len(calls) == r["tool_calls"]
        agg = json.loads((tmp_path / "results" / "rollouts_aggregate_metrics.json").read_text())
        metrics = agg[0]["agent_metrics"]
        assert "success_rate" in metrics and metrics["mean/reward"] == pytest.approx(
            sum(r["reward"] for r in rows) / N_TASKS
        )
    finally:
        if servers is not None:
            stop(servers, group=True)
        if fake is not None:
            stop(fake, group=False)
