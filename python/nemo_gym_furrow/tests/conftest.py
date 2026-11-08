from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]  # python/nemo_gym_furrow
PYTHON_ROOT = PKG_ROOT.parent
APP_PATH = PKG_ROOT / "resources_servers" / "furrow" / "app.py"
EXAMPLE_JSONL = PKG_ROOT / "resources_servers" / "furrow" / "data" / "example.jsonl"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def load_app_module():
    """Import resources_servers/furrow/app.py by path (Gym runs it as a script, not a package)."""
    name = "nemo_gym_furrow_resources_app"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def app_module():
    pytest.importorskip("nemo_gym")
    return load_app_module()


@pytest.fixture
def make_server(app_module):
    from unittest.mock import MagicMock

    from nemo_gym.server_utils import ServerClient

    def make(**overrides):
        config = app_module.FurrowResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="app.py", name="furrow", domain="games", **overrides
        )
        return app_module.FurrowResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    return make


@pytest.fixture(scope="session")
def example_rows() -> list[dict]:
    return [json.loads(line) for line in EXAMPLE_JSONL.read_text().splitlines() if line.strip()]
