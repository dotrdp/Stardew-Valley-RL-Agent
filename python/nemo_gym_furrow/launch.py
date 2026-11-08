"""Run NeMo Gym's servers for Furrow offline from one prepared venv.

Gym 0.7.0 starts every server with ``<uv_venv_dir>/<type>/<name>/.venv/bin/python app.py`` and, with
``skip_venv_if_present=true``, reuses that venv instead of creating it and ``uv pip install``-ing the
server's requirements (``nemo_gym/cli/setup_command.py``). :func:`prepare_shared_venvs` points each of
those paths at one existing venv (a symlink), so no network or per-server install is needed.

    python -m nemo_gym_furrow.launch venvs --root /tmp/furrow-gym-venvs   # prints the Gym overrides
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

SEARCH_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = "resources_servers/furrow/configs/furrow.yaml"  # relative to SEARCH_ROOT
SERVERS: tuple[tuple[str, str], ...] = (
    ("resources_servers", "furrow"),
    ("responses_api_agents", "simple_agent"),
    ("environment_servers", "legacy_agent"),
    ("responses_api_models", "openai_model"),
    ("responses_api_models", "vllm_model"),
)


def prepare_shared_venvs(
    root: Path, venv: Optional[Path] = None, servers: Iterable[tuple[str, str]] = SERVERS
) -> list[str]:
    """Symlink ``root/<type>/<name>/.venv -> venv`` (default: this interpreter's venv) and return the
    Gym overrides that make Gym use them."""
    target = Path(venv or sys.prefix).resolve()
    if not (target / "bin" / "python").exists():
        raise FileNotFoundError(f"{target} is not a venv (no bin/python)")
    for kind, name in servers:
        link = root / kind / name / ".venv"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.resolve() == target:
                continue
            raise FileExistsError(f"{link} exists and is not a link to {target}")
        link.symlink_to(target, target_is_directory=True)
    return gym_overrides(root)


def gym_overrides(root: Path) -> list[str]:
    return ["+skip_venv_if_present=true", f"+uv_venv_dir={Path(root).resolve()}"]


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("venvs", help="link Gym's per-server venv paths to one venv; print the overrides")
    v.add_argument("--root", type=Path, required=True)
    v.add_argument("--venv", type=Path, default=None, help="venv to share (default: the running one)")
    a = p.parse_args(argv)
    if a.cmd == "venvs":
        print(" ".join(prepare_shared_venvs(a.root, a.venv)))


if __name__ == "__main__":
    main()
