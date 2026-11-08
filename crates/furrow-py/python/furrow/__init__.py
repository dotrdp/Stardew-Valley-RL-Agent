"""Furrow — original farming-sim RL environment (see docs/DESIGN.md §3.7).

This shim re-exports the native extension ``furrow._furrow`` built with maturin.
"""
try:
    from ._furrow import *  # noqa: F401,F403
    from ._furrow import __version__  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "furrow._furrow is not built. Run: maturin develop --release -m crates/furrow-py/Cargo.toml"
    ) from e
