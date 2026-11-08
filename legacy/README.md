# Legacy HTTP/curl agent (superseded)

This is the first iteration of the project: a Python client that talked to the game through an
HTTP reflection proxy mod (one `curl` subprocess per property read) plus a keyboard-emulation mod.
It is kept for reference only. The current stack is described in `docs/DESIGN.md`:

- the original **Furrow** simulator (`crates/furrow*`) for fast, large-scale training, and
- the **RLBridge** SMAPI mod + Rust client (`mod/RLBridge`, `crates/sdv-bridge`) for the real game.

The old code still runs against the old mods (see git history before this commit for its README).
