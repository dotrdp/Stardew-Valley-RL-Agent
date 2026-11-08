# RLBridge protocol v1 (Stardew Valley ↔ agent)

The SMAPI mod `mod/RLBridge` turns the player's **own installed copy** of Stardew Valley (1.6.x,
SMAPI 4.x) into an RL environment. It never redistributes game code or data; it reads and drives
the running game through SMAPI's public modding API. The Rust client is `crates/sdv-bridge`
(`SdvClient`, `SdvEnv`, `MockServer`).

## Transport

TCP on `127.0.0.1:7777` (configurable in the mod's `config.json`). One client at a time.
Newline-delimited JSON (UTF-8): every request is one line, every response is one line.
Requests carry an integer `id`; the response echoes it.

```json
{"id": 1, "cmd": "hello"}
{"id": 1, "ok": true, "protocol": 1, "game_version": "1.6.15", "smapi_version": "4.1.10", "mod_version": "0.1.0"}
```

Errors: `{"id": n, "ok": false, "error": "message"}`.

## Lockstep

`{"cmd": "lockstep", "enabled": true}` — while enabled, the mod **blocks the game's update thread**
at the end of every update tick (in `GameLoop.UpdateTicked`) until it receives a `step`
(or `observe` / `macro`, which are answered while paused). The game is therefore frozen between
agent decisions and advances exactly the requested number of ticks per `step`.
`{"cmd": "lockstep", "enabled": false}` resumes free running (human play / recording).
The mod releases the lock if the client disconnects.

## Commands

| cmd | fields | effect | response |
|---|---|---|---|
| `hello` | – | handshake | versions |
| `observe` | – | – | `{"ok": true, "obs": Snapshot}` |
| `step` | `ticks` (1–600), `input` (InputFrame) | hold `input` for `ticks` update ticks (≈ 60 ticks per real second, ≈ 7 s of game time per 10 game minutes) | `{"ok": true, "obs": Snapshot, "events": [Event]}` |
| `macro` | `name`, `args` | run a macro with the game's own systems (see below) | `{"ok": bool, "obs": Snapshot, "events": [...], "detail": str}` |
| `lockstep` | `enabled` | see above | `{"ok": true}` |
| `record` | `enabled`, `path` | start/stop recording human play to JSONL.gz | `{"ok": true}` |
| `load_save` | `save_name` | load a save slot (episode reset) | `{"ok": true, "obs": Snapshot}` |

### InputFrame

```json
{"move": "up|down|left|right|null", "use_tool": false, "action": false, "select_slot": null, "face": null}
```

### Macros (v1)

| name | args | behaviour |
|---|---|---|
| `warp` | `location`, `x`, `y` | `Game1.warpFarmer` (debug/teleport macro; flagged in events) |
| `goto` | `location` | path to the location's entrance using the game's pathfinding (`PathFindController`) |
| `face` | `direction` (0–3) | face a direction |
| `select_item` | `item_id` (qualified, e.g. `(O)24`) | select the first slot holding the item |
| `deliver_bundles` | – | donate every held item that fits an incomplete Community Center bundle (as the bundle menu would) |
| `sleep` | – | end the day as when going to bed |

## Snapshot

```json
{
  "tick": 123456, "day": 5, "season": "spring", "year": 1, "time": 1130,
  "weather": "sunny", "weather_tomorrow": "rain",
  "player": {"location": "Farm", "tile": [64, 15], "facing": 2, "stamina": 260.0, "max_stamina": 270,
             "money": 500, "current_slot": 0,
             "items": [{"slot": 0, "id": "(T)Hoe", "name": "Hoe", "stack": 1, "quality": 0}],
             "skills": {"farming": 0, "fishing": 0, "foraging": 0, "mining": 0, "combat": 0}},
  "location": {"name": "Farm", "width": 80, "height": 65, "radius": 6,
               "passable": "row-major string of '0'/'1' for the (2r+1)^2 window around the player"},
  "community_center": {"complete": false, "areas_complete": [false, false, false, false, false, false],
                       "bundles": [{"key": "Pantry/0", "name": "Spring Crops", "area": "Pantry",
                                    "items": [{"id": "(O)24", "qty": 1, "quality": 0, "done": false}],
                                    "need": 4, "complete": false}]},
  "menu": "null|ShopMenu|DialogueBox|...", "event_active": false
}
```

### Events

`{"kind": "warp|purchase|harvest|ship|bundle_delivery|day_end|menu_open|menu_close", ...}` — emitted
between snapshots; used by the recorder and by the human-trajectory segmenter.

## Recording

With `record` enabled the mod appends one JSON line per decision-relevant change (snapshot diff,
held input, events) to a gzip file. `sdv-bridge`'s converter turns recordings into
`furrow-traj/2` datasets (`env: "sdv"`).
