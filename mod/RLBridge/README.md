# RLBridge (SMAPI mod)

RLBridge turns **your own installed copy** of Stardew Valley 1.6.x (SMAPI 4.x, .NET 6) into a
reinforcement-learning environment. It speaks protocol v1 from [`docs/BRIDGE_PROTOCOL.md`](../../docs/BRIDGE_PROTOCOL.md)
over local TCP. The Rust client is [`crates/sdv-bridge`](../../crates/sdv-bridge). The mod only uses SMAPI's
public API and the game's public types, and it never copies or redistributes game code or data.

> **Status: never compiled against the real game.** The .NET SDK and the game aren't installed in
> the environment where this was written. Here is what was checked:
> * All sources compile cleanly (no warnings, C# 10, `net6.0`) against hand-written **stubs** of the
>   game and SMAPI API surface the mod uses.
> * An end-to-end harness ran the real `ModEntry` with a simulated game loop and drove it with the
>   real `sdv_bridge::SdvClient`. It covered handshake, lockstep pause, exact tick counts per `step`,
>   macros answered while paused, errors, recording to `.jsonl.gz`, and lock release on disconnect.
>
> The table at the end lists which game members were checked against official or open-source code,
> and which ones are still **UNVERIFIED**. Every unverified member is called by reflection
> (`GameCompat.cs`), so a wrong guess can't break the build. Instead, that one feature fails at run
> time with a readable message.

## Install & build

1. Install [SMAPI](https://smapi.io) 4.x into your game. Use 4.4.0 or later for the best input
   handling (see *Input modes*).
2. Install the .NET 6 SDK or newer.
3. Run `cd mod/RLBridge && dotnet build`.
   [`Pathoschild.Stardew.ModBuildConfig`](https://www.nuget.org/packages/Pathoschild.Stardew.ModBuildConfig)
   4.4.0 auto-detects the game folder and copies the built mod to `<game>/Mods/RLBridge/`. If it
   can't find the game, add `<GamePath>/path/to/Stardew Valley</GamePath>` to `RLBridge.csproj`.
   If the build fails on a game member, check the **UNVERIFIED** rows below first.
4. Start the game through SMAPI. The console should show `[RLBridge] Listening on 127.0.0.1:7777.`
   On first run, SMAPI writes `Mods/RLBridge/config.json` with these defaults:

| key | default | meaning |
|---|---|---|
| `Host` / `Port` | `127.0.0.1` / `7777` | listen address. The protocol has no authentication, so keep it on loopback. |
| `WindowRadius` | `6` | r for the (2r+1)² passability window |
| `InputMode` | `auto` | `auto` (`press` if available, else `direct`), `press` or `direct` (see *Input modes*) |
| `DisablePauseWhenUnfocused` | `true` | turns off the game's pause-on-unfocus option after a save loads |
| `AllowWarpMacro` | `true` | allows the `warp` teleport macro |
| `MacroTimeoutTicks` | `1500` | tick budget for `warp` and `goto` (~25 s; the Rust client times out reads after 30 s) |
| `SlowMacroTimeoutSeconds` | `28` | wall-clock budget for `sleep` and `load_save`, which include saving or loading time |
| `LockstepIdleTimeoutSeconds` | `600` | releases a lockstep pause after this long with no command (`0` = never) |
| `RecordEveryTicks` | `15` | recording sample period. A line is written only if something changed. |

## Protocol summary (v1)

Newline-delimited JSON over TCP, one client at a time. Each request is `{"id": n, "cmd": ...}` and
gets one response line with the same `id`. Errors look like `{"id": n, "ok": false, "error": "..."}`.

| cmd | fields | response |
|---|---|---|
| `hello` | – | `{"ok", "protocol": 1, "game_version", "smapi_version", "mod_version", "input_mode", "world_ready"}` |
| `observe` | – | `{"ok": true, "obs": Snapshot}` |
| `step` | `ticks` 1–600, `input` InputFrame | holds the input for `ticks` update ticks, then replies `{"ok", "obs", "events", "detail"}` |
| `macro` | `name`, `args` | `{"ok", "obs", "events", "detail"}`. `ok: false` with `obs` means the macro ran but failed. |
| `lockstep` | `enabled` | `{"ok": true}` |
| `record` | `enabled`, `path` | `{"ok": true, "path"}` (stopping also returns `lines`) |
| `load_save` | `save_name`, a folder in the Saves directory such as `Farm_123456789` | `{"ok": true, "obs"}` |

**Lockstep.** While lockstep is on, the game thread blocks at the end of every `UpdateTicked` until a
command arrives. `hello`, `observe`, `lockstep`, `record` and the instant macros (`face`,
`select_item`, `deliver_bundles`) are answered while the game stays paused. `step`, the multi-tick
macros (`warp`, `goto`, `sleep`) and `load_save` let the game run until they finish, then reply and
pause again. So a `step` of N ticks advances exactly N update ticks. The lock is released when the
client disconnects, or after `LockstepIdleTimeoutSeconds` with no command. SMAPI raises no
update-tick events while the game is saving or loading, so the mod never blocks during either.

**InputFrame** is `{"move": "up|down|left|right|null", "use_tool": bool, "action": bool,
"select_slot": int|null, "face": 0-3|null}`. Movement is held on every tick of the step. The other
fields fire once, on the step's first tick. Directions are 0 up, 1 right, 2 down, 3 left.

**Snapshot** has exactly the shape in the protocol doc, and every field is present even on the
title screen (with empty or zero values):
* calendar: `tick` (SMAPI's tick counter), `day`, `season`, `year`, `time`
* weather: `weather` and `weather_tomorrow`, one of `sunny|rain|storm|snow|windy|green_rain`
* `player`: location, tile, facing, stamina, max_stamina, money, current_slot, items with qualified
  IDs, and skills
* `location`: name, width, height, radius, and `passable`, a row-major `0`/`1` window centred on
  the player
* `community_center`: `complete`, `areas_complete[6]`, and `bundles`. Each bundle has `key`,
  `name`, `area`, `items[{id, qty, quality, done}]`, `need` and `complete`. Bundles of the
  Abandoned Joja Mart are left out.
* `menu`: the type name of the active menu, or `null`
* `event_active`

**Events** are emitted between replies: `warp` (`from`, `to`, `tile`, `teleport`; `teleport` is
true for the `warp` macro), `menu_open` / `menu_close` (`menu`), `day_end`, `bundle_delivery`, and
`purchase` / `harvest` (`item`, `qty`; see the limitations).

### Macros

| name | args | behaviour |
|---|---|---|
| `warp` | `location`, `x`, `y` | `Game1.warpFarmer`. Waits until the player arrives. Flagged `teleport: true`. |
| `goto` | `location`, optional `x`, `y` | Walks there with the game's `PathFindController`, one location at a time across the warp and door graph (touch warps, map-edge warps, door tile actions, farm building doors). Re-plans after each warp. |
| `face` | `direction` 0–3 | `Farmer.faceDirection` |
| `select_item` | `item_id` (`(O)24`; unqualified IDs get the `(O)` prefix) | selects the first slot holding the item |
| `deliver_bundles` | optional `anywhere` (default `false`: you must be in the Community Center), optional `pay_vault` (default `false`) | Donates every held stack that fills an open slot of an incomplete bundle (enough items at high enough quality), until each bundle has its `need`. Then flags finished bundles and rooms. |
| `sleep` | – | Ends the day. Waits for the next morning and clicks OK on the shipping summary. |

### Recording

`{"cmd": "record", "enabled": true, "path": "recordings/run1.jsonl.gz"}` appends gzipped JSONL. A
relative path is resolved inside the mod folder, and no path gives an auto-named file. The output is:
1. a header line: `{"kind": "header", "format": "rlbridge-record/1", "protocol", "game_version",
   "mod_version", "save", "started_utc", "tick", "every_ticks"}`
2. then, every `RecordEveryTicks` ticks (or as soon as an event happens), a line
   `{"tick", "obs": Snapshot, "input": InputFrame, "events": [...]}`, but only if the snapshot
   (ignoring `tick`), the input or the events changed.

`input` is inferred from the keys the human is holding:
* `move` comes from the bound movement keys.
* `use_tool` and `action` come from their bound keys or a mouse click.
* `select_slot` is set when the toolbar slot changes.

Appending to an existing file adds a new gzip member, so readers must support multi-member gzip
(for example `flate2::read::MultiGzDecoder` or Python's `gzip`). The stream is flushed every 30
lines.

## Connecting from Rust (`crates/sdv-bridge`)

```rust
use sdv_bridge::{InputFrame, SdvClient, SdvEnv};
use serde_json::json;

// Episode API: connects, sends hello (checks protocol 1), enables lockstep, observes.
let mut env = SdvEnv::connect("127.0.0.1:7777", "Farm_123456789")?;
env.reset()?; // load_save: back to title (if needed) and load the start save
let down = InputFrame { r#move: Some("down".into()), ..Default::default() };
let (reward, done, obs) = env.step(&down)?; // 15 ticks per primitive step by default
let (reward, done, obs) = env.step_macro("goto", json!({"location": "CommunityCenter"}))?;
let (reward, done, obs) = env.step_macro("deliver_bundles", json!({}))?;

// Low-level client: recording a human playing.
let (mut client, _hello) = SdvClient::connect("127.0.0.1:7777", std::time::Duration::from_secs(30))?;
client.request(json!({"cmd": "record", "enabled": true, "path": "recordings/human1.jsonl.gz"}))?;
```

## Input modes

* **press** (the default on SMAPI 4.4.0+): uses
  [`helper.Input.Press(SButton)`](https://github.com/Pathoschild/SMAPI/blob/develop/src/SMAPI/IInputHelper.cs).
  SMAPI documents this as equivalent to the player physically pressing the button for one input
  tick. The mod presses the first keyboard binding for move up/down/left/right, use tool and action
  (defaults W/S/A/D, C, X), so the game's own input code moves the player, swings tools, opens
  doors, talks to NPCs and so on. When a step ends, the mod calls `Suppress` on those keys so the
  hold doesn't spill into the next tick. `Press` is found by reflection, so the mod still builds
  and runs on SMAPI 4.0–4.3.
* **direct** (the fallback):
  * movement uses `Farmer.SetMovingUp/Right/Down/Left(true)` on every tick, then `Halt()`
  * `action` uses `GameLocation.checkAction` on the faced tile
  * `use_tool` uses the UNVERIFIED `Game1.pressUseToolButton()`

`action` and the tool act on the **faced** tile, so face the target first (with `face` in the
InputFrame or the `face` macro).

## Verification table

"Verified" means the exact member, and usually the exact call form, appears in the SMAPI source,
in MIT-licensed mods (Pathoschild/StardewMods, CJBok/SDV-Mods, JoeStrout/Farmtronics), or on
nuget.org. No code was copied from them. The Stardew Valley wiki's Modding pages returned HTTP 403
(bot protection) during development, so nothing below relies on the wiki. No decompiled game code
was consulted.

| API | status | where verified / notes |
|---|---|---|
| `IModHelper.ReadConfig`, `DirectoryPath`; events `GameLoop.UpdateTicking/UpdateTicked/SaveLoaded/DayStarted/DayEnding/ReturnedToTitle`, `Player.Warped/InventoryChanged`, `Display.MenuChanged`; `UpdateTickedEventArgs.Ticks`; `ItemStackSizeChange` | VERIFIED | SMAPI source |
| `IInputHelper.Press(SButton)` (added in SMAPI 4.4.0), `Suppress`, `IsDown`; override timing (applied before the game update, re-applied once on the next input update) | VERIFIED | SMAPI `IInputHelper.cs`, `SInputState.cs`, `SCore.cs`, release notes |
| `Constants.ApiVersion/SavesPath/SaveFolderName`, `Context.IsWorldReady/IsMultiplayer`, `Mod.Dispose(bool)`, `IMonitor.LogOnce`, `Game1.version` | VERIFIED | SMAPI source |
| SMAPI skips `UpdateTicking`/`UpdateTicked` while the game is loading or saving | VERIFIED | SMAPI `SCore.cs` |
| `Pathoschild.Stardew.ModBuildConfig` 4.4.0 | VERIFIED | nuget.org (latest stable, 13 Apr 2025) |
| `Game1.player.Tile`, `TilePoint`, `FacingDirection`, `Position` | VERIFIED | TractorMod, DebugMode, Lookup Anything |
| `Game1.timeOfDay`, `dayOfMonth`, `season` (`Season` enum), `year`; `Game1.currentSeason` is a string (not used) | VERIFIED | Content Patcher, TractorMod, SMAPI 1.6 facades |
| Weather: `netWorldState.Value.GetWeatherForLocation(ctx)?.weather?.Value`, `GameLocation.GetLocationContextId()`, `Game1.weatherForTomorrow`, `netWorldState.Value.WeatherForTomorrow`, `Game1.getWeatherModificationsForDate`, the `Game1.weather_*` constants, `WorldDate.Now()/ForDaysPlayed()/TotalDays` | VERIFIED | Content Patcher, CJB Cheats Menu. (`Game1.isRaining` isn't used: weather comes from the world state.) |
| `Farmer.stamina`, `MaxStamina`, `Money`, `Items[i]`/`Items.Count`, `CurrentToolIndex`, `FarmingLevel/FishingLevel/ForagingLevel/MiningLevel/CombatLevel`, `mailReceived.Contains/Add`, `removeItemFromInventory`, `swimming.Value`, `changeOutOfSwimSuit()` | VERIFIED | CJB, TractorMod, Content Patcher, CentralStation |
| `Item.QualifiedItemId`, `Name`, `Stack` (set); `Object.Quality`, `Category`, `bigCraftable.Value` | VERIFIED | Lookup Anything, TractorMod |
| `GameLocation.isTilePassable(Vector2)`, `IsTileBlockedBy(tile, ignorePassables: CollisionMask.All)`, `doesTileHaveProperty(x, y, prop, layer)`, `Map.Layers[0].LayerWidth/LayerHeight`, `NameOrUniqueName`, `buildings`, `checkAction(Location, Game1.viewport, Farmer)` | VERIFIED | Data Layers, TractorMod, Farmtronics, CropsAnytimeAnywhere |
| `Game1.activeClickableMenu`, `eventUp`, `dialogueUp`, `fadeToBlack`, `exitActiveMenu()`, `getLocationFromName`, `MasterPlayer`, `IsMasterGame` | VERIFIED | SMAPI `Context`, CJB, FastAnimations |
| Bundles: `netWorldState.Value.BundleData` (key `Area/id`; value fields `[0]` name, `[1]` reward, `[2]` "id count quality" triplets), `CommunityCenter.bundles.TryGetValue(id, out bool[])`, `isBundleComplete(id)`, `areAllAreasComplete()`, `areasComplete.Length`/`[i]` (get/set), `CommunityCenter.AREA_*`, room flags `ccPantry/ccCraftsRoom/ccFishTank/ccBoilerRoom/ccVault/ccBulletin`, `ccDoorUnlock`, `JojaMember`, `ccIsComplete`, `hasCompletedCommunityCenter()`, `netWorldState.Value.BundleRewards` (exists) | VERIFIED | Lookup Anything, CJB BundlesCheat, Content Patcher, SMAPI `regenerate_bundles` command |
| Bundle data field `[4]` = number of items required (else all) | **UNVERIFIED** | |
| `cc.bundles[id] = bool[]` stores and syncs the donated flags | **UNVERIFIED** | The setter compiles by analogy with other net dictionaries. `deliver_bundles` reads the flags back and returns `ok: false` if they didn't stick. |
| `netWorldState.Value.BundleRewards[id] = true` makes the bundle reward collectible | **UNVERIFIED** | |
| Setting `areasComplete[i]` plus the room's mail flag restores the room (as CJB does), without the Junimo cutscene | **UNVERIFIED** (behaviour) | |
| `Game1.warpFarmer(string, int, int, bool)` | VERIFIED | CJB WarpCheat |
| `Farmer.SetMovingUp/Right/Down/Left(bool)`, `Halt()`, `faceDirection(int)`, `setMovingInFacingDirection()` | VERIFIED | Farmtronics, DebugMode |
| In direct mode, the game's keyboard handler leaves those movement flags alone | **UNVERIFIED** (behaviour) | |
| `InputButton.key/mouseLeft/mouseRight`; `Game1.options.moveUp/Down/Left/RightButton`, `useToolButton`, `actionButton` (`InputButton[]`) | VERIFIED | SMAPI `SButton.cs`, Pathoschild |
| The keyboard tool key aims at the faced tile, not the mouse cursor | **UNVERIFIED** (behaviour) | |
| `StardewValley.Pathfinding.PathFindController(Character, GameLocation, Point, int, bool)` | VERIFIED | SMAPI `PathFindControllerFacade` |
| `Character.controller` (assign and poll) and `PathFindController.pathToEndPoint` | **UNVERIFIED** | reflection, `GameCompat` |
| `GameLocation.warps` | **UNVERIFIED** | reflection. `Warp.X/Y/TargetName` are VERIFIED (Lookup Anything). |
| Door tile actions `Warp`, `LockedDoorWarp`, `WarpCommunityCenter`: names verified (Data Layers); argument layout `<x> <y> <location> ...` | **UNVERIFIED** | |
| `Building.HasIndoors()`, `humanDoor`, `tileX/tileY.Value`, `GetIndoorsName()` | VERIFIED | Data Layers, Lookup Anything |
| `Game1.pressUseToolButton()` (direct mode only) | **UNVERIFIED** | reflection |
| `Game1.NewDay(float)` for `sleep` | **UNVERIFIED** | reflection |
| `ShippingMenu.okClicked()` | **UNVERIFIED** | reflection |
| `Game1.ExitToTitle()`, then `SaveGame.Load(folder)` + `Game1.exitActiveMenu()` (`load_save`, mirroring a slot click in the load menu) | **UNVERIFIED** | reflection |
| `Options.pauseWhenOutOfFocus` | **UNVERIFIED** | reflection |

## Known limitations

* **Single player only.** Split-screen and multiplayer aren't handled, and `sleep` and `load_save`
  refuse to run in them.
* **The window is frozen while lockstep blocks.** Nothing is drawn, and the OS may mark the window as
  not responding. Disconnecting, sending `lockstep false`, or the idle timeout releases it.
* **Client timeout.** `goto`, `warp`, `sleep` and `load_save` must finish within the client's 30 s
  read timeout, so they are capped by `MacroTimeoutTicks` / `SlowMacroTimeoutSeconds`. A `goto`
  that runs out of time returns `ok: false` with progress in `detail`; call it again to continue.
* **`goto` is best effort.** It walks only, and only through warps and doors it can parse. It
  doesn't handle opening hours, locked doors, mine ladders, minecarts or boats. A door that opens a
  dialogue makes it fail.
* **`sleep` uses `Game1.NewDay`** (UNVERIFIED) instead of walking to the bed. Level-up and
  profession menus at night are not dismissed automatically. If one is still open at the timeout,
  `sleep` returns `ok: false` with the menu name.
* **`deliver_bundles` writes the bundle and room flags directly.** There is no Junimo animation or
  room-restoration cutscene. It checks `ccDoorUnlock` and Joja membership, but not which rooms'
  notes are unlocked. Vault (gold) bundles are paid only with `pay_vault: true`.
* **Approximate events.**
  * `purchase` means an item was gained while a `ShopMenu` was open.
  * `harvest` means any other item gained: forage, crops, drops or chests.
  * `ship` is not emitted.
* **Tool targeting may follow the mouse.** If the mouse cursor is visible and was just moved, the
  game may aim tools at it. Keep the mouse still or outside the window.
* **No authentication.** Anything that can reach the port can control the game and write the
  recording paths it asks for, so keep `Host` on `127.0.0.1`.
