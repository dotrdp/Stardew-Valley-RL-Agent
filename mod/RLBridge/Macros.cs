using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewValley;
using StardewValley.Buildings;
using StardewValley.Locations;
using StardewValley.Menus;
using StardewValley.Pathfinding;
using SObject = StardewValley.Object;

namespace RLBridge
{
    /// <summary>A command that runs over several update ticks before it is answered.</summary>
    internal interface IJob
    {
        /// <summary>Begin the job. Returns an error message to fail immediately, else null.</summary>
        string? Start();

        /// <summary>Called in <c>UpdateTicking</c>, before the game update.</summary>
        void BeforeTick();

        /// <summary>Called in <c>UpdateTicked</c>, after the game update. Returns true when finished.</summary>
        bool AfterTick(out bool ok, out string detail);

        /// <summary>Abort (client disconnected).</summary>
        void Cancel();
    }

    /// <summary><c>step</c>: hold an input frame for N update ticks.</summary>
    internal sealed class StepJob : IJob
    {
        private readonly InputDriver input;
        private readonly InputFrame frame;
        private int remaining;
        private bool firstTick = true;

        public StepJob(InputDriver input, InputFrame frame, int ticks)
        {
            this.input = input;
            this.frame = frame;
            this.remaining = ticks;
        }

        public string? Start() => null;

        public void BeforeTick()
        {
            if (!Context.IsWorldReady)
                return;
            this.input.Apply(this.frame, this.firstTick);
            this.firstTick = false;
        }

        public bool AfterTick(out bool ok, out string detail)
        {
            ok = true;
            detail = "";
            if (--this.remaining > 0)
                return false;
            this.input.Release(this.frame);
            return true;
        }

        public void Cancel()
        {
            this.input.Release(this.frame);
        }
    }

    /// <summary><c>warp</c>: teleport with <c>Game1.warpFarmer(location, x, y, flip)</c> (the call CJB Cheats Menu's warps use, MIT).</summary>
    internal sealed class WarpJob : IJob
    {
        private readonly string location;
        private readonly int x;
        private readonly int y;
        private readonly int maxTicks;
        private readonly Action markTeleport;
        private int ticks;

        public WarpJob(string location, int x, int y, int maxTicks, Action markTeleport)
        {
            this.location = location;
            this.x = x;
            this.y = y;
            this.maxTicks = maxTicks;
            this.markTeleport = markTeleport;
        }

        public string? Start()
        {
            GameLocation? target = Game1.getLocationFromName(this.location);
            if (target == null)
                return $"unknown location '{this.location}'";

            // same preparation as CJB Cheats Menu's warp
            Game1.exitActiveMenu();
            Game1.player.swimming.Value = false;
            Game1.player.changeOutOfSwimSuit();

            this.markTeleport();
            Game1.warpFarmer(this.location, this.x, this.y, false);
            return null;
        }

        public void BeforeTick()
        {
        }

        public bool AfterTick(out bool ok, out string detail)
        {
            ok = false;
            detail = "";
            this.ticks++;
            GameLocation? here = Game1.currentLocation;
            if (here != null && Router.Same(here.NameOrUniqueName, this.location) && !Game1.fadeToBlack && this.ticks > 1)
            {
                Point tile = Game1.player.TilePoint;
                ok = true;
                detail = $"warped to {here.NameOrUniqueName} {tile.X},{tile.Y}";
                return true;
            }
            if (this.ticks >= this.maxTicks)
            {
                detail = $"warp to {this.location} did not complete in {this.ticks} ticks";
                return true;
            }
            return false;
        }

        public void Cancel()
        {
        }
    }

    /// <summary>
    /// <c>goto</c>: walk to another location using the game's pathfinding, hop by hop through the location graph
    /// (touch warps, map-edge warps, door tile actions and farm building doors).
    /// </summary>
    internal sealed class GotoJob : IJob
    {
        private enum Phase { Plan, Walk, Leave }

        private readonly string target;
        private readonly Point? targetTile;
        private readonly InputDriver input;
        private readonly Router router;
        private readonly int maxTicks;

        private Phase phase = Phase.Plan;
        private Exit? exit;
        private string currentLocation = "";
        private int ticks;
        private int phaseTicks;
        private int stuckTicks;
        private Point lastTile;
        private InputFrame? held;
        private bool heldFirstTick;
        private int hops;

        public GotoJob(string target, Point? targetTile, InputDriver input, Router router, int maxTicks)
        {
            this.target = target;
            this.targetTile = targetTile;
            this.input = input;
            this.router = router;
            this.maxTicks = maxTicks;
        }

        public string? Start()
        {
            if (Game1.getLocationFromName(this.target) == null)
                return $"unknown location '{this.target}'";
            this.currentLocation = Game1.currentLocation?.NameOrUniqueName ?? "";
            return null;
        }

        public void BeforeTick()
        {
            if (this.held != null && Context.IsWorldReady)
            {
                this.input.Apply(this.held, this.heldFirstTick);
                this.heldFirstTick = false;
            }
        }

        public bool AfterTick(out bool ok, out string detail)
        {
            ok = false;
            detail = "";
            this.ticks++;
            this.phaseTicks++;

            if (this.ticks > this.maxTicks)
            {
                this.Stop();
                detail = $"timeout after {this.ticks} ticks in {Game1.currentLocation?.NameOrUniqueName} ({this.hops} hops done); call goto again to continue";
                return true;
            }
            if (!Context.IsWorldReady || Game1.currentLocation == null || Game1.fadeToBlack)
                return false; // mid-warp

            // arrived in a new location: re-plan
            string here = Game1.currentLocation.NameOrUniqueName;
            if (here != this.currentLocation)
            {
                this.currentLocation = here;
                this.Stop();
                this.hops++;
                this.phase = Phase.Plan;
                this.phaseTicks = 0;
            }

            // a menu/dialogue/event blocks walking
            if (Game1.activeClickableMenu != null || Game1.dialogueUp || Game1.eventUp)
            {
                if (this.phase == Phase.Leave)
                {
                    this.Stop();
                    detail = $"exit to {this.exit?.Target} opened {Game1.activeClickableMenu?.GetType().Name ?? "a dialogue/event"} (locked door?)";
                    return true;
                }
                return false; // wait; the tick budget bounds this
            }

            switch (this.phase)
            {
                case Phase.Plan:
                    return this.Plan(here, out ok, out detail);

                case Phase.Walk:
                    {
                        bool finished = GameCompat.GetController(Game1.player) == null;
                        Point tile = Game1.player.TilePoint;
                        if (tile != this.lastTile)
                        {
                            this.lastTile = tile;
                            this.stuckTicks = 0;
                        }
                        else
                            this.stuckTicks++;
                        if (!finished && this.stuckTicks < 240)
                            return false;

                        this.Stop();
                        Point goal = this.exit?.Approach ?? this.targetTile ?? tile;
                        if (tile != goal)
                        {
                            detail = $"pathfinding stopped at {tile.X},{tile.Y}, short of {goal.X},{goal.Y} in {here}";
                            return true;
                        }
                        if (this.exit == null)
                        {
                            ok = true;
                            detail = $"arrived at {here} {tile.X},{tile.Y}";
                            return true;
                        }
                        this.BeginExit();
                        return false;
                    }

                default: // Phase.Leave: pushing through an edge, stepping on a touch warp, or waiting for a door to warp us
                    if (this.phaseTicks > 180)
                    {
                        this.Stop();
                        detail = $"exit to {this.exit?.Target} at {this.exit?.Tile.X},{this.exit?.Tile.Y} did not trigger";
                        return true;
                    }
                    return false;
            }
        }

        public void Cancel()
        {
            this.Stop();
        }

        private bool Plan(string here, out bool ok, out string detail)
        {
            ok = false;
            detail = "";
            if (Router.Same(here, this.target))
            {
                if (this.targetTile == null || Game1.player.TilePoint == this.targetTile.Value)
                {
                    Point tile = Game1.player.TilePoint;
                    ok = true;
                    detail = $"arrived at {here} {tile.X},{tile.Y}";
                    return true;
                }
                this.exit = null;
                return !this.StartWalk(this.targetTile.Value, 2, out detail);
            }

            this.exit = this.router.NextExit(Game1.currentLocation, this.target, out string? error);
            if (this.exit == null)
            {
                detail = error ?? "no route";
                return true;
            }
            if (Game1.player.TilePoint == this.exit.Approach)
            {
                this.BeginExit();
                return false;
            }
            return !this.StartWalk(this.exit.Approach, this.exit.Facing, out detail);
        }

        private bool StartWalk(Point goal, int facing, out string detail)
        {
            detail = "";
            // Verified: 1.6 constructor PathFindController(Character, GameLocation, Point endPoint, int finalFacingDirection,
            // bool clearMarriageDialogues), as called by SMAPI's PathFindControllerFacade.
            var controller = new PathFindController(Game1.player, Game1.currentLocation, goal, facing, false);
            if (GameCompat.ControllerHasPath(controller) == false)
            {
                detail = $"no path to {goal.X},{goal.Y} in {Game1.currentLocation.NameOrUniqueName}";
                return false;
            }
            if (!GameCompat.SetController(Game1.player, controller))
            {
                detail = "couldn't assign the player's path controller (UNVERIFIED member Character.controller)";
                return false;
            }
            this.phase = Phase.Walk;
            this.phaseTicks = 0;
            this.stuckTicks = 0;
            this.lastTile = Game1.player.TilePoint;
            return true;
        }

        private void BeginExit()
        {
            Exit e = this.exit!;
            this.phase = Phase.Leave;
            this.phaseTicks = 0;
            switch (e.Kind)
            {
                case ExitKind.Door:
                    // face the door and press the action button once
                    this.held = new InputFrame { Face = e.Facing, Action = true };
                    break;
                case ExitKind.Touch:
                    // standing on/next to a touch warp: keep walking the way we were going until the game warps us
                    this.held = new InputFrame { Move = Game1.player.FacingDirection };
                    break;
                default:
                    // map edge: walk off the edge
                    this.held = new InputFrame { Move = e.Facing };
                    break;
            }
            this.heldFirstTick = true;
        }

        private void Stop()
        {
            if (this.held != null)
            {
                this.input.Release(this.held);
                this.held = null;
            }
            if (Game1.player != null && GameCompat.GetController(Game1.player) != null)
                GameCompat.SetController(Game1.player, null);
        }
    }

    /// <summary><c>sleep</c>: end the day and wait for the next morning.</summary>
    internal sealed class SleepJob : IJob
    {
        private readonly Func<int> dayStartedCount;
        private readonly int timeoutSeconds;
        private readonly Stopwatch clock = new();
        private int startDay;
        private int startCount;
        private int menuTicks;

        public SleepJob(Func<int> dayStartedCount, int timeoutSeconds)
        {
            this.dayStartedCount = dayStartedCount;
            this.timeoutSeconds = timeoutSeconds;
        }

        public string? Start()
        {
            if (Context.IsMultiplayer)
                return "sleep macro is single-player only";
            this.startDay = Snapshot.DaysPlayed();
            this.startCount = this.dayStartedCount();
            Game1.exitActiveMenu();
            // UNVERIFIED: Game1.NewDay(0f) runs the same end-of-day sequence as confirming the bed prompt.
            if (!GameCompat.NewDay(out string error))
                return error;
            this.clock.Start();
            return null;
        }

        public void BeforeTick()
        {
        }

        public bool AfterTick(out bool ok, out string detail)
        {
            ok = false;
            detail = "";
            IClickableMenu? menu = Game1.activeClickableMenu;
            if (this.clock.Elapsed.TotalSeconds > this.timeoutSeconds)
            {
                detail = $"timeout waiting for the next morning (menu: {menu?.GetType().Name ?? "none"})";
                return true;
            }
            if (!Context.IsWorldReady)
                return false;

            // the shipping summary needs an OK click
            if (menu is ShippingMenu)
            {
                if (++this.menuTicks % 120 == 90 && !GameCompat.TryClickOk(menu, out string error))
                    detail = error;
                return false;
            }
            this.menuTicks = 0;

            if (this.dayStartedCount() > this.startCount && Snapshot.DaysPlayed() != this.startDay && menu == null && !Game1.fadeToBlack)
            {
                ok = true;
                detail = $"new day: {Game1.season.ToString().ToLowerInvariant()} {Game1.dayOfMonth}, year {Game1.year}";
                return true;
            }
            return false;
        }

        public void Cancel()
        {
        }
    }

    /// <summary><c>load_save</c>: return to title if needed, then load a save folder.</summary>
    internal sealed class LoadJob : IJob
    {
        private readonly string saveName;
        private readonly Func<int> saveLoadedCount;
        private readonly int timeoutSeconds;
        private readonly Stopwatch clock = new();
        private int startCount;
        private int phase; // 0 exiting to title, 1 at title, 2 loading

        public LoadJob(string saveName, Func<int> saveLoadedCount, int timeoutSeconds)
        {
            this.saveName = saveName;
            this.saveLoadedCount = saveLoadedCount;
            this.timeoutSeconds = timeoutSeconds;
        }

        public string? Start()
        {
            if (!Directory.Exists(Path.Combine(Constants.SavesPath, this.saveName)))
                return $"save folder '{this.saveName}' not found in {Constants.SavesPath}";
            if (Context.IsMultiplayer)
                return "load_save is single-player only";

            this.startCount = this.saveLoadedCount();
            this.clock.Start();
            if (Context.IsWorldReady)
            {
                // UNVERIFIED: Game1.ExitToTitle()
                if (!GameCompat.ExitToTitle(out string error))
                    return error;
                this.phase = 0;
            }
            else
                this.phase = 1;
            return null;
        }

        public void BeforeTick()
        {
        }

        public bool AfterTick(out bool ok, out string detail)
        {
            ok = false;
            detail = "";
            if (this.clock.Elapsed.TotalSeconds > this.timeoutSeconds)
            {
                detail = $"timeout loading '{this.saveName}' (phase {this.phase})";
                return true;
            }

            switch (this.phase)
            {
                case 0:
                    if (!Context.IsWorldReady && Game1.activeClickableMenu is TitleMenu)
                        this.phase = 1;
                    return false;

                case 1:
                    // UNVERIFIED: SaveGame.Load(folderName) then closing the title menu is what the load menu's slot click does.
                    if (!GameCompat.LoadSave(this.saveName, out string error))
                    {
                        detail = error;
                        return true;
                    }
                    Game1.exitActiveMenu();
                    this.phase = 2;
                    return false;

                default:
                    if (this.saveLoadedCount() > this.startCount && Context.IsWorldReady && !Game1.fadeToBlack)
                    {
                        ok = true;
                        detail = $"loaded {this.saveName}";
                        return true;
                    }
                    return false;
            }
        }

        public void Cancel()
        {
        }
    }

    /*********
    ** Routing for goto
    *********/
    internal enum ExitKind { Edge, Touch, Door }

    /// <summary>A way out of a location.</summary>
    internal sealed class Exit
    {
        public string Target = "";
        public Point Tile;
        public Point Approach;
        public int Facing;
        public ExitKind Kind;
    }

    /// <summary>Builds the location graph from warps/doors and finds the next exit towards a target location.</summary>
    internal sealed class Router
    {
        private readonly Dictionary<string, List<Exit>> cache = new(StringComparer.OrdinalIgnoreCase);

        public static bool Same(string? a, string? b)
        {
            return string.Equals(a, b, StringComparison.OrdinalIgnoreCase);
        }

        public void Clear()
        {
            this.cache.Clear();
        }

        public Exit? NextExit(GameLocation from, string target, out string? error)
        {
            error = null;
            string start = from.NameOrUniqueName;
            var previous = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { start };
            var queue = new Queue<string>();
            queue.Enqueue(start);
            string? found = null;

            while (queue.Count > 0)
            {
                string current = queue.Dequeue();
                if (Same(current, target))
                {
                    found = current;
                    break;
                }
                GameLocation? location = Same(current, start) ? from : Game1.getLocationFromName(current);
                if (location == null)
                    continue;
                foreach (Exit exit in this.ExitsOf(location))
                {
                    if (seen.Add(exit.Target))
                    {
                        previous[exit.Target] = current;
                        queue.Enqueue(exit.Target);
                    }
                }
            }

            if (found == null)
            {
                error = $"no route from {start} to {target} through warps/doors";
                return null;
            }

            // walk back to the first hop
            string hop = found;
            while (previous.TryGetValue(hop, out string? parent) && !Same(parent, start))
                hop = parent;

            Point me = Game1.player.TilePoint;
            return this.ExitsOf(from)
                .Where(e => Same(e.Target, hop))
                .OrderBy(e => Math.Abs(e.Approach.X - me.X) + Math.Abs(e.Approach.Y - me.Y))
                .FirstOrDefault();
        }

        private List<Exit> ExitsOf(GameLocation location)
        {
            string name = location.NameOrUniqueName;
            if (this.cache.TryGetValue(name, out List<Exit>? cached))
                return cached;

            var exits = new List<Exit>();
            try
            {
                (int width, int height) = Snapshot.MapSize(location);

                // touch warps (map 'Warp' property); X/Y/TargetName verified via Lookup Anything (MIT)
                foreach (Warp warp in GameCompat.GetWarps(location))
                {
                    if (string.IsNullOrEmpty(warp.TargetName) || Same(warp.TargetName, name))
                        continue;
                    int x = warp.X, y = warp.Y;
                    bool inside = x >= 0 && y >= 0 && x < width && y < height;
                    int facing = x < 0 ? 3 : x >= width ? 1 : y < 0 ? 0 : y >= height ? 2 : 2;
                    var tile = new Point(Math.Clamp(x, 0, width - 1), Math.Clamp(y, 0, height - 1));
                    exits.Add(new Exit
                    {
                        Target = warp.TargetName,
                        Tile = new Point(x, y),
                        Approach = tile,
                        Facing = facing,
                        Kind = inside ? ExitKind.Touch : ExitKind.Edge,
                    });
                }

                // door-like tile actions on the Buildings layer (action names as in Data Layers' warp list, MIT)
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        string? action = location.doesTileHaveProperty(x, y, "Action", "Buildings");
                        string? doorTarget = ParseDoorAction(action);
                        if (doorTarget == null || Same(doorTarget, name) || y + 1 >= height)
                            continue;
                        exits.Add(new Exit { Target = doorTarget, Tile = new Point(x, y), Approach = new Point(x, y + 1), Facing = 0, Kind = ExitKind.Door });
                    }
                }

                // farm building doors (members as used by Data Layers / Lookup Anything, MIT)
                foreach (Building building in location.buildings)
                {
                    if (!building.HasIndoors() || (building.humanDoor.X < 0 && building.humanDoor.Y < 0))
                        continue;
                    string? indoors = building.GetIndoorsName();
                    if (string.IsNullOrEmpty(indoors))
                        continue;
                    int x = building.tileX.Value + building.humanDoor.X;
                    int y = building.tileY.Value + building.humanDoor.Y;
                    exits.Add(new Exit { Target = indoors, Tile = new Point(x, y), Approach = new Point(x, y + 1), Facing = 0, Kind = ExitKind.Door });
                }
            }
            catch (Exception)
            {
                // a broken custom map shouldn't break routing elsewhere
            }

            this.cache[name] = exits;
            return exits;
        }

        /// <summary>Target location of a door-like tile action, or null.</summary>
        // UNVERIFIED: argument layout "Warp <x> <y> <location>" / "LockedDoorWarp <x> <y> <location> <open> <close> ...";
        // "WarpCommunityCenter" leads to the CommunityCenter.
        private static string? ParseDoorAction(string? action)
        {
            if (string.IsNullOrWhiteSpace(action))
                return null;
            string[] parts = action.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            switch (parts[0])
            {
                case "Warp":
                case "LockedDoorWarp":
                    return parts.Length >= 4 ? parts[3] : null;
                case "WarpCommunityCenter":
                    return "CommunityCenter";
                default:
                    return null;
            }
        }
    }

    /*********
    ** Bundles
    *********/
    internal static class BundleDelivery
    {
        /// <summary>
        /// Donate every held item that fits an unfilled slot of an incomplete Community Center bundle, the way the
        /// bundle menu does: one stack of at least the required count and quality per slot, until the bundle has the
        /// number of items it needs. Completed rooms are flagged like CJB Cheats Menu (MIT) does.
        /// </summary>
        public static bool Deliver(bool anywhere, bool payVault, EventLog events, out string detail)
        {
            CommunityCenter? cc = Snapshot.GetCommunityCenter();
            if (cc == null)
            {
                detail = "CommunityCenter location not found";
                return false;
            }
            Farmer player = Game1.player;
            Farmer master = Game1.MasterPlayer;
            if (master.mailReceived.Contains("JojaMember"))
            {
                detail = "Joja membership: Community Center bundles are unavailable";
                return false;
            }
            if (!master.mailReceived.Contains("ccDoorUnlock"))
            {
                detail = "the Community Center is not unlocked yet (mail flag ccDoorUnlock)";
                return false;
            }
            if (!anywhere && player.currentLocation is not CommunityCenter)
            {
                detail = "not in the CommunityCenter (pass {\"anywhere\": true} to donate remotely)";
                return false;
            }

            var bundles = Snapshot.ParseBundles();
            int donated = 0;
            var completed = new List<string>();
            var problems = new List<string>();

            foreach (BundleInfo bundle in bundles)
            {
                if (bundle.Area == Snapshot.JojaArea || Snapshot.IsBundleComplete(cc, bundle.Id))
                    continue;
                bool[] current = Snapshot.DoneFlags(cc, bundle.Id);
                if (current.Length == 0)
                    continue;

                bool[] next = (bool[])current.Clone();
                int done = 0;
                for (int i = 0; i < Math.Min(next.Length, bundle.Ingredients.Count); i++)
                {
                    if (next[i])
                        done++;
                }

                bool changed = false;
                for (int i = 0; i < bundle.Ingredients.Count && i < next.Length && done < bundle.Need; i++)
                {
                    if (next[i])
                        continue;
                    BundleIngredient ingredient = bundle.Ingredients[i];
                    if (ingredient.IsMoney)
                    {
                        if (!payVault || player.Money < ingredient.Qty)
                            continue;
                        player.Money -= ingredient.Qty;
                    }
                    else
                    {
                        Item? item = FindIngredient(player, ingredient);
                        if (item == null)
                            continue;
                        // same consumption pattern as TractorMod (MIT)
                        item.Stack -= ingredient.Qty;
                        if (item.Stack <= 0)
                            player.removeItemFromInventory(item);
                    }

                    next[i] = true;
                    changed = true;
                    done++;
                    donated++;
                    events.Add("bundle_delivery", ("bundle", bundle.Key), ("name", bundle.Name), ("item", ingredient.Id), ("qty", ingredient.Qty), ("quality", ingredient.Quality));
                }

                if (!changed)
                    continue;

                // UNVERIFIED: assigning a new bool[] through the NetBundles indexer stores (and net-syncs) the donated flags.
                cc.bundles[bundle.Id] = next;
                if (!Snapshot.DoneFlags(cc, bundle.Id).SequenceEqual(next))
                    problems.Add($"{bundle.Key}: donated flags did not persist");

                if (done >= bundle.Need)
                {
                    // UNVERIFIED: BundleRewards[id] = true makes the bundle's reward collectible in the bundle menu.
                    Game1.netWorldState.Value.BundleRewards[bundle.Id] = true;
                    completed.Add(bundle.Key);
                    events.Add("bundle_delivery", ("bundle", bundle.Key), ("name", bundle.Name), ("complete", true));
                }
            }

            // rooms whose bundles are now all complete
            var rooms = new List<string>();
            foreach ((string area, int index, string flag) in Snapshot.Areas)
            {
                if (index < 0 || index >= cc.areasComplete.Length || cc.areasComplete[index])
                    continue;
                List<BundleInfo> inArea = bundles.Where(b => b.Area == area).ToList();
                if (inArea.Count == 0 || !inArea.All(b => Snapshot.IsBundleComplete(cc, b.Id)))
                    continue;

                // UNVERIFIED (behaviour): the room's restoration cutscene/visuals are not played here; the flags are what
                // CJB Cheats Menu sets to mark a room complete (areasComplete[i] + the room's mail flag).
                cc.areasComplete[index] = true;
                if (!player.mailReceived.Contains(flag))
                    player.mailReceived.Add(flag);
                rooms.Add(area);
                events.Add("bundle_delivery", ("area", area), ("area_complete", true));
            }

            detail = $"donated {donated} item(s); completed bundles: [{string.Join(", ", completed)}]; completed rooms: [{string.Join(", ", rooms)}]";
            if (problems.Count > 0)
                detail += "; problems: " + string.Join("; ", problems);
            return problems.Count == 0;
        }

        private static Item? FindIngredient(Farmer player, BundleIngredient ingredient)
        {
            for (int i = 0; i < player.Items.Count; i++)
            {
                if (player.Items[i] is not SObject obj)
                    continue;
                if (obj.Stack < ingredient.Qty || obj.Quality < ingredient.Quality)
                    continue;
                bool matches = ingredient.Category != null
                    ? !obj.bigCraftable.Value && obj.Category == ingredient.Category.Value
                    : obj.QualifiedItemId == ingredient.Id;
                if (matches)
                    return obj;
            }
            return null;
        }
    }
}
