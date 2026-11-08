using System;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewModdingAPI.Events;
using StardewValley;

namespace RLBridge
{
    /// <summary>
    /// SMAPI entry point. Owns the TCP server, the game-thread command dispatcher, the lockstep loop, the event
    /// buffer and the recorder. Every game access happens on the game thread (in SMAPI event handlers).
    /// Protocol: docs/BRIDGE_PROTOCOL.md (v1); client: crates/sdv-bridge.
    /// </summary>
    public sealed class ModEntry : Mod
    {
        private const int Protocol = 1;

        private sealed class ActiveJob
        {
            public ActiveJob(IJob job, PendingRequest request, long id, bool isLoad)
            {
                this.Job = job;
                this.Request = request;
                this.Id = id;
                this.IsLoad = isLoad;
            }

            public IJob Job { get; }
            public PendingRequest Request { get; }
            public long Id { get; }
            public bool IsLoad { get; }
        }

        private ModConfig config = new();
        private BridgeServer? server;
        private InputDriver input = null!;
        private readonly EventLog events = new();
        private readonly Recorder recorder = new();
        private readonly Router router = new();
        private ActiveJob? job;
        private bool lockstep;
        private ulong tick;
        private int saveLoadedCount;
        private int dayStartedCount;
        private bool pendingTeleport;
        private int lastRecordedSlot = -1;

        /*********
        ** Entry
        *********/
        public override void Entry(IModHelper helper)
        {
            this.config = helper.ReadConfig<ModConfig>();
            this.config.WindowRadius = Math.Clamp(this.config.WindowRadius, 0, 32);
            this.input = new InputDriver(helper, this.Monitor, this.config.InputMode);

            helper.Events.GameLoop.UpdateTicking += this.OnUpdateTicking;
            helper.Events.GameLoop.UpdateTicked += this.OnUpdateTicked;
            helper.Events.GameLoop.SaveLoaded += this.OnSaveLoaded;
            helper.Events.GameLoop.DayStarted += this.OnDayStarted;
            helper.Events.GameLoop.DayEnding += this.OnDayEnding;
            helper.Events.GameLoop.ReturnedToTitle += this.OnReturnedToTitle;
            helper.Events.Player.Warped += this.OnWarped;
            helper.Events.Player.InventoryChanged += this.OnInventoryChanged;
            helper.Events.Display.MenuChanged += this.OnMenuChanged;
            AppDomain.CurrentDomain.ProcessExit += (_, _) => this.recorder.Dispose();

            try
            {
                this.server = new BridgeServer(this.config.Host, this.config.Port, this.Monitor);
                this.server.Start();
            }
            catch (Exception ex)
            {
                this.server = null;
                this.Monitor.Log($"Couldn't listen on {this.config.Host}:{this.config.Port}: {ex.Message}", LogLevel.Error);
            }
        }

        protected override void Dispose(bool disposing)
        {
            this.server?.Dispose();
            this.recorder.Dispose();
            base.Dispose(disposing);
        }

        /*********
        ** Game loop
        *********/
        private void OnUpdateTicking(object? sender, UpdateTickingEventArgs e)
        {
            try
            {
                this.job?.Job.BeforeTick();
            }
            catch (Exception ex)
            {
                this.FailActiveJob(ex);
            }
        }

        private void OnUpdateTicked(object? sender, UpdateTickedEventArgs e)
        {
            this.tick = e.Ticks;

            // advance the running step/macro
            if (this.job != null)
            {
                try
                {
                    if (this.job.Job.AfterTick(out bool ok, out string detail))
                        this.FinishJob(ok, detail);
                }
                catch (Exception ex)
                {
                    this.FailActiveJob(ex);
                }
            }

            if (this.recorder.Enabled)
                this.SampleRecording();

            if (this.server == null)
                return;

            // answer anything queued (free-running mode, or requests that arrived during a job)
            while (this.server.Queue.TryTake(out PendingRequest? request))
                this.Handle(request);

            // lockstep: hold the game thread here until the agent sends something that advances time
            if (this.lockstep && this.job == null && this.server.IsConnected)
            {
                var idle = Stopwatch.StartNew();
                while (this.lockstep && this.job == null && this.server.IsConnected)
                {
                    if (this.server.Queue.TryTake(out PendingRequest? request, 100))
                    {
                        this.Handle(request);
                        idle.Restart();
                    }
                    else if (this.config.LockstepIdleTimeoutSeconds > 0 && idle.Elapsed.TotalSeconds > this.config.LockstepIdleTimeoutSeconds)
                    {
                        this.lockstep = false;
                        this.Monitor.Log($"No command for {this.config.LockstepIdleTimeoutSeconds}s; lockstep released.", LogLevel.Warn);
                    }
                }
            }
        }

        /*********
        ** Commands
        *********/
        private void Handle(PendingRequest request)
        {
            if (request.IsDisconnect)
            {
                this.OnClientDisconnected();
                return;
            }

            long id = 0;
            try
            {
                if (JsonNode.Parse(request.Line) is not JsonObject message)
                {
                    request.Respond(Error(0, "request must be a JSON object"));
                    return;
                }
                id = Json.Long(message, "id");
                string cmd = Json.Str(message, "cmd") ?? "";

                if (this.job != null && cmd != "hello" && cmd != "observe")
                {
                    request.Respond(Error(id, "busy: a step/macro is still running"));
                    return;
                }

                switch (cmd)
                {
                    case "hello":
                        request.Respond(new JsonObject
                        {
                            ["id"] = id,
                            ["ok"] = true,
                            ["protocol"] = Protocol,
                            ["game_version"] = Game1.version,
                            ["smapi_version"] = Constants.ApiVersion.ToString(),
                            ["mod_version"] = this.ModManifest.Version.ToString(),
                            ["input_mode"] = this.input.UsePress ? "press" : "direct",
                            ["world_ready"] = Context.IsWorldReady,
                        }.ToJsonString());
                        break;

                    case "observe":
                        request.Respond(new JsonObject { ["id"] = id, ["ok"] = true, ["obs"] = this.BuildSnapshot() }.ToJsonString());
                        break;

                    case "lockstep":
                        this.lockstep = Json.Bool(message, "enabled");
                        request.Respond(new JsonObject { ["id"] = id, ["ok"] = true, ["lockstep"] = this.lockstep }.ToJsonString());
                        break;

                    case "record":
                        this.HandleRecord(request, id, message);
                        break;

                    case "step":
                        {
                            if (!Context.IsWorldReady)
                            {
                                request.Respond(Error(id, "no save loaded"));
                                return;
                            }
                            int ticks = Json.Int(message, "ticks", 1);
                            if (ticks < 1 || ticks > 600)
                            {
                                request.Respond(Error(id, "ticks must be in 1..600"));
                                return;
                            }
                            InputFrame frame = InputFrame.Parse(Json.Obj(message, "input"));
                            this.StartJob(new StepJob(this.input, frame, ticks), request, id, isLoad: false);
                            break;
                        }

                    case "macro":
                        this.HandleMacro(request, id, Json.Str(message, "name") ?? "", Json.Obj(message, "args"));
                        break;

                    case "load_save":
                        {
                            string? save = Json.Str(message, "save_name");
                            if (string.IsNullOrWhiteSpace(save) || save.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
                            {
                                request.Respond(Error(id, "save_name must be a save folder name, e.g. 'Farm_123456789'"));
                                return;
                            }
                            this.StartJob(new LoadJob(save, () => this.saveLoadedCount, this.config.SlowMacroTimeoutSeconds), request, id, isLoad: true);
                            break;
                        }

                    default:
                        request.Respond(Error(id, $"unknown cmd '{cmd}'"));
                        break;
                }
            }
            catch (JsonException ex)
            {
                request.Respond(Error(id, $"invalid JSON: {ex.Message}"));
            }
            catch (Exception ex)
            {
                this.Monitor.Log($"Command failed: {ex}", LogLevel.Error);
                request.Respond(Error(id, ex.Message));
            }
        }

        private void HandleMacro(PendingRequest request, long id, string name, JsonObject? args)
        {
            if (!Context.IsWorldReady)
            {
                request.Respond(Error(id, "no save loaded"));
                return;
            }

            switch (name)
            {
                case "face":
                    {
                        int direction = Json.Int(args, "direction", -1);
                        if (direction < 0 || direction > 3)
                        {
                            request.Respond(this.StepResponse(id, false, "direction must be 0 (up), 1 (right), 2 (down) or 3 (left)"));
                            return;
                        }
                        Game1.player.faceDirection(direction);
                        request.Respond(this.StepResponse(id, true, $"facing {InputFrame.DirectionName(direction)}"));
                        return;
                    }

                case "select_item":
                    {
                        string? itemId = Json.Str(args, "item_id");
                        if (string.IsNullOrWhiteSpace(itemId))
                        {
                            request.Respond(this.StepResponse(id, false, "item_id is required, e.g. '(O)24'"));
                            return;
                        }
                        if (!itemId.StartsWith("("))
                            itemId = "(O)" + itemId;
                        for (int slot = 0; slot < Game1.player.Items.Count; slot++)
                        {
                            Item? item = Game1.player.Items[slot];
                            if (item != null && string.Equals(item.QualifiedItemId, itemId, StringComparison.OrdinalIgnoreCase))
                            {
                                Game1.player.CurrentToolIndex = slot;
                                request.Respond(this.StepResponse(id, true, $"selected slot {slot}"));
                                return;
                            }
                        }
                        request.Respond(this.StepResponse(id, false, $"no {itemId} in inventory"));
                        return;
                    }

                case "deliver_bundles":
                    {
                        bool ok = BundleDelivery.Deliver(Json.Bool(args, "anywhere"), Json.Bool(args, "pay_vault"), this.events, out string detail);
                        request.Respond(this.StepResponse(id, ok, detail));
                        return;
                    }

                case "warp":
                    {
                        if (!this.config.AllowWarpMacro)
                        {
                            request.Respond(this.StepResponse(id, false, "warp macro disabled in config.json (AllowWarpMacro)"));
                            return;
                        }
                        string? location = Json.Str(args, "location");
                        int? x = Json.NullableInt(args, "x");
                        int? y = Json.NullableInt(args, "y");
                        if (string.IsNullOrWhiteSpace(location) || x == null || y == null)
                        {
                            request.Respond(this.StepResponse(id, false, "warp needs location, x, y"));
                            return;
                        }
                        this.StartJob(new WarpJob(location, x.Value, y.Value, this.config.MacroTimeoutTicks, () => this.pendingTeleport = true), request, id, isLoad: false);
                        return;
                    }

                case "goto":
                    {
                        string? location = Json.Str(args, "location");
                        if (string.IsNullOrWhiteSpace(location))
                        {
                            request.Respond(this.StepResponse(id, false, "goto needs location"));
                            return;
                        }
                        int? x = Json.NullableInt(args, "x");
                        int? y = Json.NullableInt(args, "y");
                        Point? tile = x != null && y != null ? new Point(x.Value, y.Value) : null;
                        this.StartJob(new GotoJob(location, tile, this.input, this.router, this.config.MacroTimeoutTicks), request, id, isLoad: false);
                        return;
                    }

                case "sleep":
                    this.StartJob(new SleepJob(() => this.dayStartedCount, this.config.SlowMacroTimeoutSeconds), request, id, isLoad: false);
                    return;

                default:
                    request.Respond(this.StepResponse(id, false, $"unknown macro '{name}'"));
                    return;
            }
        }

        private void HandleRecord(PendingRequest request, long id, JsonObject message)
        {
            if (!Json.Bool(message, "enabled"))
            {
                long lines = this.recorder.LinesWritten;
                string? path = this.recorder.FilePath;
                this.recorder.Stop();
                this.events.RecordEnabled = false;
                request.Respond(new JsonObject { ["id"] = id, ["ok"] = true, ["path"] = path, ["lines"] = lines }.ToJsonString());
                return;
            }

            string file = this.ResolveRecordPath(Json.Str(message, "path"));
            var header = new JsonObject
            {
                ["kind"] = "header",
                ["format"] = "rlbridge-record/1",
                ["protocol"] = Protocol,
                ["game_version"] = Game1.version,
                ["mod_version"] = this.ModManifest.Version.ToString(),
                ["save"] = Constants.SaveFolderName,
                ["started_utc"] = DateTime.UtcNow.ToString("o"),
                ["tick"] = this.tick,
                ["every_ticks"] = this.config.RecordEveryTicks,
            };
            this.recorder.Start(file, header);
            this.events.RecordEnabled = true;
            this.lastRecordedSlot = -1;
            request.Respond(new JsonObject { ["id"] = id, ["ok"] = true, ["path"] = file }.ToJsonString());
        }

        private string ResolveRecordPath(string? path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return Path.Combine(this.Helper.DirectoryPath, "recordings", $"rec-{DateTime.Now:yyyyMMdd-HHmmss}.jsonl.gz");
            return Path.IsPathRooted(path) ? path : Path.Combine(this.Helper.DirectoryPath, path);
        }

        /*********
        ** Jobs
        *********/
        private void StartJob(IJob newJob, PendingRequest request, long id, bool isLoad)
        {
            string? error = newJob.Start();
            if (error != null)
            {
                request.Respond(isLoad ? Error(id, error) : this.StepResponse(id, false, error));
                return;
            }
            this.job = new ActiveJob(newJob, request, id, isLoad);
        }

        private void FinishJob(bool ok, string detail)
        {
            ActiveJob done = this.job!;
            this.job = null;
            if (done.IsLoad)
            {
                done.Request.Respond(ok
                    ? new JsonObject { ["id"] = done.Id, ["ok"] = true, ["obs"] = this.BuildSnapshot(), ["detail"] = detail }.ToJsonString()
                    : Error(done.Id, detail));
            }
            else
                done.Request.Respond(this.StepResponse(done.Id, ok, detail));
        }

        private void FailActiveJob(Exception ex)
        {
            this.Monitor.Log($"Step/macro failed: {ex}", LogLevel.Error);
            if (this.job == null)
                return;
            try
            {
                this.job.Job.Cancel();
            }
            catch (Exception)
            {
                // ignore
            }
            this.FinishJob(false, ex.Message);
        }

        private void OnClientDisconnected()
        {
            this.lockstep = false;
            if (this.job != null)
            {
                try
                {
                    this.job.Job.Cancel();
                }
                catch (Exception ex)
                {
                    this.Monitor.Log($"Cancel failed: {ex.Message}", LogLevel.Trace);
                }
                this.job.Request.Respond(Error(this.job.Id, "client disconnected"));
                this.job = null;
            }
        }

        /*********
        ** Responses
        *********/
        private JsonObject BuildSnapshot()
        {
            return Snapshot.Build(this.tick, this.config.WindowRadius);
        }

        private string StepResponse(long id, bool ok, string detail)
        {
            return new JsonObject
            {
                ["id"] = id,
                ["ok"] = ok,
                ["obs"] = this.BuildSnapshot(),
                ["events"] = this.events.DrainAgent(),
                ["detail"] = detail,
            }.ToJsonString();
        }

        private static string Error(long id, string message)
        {
            return new JsonObject { ["id"] = id, ["ok"] = false, ["error"] = message }.ToJsonString();
        }

        /*********
        ** Recording
        *********/
        private void SampleRecording()
        {
            if (!Context.IsWorldReady)
                return;
            int every = Math.Max(1, this.config.RecordEveryTicks);
            if (this.tick % (ulong)every != 0 && !this.events.HasRecorderEvents)
                return;

            try
            {
                InputFrame held = this.input.ReadHumanInput();
                int slot = Game1.player.CurrentToolIndex;
                if (slot != this.lastRecordedSlot)
                {
                    held.SelectSlot = slot;
                    this.lastRecordedSlot = slot;
                }
                this.recorder.Sample(this.tick, this.BuildSnapshot(), held, this.events.DrainRecorder());
            }
            catch (Exception ex)
            {
                this.Monitor.Log($"Recording failed, stopping: {ex}", LogLevel.Error);
                this.recorder.Stop();
                this.events.RecordEnabled = false;
            }
        }

        /*********
        ** Game events -> protocol events
        *********/
        private void OnSaveLoaded(object? sender, SaveLoadedEventArgs e)
        {
            this.saveLoadedCount++;
            this.router.Clear();
            // UNVERIFIED: Options.pauseWhenOutOfFocus
            if (this.config.DisablePauseWhenUnfocused && !GameCompat.SetPauseWhenOutOfFocus(false))
                this.Monitor.LogOnce("Couldn't turn off 'pause when unfocused'; keep the game window focused.", LogLevel.Warn);
        }

        private void OnDayStarted(object? sender, DayStartedEventArgs e)
        {
            this.dayStartedCount++;
            this.router.Clear(); // maps can change overnight (e.g. restored rooms)
        }

        private void OnDayEnding(object? sender, DayEndingEventArgs e)
        {
            this.events.Add("day_end", ("day", Game1.dayOfMonth), ("season", Game1.season.ToString().ToLowerInvariant()), ("year", Game1.year));
        }

        private void OnReturnedToTitle(object? sender, ReturnedToTitleEventArgs e)
        {
            this.router.Clear();
        }

        private void OnWarped(object? sender, WarpedEventArgs e)
        {
            if (!e.IsLocalPlayer)
                return;
            Point tile = e.Player.TilePoint;
            this.events.Add("warp",
                ("from", e.OldLocation?.NameOrUniqueName),
                ("to", e.NewLocation?.NameOrUniqueName),
                ("tile", new[] { tile.X, tile.Y }),
                ("teleport", this.pendingTeleport));
            this.pendingTeleport = false;
        }

        private void OnMenuChanged(object? sender, MenuChangedEventArgs e)
        {
            if (e.OldMenu != null)
                this.events.Add("menu_close", ("menu", e.OldMenu.GetType().Name));
            if (e.NewMenu != null)
                this.events.Add("menu_open", ("menu", e.NewMenu.GetType().Name));
        }

        /// <summary>Approximate purchase/harvest events: items gained while a ShopMenu is open count as purchases, others as 'harvest' (any pickup).</summary>
        private void OnInventoryChanged(object? sender, InventoryChangedEventArgs e)
        {
            if (!e.IsLocalPlayer)
                return;
            string kind = Game1.activeClickableMenu?.GetType().Name == "ShopMenu" ? "purchase" : "harvest";
            foreach (Item item in e.Added)
                this.events.Add(kind, ("item", item.QualifiedItemId), ("qty", item.Stack));
            foreach (ItemStackSizeChange change in e.QuantityChanged)
            {
                if (change.NewSize > change.OldSize)
                    this.events.Add(kind, ("item", change.Item.QualifiedItemId), ("qty", change.NewSize - change.OldSize));
            }
        }
    }
}
