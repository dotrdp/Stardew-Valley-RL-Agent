using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text.Json.Nodes;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;
using StardewModdingAPI;
using StardewValley;

namespace RLBridge
{
    /// <summary>One protocol InputFrame: <c>{"move", "use_tool", "action", "select_slot", "face"}</c>.</summary>
    internal sealed class InputFrame
    {
        /// <summary>Movement direction in the game's convention (0 up, 1 right, 2 down, 3 left), or null.</summary>
        public int? Move;
        public bool UseTool;
        public bool Action;
        public int? SelectSlot;
        public int? Face;

        public bool IsEmpty => this.Move == null && !this.UseTool && !this.Action && this.SelectSlot == null && this.Face == null;

        public static InputFrame Parse(JsonObject? json)
        {
            var frame = new InputFrame();
            if (json == null)
                return frame;
            frame.Move = DirectionFromName(Json.Str(json, "move"));
            frame.UseTool = Json.Bool(json, "use_tool");
            frame.Action = Json.Bool(json, "action");
            frame.SelectSlot = Json.NullableInt(json, "select_slot");
            int? face = Json.NullableInt(json, "face");
            frame.Face = face is >= 0 and <= 3 ? face : null;
            return frame;
        }

        public JsonObject ToJson()
        {
            return new JsonObject
            {
                ["move"] = this.Move == null ? null : DirectionName(this.Move.Value),
                ["use_tool"] = this.UseTool,
                ["action"] = this.Action,
                ["select_slot"] = this.SelectSlot,
                ["face"] = this.Face,
            };
        }

        public static int? DirectionFromName(string? name)
        {
            switch (name)
            {
                case "up": return 0;
                case "right": return 1;
                case "down": return 2;
                case "left": return 3;
                default: return null;
            }
        }

        public static string DirectionName(int direction)
        {
            switch (direction)
            {
                case 0: return "up";
                case 1: return "right";
                case 2: return "down";
                default: return "left";
            }
        }

        /// <summary>Tile offset for a facing direction (0 up, 1 right, 2 down, 3 left), as in Lookup Anything (MIT).</summary>
        public static Point Offset(int direction)
        {
            switch (direction)
            {
                case 0: return new Point(0, -1);
                case 1: return new Point(1, 0);
                case 2: return new Point(0, 1);
                default: return new Point(-1, 0);
            }
        }
    }

    /// <summary>
    /// Applies input frames to the local player. Two mechanisms:
    /// <list type="bullet">
    ///   <item><b>press</b>: SMAPI's <c>IInputHelper.Press(SButton)</c> (added in SMAPI 4.4.0; verified in the SMAPI source),
    ///   documented as equivalent to the player physically pressing the button for one input tick. The game's own input
    ///   handling then moves the player, swings tools, checks actions, etc. Called each tick to hold a key.</item>
    ///   <item><b>direct</b>: verified farmer methods (<c>SetMovingUp/Right/Down/Left</c>, <c>Halt</c>, <c>faceDirection</c>,
    ///   <c>GameLocation.checkAction</c>), plus the UNVERIFIED <c>Game1.pressUseToolButton()</c> for tools.</item>
    /// </list>
    /// Press is looked up by reflection so the mod still builds and runs (in direct mode) against SMAPI 4.0-4.3.
    /// </summary>
    internal sealed class InputDriver
    {
        private readonly IModHelper helper;
        private readonly IMonitor monitor;
        private readonly MethodInfo? pressMethod;
        private readonly HashSet<SButton> pressedThisStep = new();

        public InputDriver(IModHelper helper, IMonitor monitor, string mode)
        {
            this.helper = helper;
            this.monitor = monitor;
            this.pressMethod = typeof(IInputHelper).GetMethod("Press", new[] { typeof(SButton) });

            switch ((mode ?? "auto").Trim().ToLowerInvariant())
            {
                case "direct":
                    this.UsePress = false;
                    break;
                case "press":
                    this.UsePress = this.pressMethod != null;
                    if (this.pressMethod == null)
                        monitor.Log("InputMode 'press' needs SMAPI 4.4.0+; falling back to 'direct'.", LogLevel.Warn);
                    break;
                default:
                    this.UsePress = this.pressMethod != null;
                    break;
            }
            monitor.Log($"Input mode: {(this.UsePress ? "press (SMAPI input API)" : "direct (farmer methods)")}.", LogLevel.Info);
        }

        public bool UsePress { get; }

        /// <summary>Apply a frame for one tick. Call from <c>UpdateTicking</c> (before the game update).</summary>
        /// <param name="frame">The held frame.</param>
        /// <param name="firstTick">Whether this is the first tick of the hold: one-shot parts (slot, face, tool, action) apply only then.</param>
        public void Apply(InputFrame frame, bool firstTick)
        {
            Farmer player = Game1.player;
            if (player == null)
                return;

            if (firstTick)
            {
                if (frame.SelectSlot is int slot && slot >= 0 && slot < player.Items.Count)
                    player.CurrentToolIndex = slot;
                if (frame.Face is int face && frame.Move == null)
                    player.faceDirection(face);
            }

            if (frame.Move is int direction)
            {
                if (this.UsePress)
                    this.Press(MoveKey(direction));
                else
                    SetMoving(player, direction, true);
            }

            if (firstTick && frame.UseTool)
            {
                if (this.UsePress)
                    this.Press(ToolKey());
                else if (!GameCompat.PressUseToolButton(out string error))
                    this.monitor.LogOnce($"use_tool unavailable in direct mode: {error}", LogLevel.Warn);
            }

            if (firstTick && frame.Action)
            {
                if (this.UsePress)
                    this.Press(ActionKey());
                else
                    CheckActionInFront(player);
            }
        }

        /// <summary>Release everything held by the current frame. Call from <c>UpdateTicked</c> on the last tick.</summary>
        public void Release(InputFrame? frame)
        {
            if (this.UsePress)
            {
                // A pressed override is re-applied once more on the next input update; suppress cancels that so the
                // hold ends exactly with the step (SMAPI SInputState semantics, verified in the SMAPI source).
                foreach (SButton button in this.pressedThisStep)
                    this.helper.Input.Suppress(button);
                this.pressedThisStep.Clear();
            }
            else if (frame?.Move != null && Game1.player != null)
            {
                SetMoving(Game1.player, frame.Move.Value, false);
                Game1.player.Halt();
            }
        }

        /// <summary>Infer the human's held input from the physical keys (for recordings).</summary>
        public InputFrame ReadHumanInput()
        {
            var frame = new InputFrame();
            for (int d = 0; d < 4; d++)
            {
                if (this.helper.Input.IsDown(MoveKey(d)))
                {
                    frame.Move = d;
                    break;
                }
            }
            frame.UseTool = this.helper.Input.IsDown(ToolKey()) || this.helper.Input.IsDown(SButton.MouseLeft);
            frame.Action = this.helper.Input.IsDown(ActionKey()) || this.helper.Input.IsDown(SButton.MouseRight);
            return frame;
        }

        private void Press(SButton button)
        {
            try
            {
                this.pressMethod!.Invoke(this.helper.Input, new object[] { button });
                this.pressedThisStep.Add(button);
            }
            catch (Exception ex)
            {
                this.monitor.LogOnce($"IInputHelper.Press failed: {ex.InnerException?.Message ?? ex.Message}", LogLevel.Error);
            }
        }

        /*********
        ** Key bindings (Game1.options.*Button are InputButton[]; fields mouseLeft/mouseRight/key verified in SMAPI's SButton.cs)
        *********/
        public static SButton MoveKey(int direction)
        {
            switch (direction)
            {
                case 0: return KeyboardBinding(Game1.options.moveUpButton, SButton.W);
                case 1: return KeyboardBinding(Game1.options.moveRightButton, SButton.D);
                case 2: return KeyboardBinding(Game1.options.moveDownButton, SButton.S);
                default: return KeyboardBinding(Game1.options.moveLeftButton, SButton.A);
            }
        }

        // UNVERIFIED (behaviour): with the keyboard tool key the game targets the faced tile; if the mouse cursor is visible
        // and was moved recently, the game may aim at the cursor instead. Keep the mouse still/outside the window.
        public static SButton ToolKey()
        {
            return KeyboardBinding(Game1.options.useToolButton, SButton.C);
        }

        public static SButton ActionKey()
        {
            return KeyboardBinding(Game1.options.actionButton, SButton.X);
        }

        /// <summary>The first keyboard key bound to an action (mouse bindings skipped), else the vanilla default.</summary>
        private static SButton KeyboardBinding(InputButton[]? bindings, SButton fallback)
        {
            if (bindings != null)
            {
                foreach (InputButton binding in bindings)
                {
                    if (!binding.mouseLeft && !binding.mouseRight && binding.key != Keys.None)
                        return (SButton)binding.key; // SMAPI defines SButton keyboard values as the XNA Keys values
                }
            }
            return fallback;
        }

        /*********
        ** Direct mode
        *********/
        /// <summary>Verified: Farmer overrides SetMovingUp/Right/Down/Left(bool) (see Farmtronics, MIT).</summary>
        // UNVERIFIED (behaviour): the game's keyboard handler might clear these flags on ticks with no physical key held.
        private static void SetMoving(Farmer player, int direction, bool value)
        {
            switch (direction)
            {
                case 0: player.SetMovingUp(value); break;
                case 1: player.SetMovingRight(value); break;
                case 2: player.SetMovingDown(value); break;
                default: player.SetMovingLeft(value); break;
            }
        }

        /// <summary>Check the tile in front of the player, as TractorMod (MIT) does: <c>location.checkAction(Location, viewport, farmer)</c>.</summary>
        public static bool CheckActionInFront(Farmer player)
        {
            GameLocation? location = player.currentLocation;
            if (location == null)
                return false;
            Point here = player.TilePoint;
            Point offset = InputFrame.Offset(player.FacingDirection);
            return location.checkAction(new xTile.Dimensions.Location(here.X + offset.X, here.Y + offset.Y), Game1.viewport, player);
        }
    }
}
