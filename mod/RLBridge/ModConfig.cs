namespace RLBridge
{
    /// <summary>The mod settings, read from <c>config.json</c> (SMAPI creates it with these defaults on first run).</summary>
    internal sealed class ModConfig
    {
        /// <summary>The address to listen on. Keep this on loopback: the protocol has no authentication.</summary>
        public string Host { get; set; } = "127.0.0.1";

        /// <summary>The TCP port to listen on.</summary>
        public int Port { get; set; } = 7777;

        /// <summary>The radius r of the (2r+1)x(2r+1) passability window in snapshots.</summary>
        public int WindowRadius { get; set; } = 6;

        /// <summary>
        /// How input frames reach the game: <c>auto</c> (SMAPI's <c>IInputHelper.Press</c> when available, i.e. SMAPI 4.4.0+,
        /// else <c>direct</c>), <c>press</c>, or <c>direct</c> (call farmer movement setters and game methods directly).
        /// </summary>
        public string InputMode { get; set; } = "auto";

        /// <summary>Whether to turn off the game's 'pause when the window is unfocused' option after a save loads.</summary>
        public bool DisablePauseWhenUnfocused { get; set; } = true;

        /// <summary>Whether the <c>warp</c> teleport macro is allowed.</summary>
        public bool AllowWarpMacro { get; set; } = true;

        /// <summary>Maximum update ticks a multi-tick macro (<c>warp</c>, <c>goto</c>) may run before it returns <c>ok: false</c>.</summary>
        /// <remarks>The Rust client uses a 30 s read timeout, so keep this under ~1700 ticks (60 ticks/s).</remarks>
        public int MacroTimeoutTicks { get; set; } = 1500;

        /// <summary>Wall-clock limit for <c>sleep</c> and <c>load_save</c>, which include saving/loading time during which no ticks are raised.</summary>
        public int SlowMacroTimeoutSeconds { get; set; } = 28;

        /// <summary>If lockstep is enabled and no command arrives for this many seconds, release the lock (0 = never).</summary>
        public int LockstepIdleTimeoutSeconds { get; set; } = 600;

        /// <summary>While recording, sample a snapshot every this many ticks (a line is written only if something changed).</summary>
        public int RecordEveryTicks { get; set; } = 15;
    }
}
