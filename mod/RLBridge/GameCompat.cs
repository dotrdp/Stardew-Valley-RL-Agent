using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using StardewValley;

namespace RLBridge
{
    /// <summary>
    /// Every game member this mod uses but could NOT verify against the official docs or an
    /// open-source (MIT) mod is reached through this class, by reflection, so that a wrong guess
    /// cannot break the build: it fails at runtime with a readable message instead.
    /// Each call site is marked <c>UNVERIFIED</c>.
    /// </summary>
    internal static class GameCompat
    {
        private const BindingFlags Instance = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        private const BindingFlags Static = BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic;

        /*********
        ** Specific members
        *********/
        // UNVERIFIED: Game1.pressUseToolButton() (static) starts using the current tool as the use-tool key would.
        public static bool PressUseToolButton(out string error)
        {
            return TryInvokeStatic(typeof(Game1), "pressUseToolButton", out _, out error);
        }

        // UNVERIFIED: Game1.NewDay(float timeToPause) (static) ends the day (fade, shipping menu, save, next morning).
        public static bool NewDay(out string error)
        {
            return TryInvokeStatic(typeof(Game1), "NewDay", out _, out error, 0f);
        }

        // UNVERIFIED: Game1.ExitToTitle(Action onExit = null) (static) unloads the save and returns to the title screen.
        public static bool ExitToTitle(out string error)
        {
            return TryInvokeStatic(typeof(Game1), "ExitToTitle", out _, out error);
        }

        // UNVERIFIED: SaveGame.Load(string filename, ...) (static) starts loading the save folder `filename`
        // (what the title screen's load menu does when a slot is clicked).
        public static bool LoadSave(string saveFolderName, out string error)
        {
            return TryInvokeStatic(typeof(SaveGame), "Load", out _, out error, saveFolderName);
        }

        // UNVERIFIED: Options.pauseWhenOutOfFocus (bool field or property).
        public static bool SetPauseWhenOutOfFocus(bool value)
        {
            return Game1.options != null && SetMember(Game1.options, "pauseWhenOutOfFocus", value);
        }

        // UNVERIFIED: Character.controller holds the active PathFindController and the character follows it each update,
        // clearing it when the path is finished.
        public static object? GetController(Character character)
        {
            return GetMember(character, "controller");
        }

        // UNVERIFIED: see GetController.
        public static bool SetController(Character character, object? controller)
        {
            return SetMember(character, "controller", controller);
        }

        // UNVERIFIED: PathFindController.pathToEndPoint (Stack<Point>) is null/empty when no path was found.
        /// <returns>False if no path was found, true if one was, null if the member doesn't exist (unknown).</returns>
        public static bool? ControllerHasPath(object controller)
        {
            if (!HasMember(controller, "pathToEndPoint"))
                return null;
            object? path = GetMember(controller, "pathToEndPoint");
            return path is ICollection c ? c.Count > 0 : path != null;
        }

        // UNVERIFIED: GameLocation.warps is the list of touch warps (StardewValley.Warp, whose X/Y/TargetName are verified).
        public static IEnumerable<Warp> GetWarps(GameLocation location)
        {
            return GetMember(location, "warps") is IEnumerable list
                ? list.OfType<Warp>().ToList()
                : Enumerable.Empty<Warp>();
        }

        // UNVERIFIED: ShippingMenu has an instance method okClicked() equivalent to clicking its OK button.
        public static bool TryClickOk(object menu, out string error)
        {
            return TryInvoke(menu, "okClicked", out error);
        }


        /*********
        ** Generic helpers
        *********/
        /// <summary>Whether the object's type has a field or (non-indexed) property with this name.</summary>
        public static bool HasMember(object target, string name)
        {
            for (Type? t = target.GetType(); t != null; t = t.BaseType)
            {
                if (t.GetField(name, Instance | BindingFlags.DeclaredOnly) != null)
                    return true;
                PropertyInfo? p = t.GetProperty(name, Instance | BindingFlags.DeclaredOnly);
                if (p != null && p.GetIndexParameters().Length == 0)
                    return true;
            }
            return false;
        }

        /// <summary>Get a field or property value by name (public or not), or null.</summary>
        public static object? GetMember(object target, string name)
        {
            try
            {
                for (Type? t = target.GetType(); t != null; t = t.BaseType)
                {
                    FieldInfo? f = t.GetField(name, Instance | BindingFlags.DeclaredOnly);
                    if (f != null)
                        return f.GetValue(target);
                    PropertyInfo? p = t.GetProperty(name, Instance | BindingFlags.DeclaredOnly);
                    if (p != null && p.GetIndexParameters().Length == 0)
                        return p.GetValue(target);
                }
            }
            catch (Exception)
            {
                // fall through
            }
            return null;
        }

        /// <summary>Set a field or property by name (public or not). Returns false if missing or not assignable.</summary>
        public static bool SetMember(object target, string name, object? value)
        {
            try
            {
                for (Type? t = target.GetType(); t != null; t = t.BaseType)
                {
                    FieldInfo? f = t.GetField(name, Instance | BindingFlags.DeclaredOnly);
                    if (f != null && !f.IsInitOnly && IsAssignable(f.FieldType, value))
                    {
                        f.SetValue(target, value);
                        return true;
                    }
                    PropertyInfo? p = t.GetProperty(name, Instance | BindingFlags.DeclaredOnly);
                    if (p != null && p.CanWrite && p.GetIndexParameters().Length == 0 && IsAssignable(p.PropertyType, value))
                    {
                        p.SetValue(target, value);
                        return true;
                    }
                }
            }
            catch (Exception)
            {
                // fall through
            }
            return false;
        }

        /// <summary>Invoke an instance method by name; trailing optional parameters get their defaults.</summary>
        public static bool TryInvoke(object target, string name, out string error, params object?[] args)
        {
            MethodInfo? method = FindMethod(target.GetType(), name, Instance, args, out object?[] fullArgs);
            if (method == null)
            {
                error = $"{target.GetType().Name}.{name}({args.Length} args) not found";
                return false;
            }
            try
            {
                method.Invoke(target, fullArgs);
                error = "";
                return true;
            }
            catch (TargetInvocationException ex)
            {
                error = $"{target.GetType().Name}.{name} failed: {ex.InnerException?.Message ?? ex.Message}";
                return false;
            }
        }

        /// <summary>Invoke a static method by name; trailing optional parameters get their defaults.</summary>
        public static bool TryInvokeStatic(Type type, string name, out object? result, out string error, params object?[] args)
        {
            result = null;
            MethodInfo? method = FindMethod(type, name, Static, args, out object?[] fullArgs);
            if (method == null)
            {
                error = $"{type.Name}.{name}({args.Length} args) not found";
                return false;
            }
            try
            {
                result = method.Invoke(null, fullArgs);
                error = "";
                return true;
            }
            catch (TargetInvocationException ex)
            {
                error = $"{type.Name}.{name} failed: {ex.InnerException?.Message ?? ex.Message}";
                return false;
            }
        }

        private static MethodInfo? FindMethod(Type type, string name, BindingFlags flags, object?[] args, out object?[] fullArgs)
        {
            foreach (MethodInfo method in type.GetMethods(flags).Where(m => m.Name == name && !m.IsGenericMethodDefinition).OrderBy(m => m.GetParameters().Length))
            {
                ParameterInfo[] ps = method.GetParameters();
                if (ps.Length < args.Length)
                    continue;

                bool ok = true;
                for (int i = 0; i < ps.Length && ok; i++)
                    ok = i < args.Length ? IsAssignable(ps[i].ParameterType, args[i]) : ps[i].IsOptional;
                if (!ok)
                    continue;

                fullArgs = new object?[ps.Length];
                for (int i = 0; i < ps.Length; i++)
                    fullArgs[i] = i < args.Length ? args[i] : (ps[i].HasDefaultValue ? ps[i].DefaultValue : Type.Missing);
                return method;
            }

            fullArgs = Array.Empty<object?>();
            return null;
        }

        private static bool IsAssignable(Type type, object? value)
        {
            return value == null
                ? !type.IsValueType || Nullable.GetUnderlyingType(type) != null
                : type.IsInstanceOfType(value);
        }
    }
}
