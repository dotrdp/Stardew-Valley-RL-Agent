using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace RLBridge
{
    /// <summary>Small tolerant readers/writers over System.Text.Json nodes.</summary>
    internal static class Json
    {
        public static string? Str(JsonObject? obj, string key)
        {
            try
            {
                JsonNode? node = obj?[key];
                return node?.GetValue<string>();
            }
            catch (Exception)
            {
                return null;
            }
        }

        public static bool Bool(JsonObject? obj, string key, bool fallback = false)
        {
            try
            {
                JsonNode? node = obj?[key];
                return node == null ? fallback : node.GetValue<bool>();
            }
            catch (Exception)
            {
                return fallback;
            }
        }

        public static long? NullableLong(JsonObject? obj, string key)
        {
            JsonNode? node;
            try
            {
                node = obj?[key];
            }
            catch (Exception)
            {
                return null;
            }
            if (node is not JsonValue value)
                return null;
            if (value.TryGetValue(out long l))
                return l;
            if (value.TryGetValue(out double d) && !double.IsNaN(d) && !double.IsInfinity(d))
                return (long)Math.Round(d);
            if (value.TryGetValue(out string? s) && long.TryParse(s, out long parsed))
                return parsed;
            return null;
        }

        public static int? NullableInt(JsonObject? obj, string key)
        {
            long? value = NullableLong(obj, key);
            return value == null ? null : (int)Math.Clamp(value.Value, int.MinValue, int.MaxValue);
        }

        public static long Long(JsonObject? obj, string key, long fallback = 0)
        {
            return NullableLong(obj, key) ?? fallback;
        }

        public static int Int(JsonObject? obj, string key, int fallback = 0)
        {
            return NullableInt(obj, key) ?? fallback;
        }

        public static JsonObject? Obj(JsonObject? obj, string key)
        {
            try
            {
                return obj?[key] as JsonObject;
            }
            catch (Exception)
            {
                return null;
            }
        }

        /// <summary>Serialize a flat event record (string/number/bool/null values) to a JSON string.</summary>
        public static string Serialize(Dictionary<string, object?> fields)
        {
            return JsonSerializer.Serialize(fields);
        }
    }
}
