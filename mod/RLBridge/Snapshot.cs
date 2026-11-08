using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewValley;
using StardewValley.Buildings; // (CollisionMask is used with these namespaces imported in Data Layers / SMAPI's facades)
using StardewValley.Extensions;
using StardewValley.Locations;
using SObject = StardewValley.Object;

namespace RLBridge
{
    /// <summary>One required ingredient of a bundle, parsed from <c>Data/Bundles</c> / <c>netWorldState.BundleData</c>.</summary>
    internal sealed class BundleIngredient
    {
        /// <summary>The raw ID from the data: an unqualified/qualified item ID, a negative category, or -1 for gold.</summary>
        public string RawId = "";

        /// <summary>The ID reported to the agent: qualified item ID (e.g. <c>(O)24</c>), or the raw negative number.</summary>
        public string Id = "";
        public int Qty;
        public int Quality;
        public bool IsMoney => this.RawId == "-1";
        public int? Category;
    }

    /// <summary>One Community Center bundle.</summary>
    internal sealed class BundleInfo
    {
        public int Id;
        public string Key = "";
        public string Area = "";
        public string Name = "";
        public List<BundleIngredient> Ingredients = new();
        public int Need;
    }

    /// <summary>Builds the protocol's Snapshot object (see docs/BRIDGE_PROTOCOL.md). Game thread only.</summary>
    internal static class Snapshot
    {
        /// <summary>The six Community Center rooms in area-index order, with their completion mail flags.</summary>
        /// <remarks>Area names as in bundle keys and flag names verified via Lookup Anything / CJB Cheats Menu (MIT).</remarks>
        public static readonly (string Area, int Index, string Flag)[] Areas =
        {
            ("Pantry", CommunityCenter.AREA_Pantry, "ccPantry"),
            ("Crafts Room", CommunityCenter.AREA_CraftsRoom, "ccCraftsRoom"),
            ("Fish Tank", CommunityCenter.AREA_FishTank, "ccFishTank"),
            ("Boiler Room", CommunityCenter.AREA_BoilerRoom, "ccBoilerRoom"),
            ("Vault", CommunityCenter.AREA_Vault, "ccVault"),
            ("Bulletin Board", CommunityCenter.AREA_Bulletin, "ccBulletin"),
        };

        public const string JojaArea = "Abandoned Joja Mart";

        /// <summary>Build a full snapshot. When no save is loaded, every field is present with empty/zero values.</summary>
        public static JsonObject Build(ulong tick, int radius)
        {
            bool ready = Context.IsWorldReady && Game1.player != null;
            return new JsonObject
            {
                ["tick"] = tick,
                ["day"] = ready ? Game1.dayOfMonth : 0,
                ["season"] = ready ? Game1.season.ToString().ToLowerInvariant() : "",
                ["year"] = ready ? Game1.year : 0,
                ["time"] = ready ? Game1.timeOfDay : 0,
                ["weather"] = ready ? WeatherToday() : "",
                ["weather_tomorrow"] = ready ? WeatherTomorrow() : "",
                ["player"] = ready ? Player() : EmptyPlayer(),
                ["location"] = ready && Game1.currentLocation != null ? Window(Game1.currentLocation, radius) : EmptyWindow(radius),
                ["community_center"] = ready ? CommunityCenterState() : EmptyCommunityCenter(),
                ["menu"] = Game1.activeClickableMenu?.GetType().Name,
                ["event_active"] = ready && Game1.eventUp,
            };
        }

        /// <summary>The absolute day number (days played), used to detect a new day.</summary>
        public static int DaysPlayed()
        {
            return WorldDate.Now().TotalDays;
        }

        /*********
        ** Weather
        *********/
        private static string WeatherToday()
        {
            string context = Game1.currentLocation?.GetLocationContextId() ?? "Default";
            var weather = Game1.netWorldState.Value.GetWeatherForLocation(context);
            return MapWeather(weather?.weather?.Value);
        }

        private static string WeatherTomorrow()
        {
            // same forecast logic as CJB Cheats Menu's weather cheat (MIT)
            WorldDate tomorrow = WorldDate.ForDaysPlayed(WorldDate.Now().TotalDays + 1);
            string forecast = Game1.IsMasterGame
                ? Game1.getWeatherModificationsForDate(tomorrow, Game1.weatherForTomorrow)
                : Game1.getWeatherModificationsForDate(tomorrow, Game1.netWorldState.Value.WeatherForTomorrow);
            return MapWeather(forecast);
        }

        /// <summary>Map the game's weather IDs to the protocol's lower-case names.</summary>
        private static string MapWeather(string? id)
        {
            switch (id)
            {
                case null:
                case "":
                case Game1.weather_sunny:
                case Game1.weather_festival:
                case Game1.weather_wedding:
                    return "sunny";
                case Game1.weather_rain:
                    return "rain";
                case Game1.weather_lightning:
                    return "storm";
                case Game1.weather_snow:
                    return "snow";
                case Game1.weather_debris:
                    return "windy";
                case Game1.weather_green_rain:
                    return "green_rain";
                default:
                    return id.ToLowerInvariant();
            }
        }

        /*********
        ** Player
        *********/
        private static JsonObject Player()
        {
            Farmer p = Game1.player;
            var items = new JsonArray();
            for (int i = 0; i < p.Items.Count; i++)
            {
                Item? item = p.Items[i];
                if (item == null)
                    continue;
                items.Add(new JsonObject
                {
                    ["slot"] = i,
                    ["id"] = item.QualifiedItemId,
                    ["name"] = item.Name,
                    ["stack"] = Math.Max(0, item.Stack),
                    ["quality"] = item is SObject obj ? Math.Max(0, obj.Quality) : 0,
                });
            }

            Point tile = p.TilePoint;
            return new JsonObject
            {
                ["location"] = p.currentLocation?.NameOrUniqueName ?? "",
                ["tile"] = new JsonArray(JsonValue.Create(tile.X), JsonValue.Create(tile.Y)),
                ["facing"] = Math.Clamp(p.FacingDirection, 0, 3),
                ["stamina"] = p.stamina,
                ["max_stamina"] = Math.Max(0, p.MaxStamina),
                ["money"] = (long)p.Money,
                ["current_slot"] = Math.Max(0, p.CurrentToolIndex),
                ["items"] = items,
                ["skills"] = new JsonObject
                {
                    ["farming"] = p.FarmingLevel,
                    ["fishing"] = p.FishingLevel,
                    ["foraging"] = p.ForagingLevel,
                    ["mining"] = p.MiningLevel,
                    ["combat"] = p.CombatLevel,
                },
            };
        }

        private static JsonObject EmptyPlayer()
        {
            return new JsonObject
            {
                ["location"] = "",
                ["tile"] = new JsonArray(JsonValue.Create(0), JsonValue.Create(0)),
                ["facing"] = 2,
                ["stamina"] = 0f,
                ["max_stamina"] = 0,
                ["money"] = 0L,
                ["current_slot"] = 0,
                ["items"] = new JsonArray(),
                ["skills"] = new JsonObject(),
            };
        }

        /*********
        ** Map window
        *********/
        public static (int Width, int Height) MapSize(GameLocation location)
        {
            var layer = location.Map.Layers[0];
            return (layer.LayerWidth, layer.LayerHeight);
        }

        private static JsonObject Window(GameLocation location, int radius)
        {
            (int width, int height) = MapSize(location);
            Point center = Game1.player.TilePoint;
            var sb = new StringBuilder((2 * radius + 1) * (2 * radius + 1));
            for (int y = center.Y - radius; y <= center.Y + radius; y++)
            {
                for (int x = center.X - radius; x <= center.X + radius; x++)
                {
                    bool open = (x == center.X && y == center.Y) || IsPassable(location, x, y, width, height);
                    sb.Append(open ? '1' : '0');
                }
            }

            return new JsonObject
            {
                ["name"] = location.NameOrUniqueName,
                ["width"] = width,
                ["height"] = height,
                ["radius"] = radius,
                ["passable"] = sb.ToString(),
            };
        }

        private static JsonObject EmptyWindow(int radius)
        {
            return new JsonObject
            {
                ["name"] = "",
                ["width"] = 0,
                ["height"] = 0,
                ["radius"] = radius,
                ["passable"] = new string('0', (2 * radius + 1) * (2 * radius + 1)),
            };
        }

        /// <summary>Whether the player could stand on a tile: map passability plus objects/furniture/etc. on it.</summary>
        /// <remarks>
        /// Same rule as Data Layers' "accessible" layer (Pathoschild, MIT): a Buildings-layer <c>Passable</c> property
        /// makes a tile passable, a Back-layer <c>Passable</c> property makes it impassable, else
        /// <c>GameLocation.isTilePassable(Vector2)</c>; then <c>IsTileBlockedBy</c> marks occupied tiles.
        /// </remarks>
        public static bool IsPassable(GameLocation location, int x, int y, int width, int height)
        {
            if (x < 0 || y < 0 || x >= width || y >= height)
                return false;

            var tile = new Vector2(x, y);
            bool passable;
            if (location.doesTileHaveProperty(x, y, "Passable", "Buildings") != null)
                passable = true;
            else if (location.doesTileHaveProperty(x, y, "Passable", "Back") != null)
                passable = false;
            else
                passable = location.isTilePassable(tile);

            return passable && !location.IsTileBlockedBy(tile, ignorePassables: CollisionMask.All);
        }

        /*********
        ** Community Center
        *********/
        public static CommunityCenter? GetCommunityCenter()
        {
            return Game1.getLocationFromName("CommunityCenter") as CommunityCenter;
        }

        /// <summary>Parse the bundle definitions of the current save.</summary>
        /// <remarks>
        /// Format (as parsed by Lookup Anything, MIT): key <c>"{area}/{id}"</c>; value fields split by '/':
        /// [0] name, [1] reward, [2] ingredients as space-separated triplets "{itemId} {count} {quality}".
        /// </remarks>
        public static List<BundleInfo> ParseBundles(IMonitor? monitor = null)
        {
            var result = new List<BundleInfo>();
            foreach (var pair in Game1.netWorldState.Value.BundleData)
            {
                string key = pair.Key;
                string? value = pair.Value;
                if (value == null)
                    continue;
                try
                {
                    int slash = key.LastIndexOf('/');
                    if (slash <= 0 || !int.TryParse(key.Substring(slash + 1), out int id))
                        continue;

                    string[] fields = value.Split('/');
                    var bundle = new BundleInfo
                    {
                        Id = id,
                        Key = key,
                        Area = key.Substring(0, slash),
                        Name = fields.Length > 0 ? fields[0] : key,
                    };

                    string[] raw = fields.Length > 2 ? fields[2].Split(' ', StringSplitOptions.RemoveEmptyEntries) : Array.Empty<string>();
                    for (int i = 0; i + 2 < raw.Length; i += 3)
                    {
                        string itemId = raw[i];
                        var ingredient = new BundleIngredient
                        {
                            RawId = itemId,
                            Id = QualifyBundleItemId(itemId),
                            Qty = int.TryParse(raw[i + 1], out int qty) ? qty : 1,
                            Quality = int.TryParse(raw[i + 2], out int quality) ? quality : 0,
                        };
                        if (itemId != "-1" && int.TryParse(itemId, out int numeric) && numeric < 0)
                            ingredient.Category = numeric;
                        bundle.Ingredients.Add(ingredient);
                    }

                    // UNVERIFIED: field [4], when present, is the number of ingredients needed to complete the bundle
                    // (e.g. 'any 4 of these 6'); otherwise all ingredients are needed.
                    bundle.Need = fields.Length > 4 && int.TryParse(fields[4], out int need) && need > 0
                        ? Math.Min(need, bundle.Ingredients.Count)
                        : bundle.Ingredients.Count;

                    result.Add(bundle);
                }
                catch (Exception ex)
                {
                    monitor?.Log($"Couldn't parse bundle '{key}': {ex.Message}", LogLevel.Trace);
                }
            }

            result.Sort((a, b) => a.Id.CompareTo(b.Id));
            return result;
        }

        /// <summary>Unqualified bundle item IDs are object IDs: '24' -> '(O)24'. Negative values (category / gold) stay raw.</summary>
        private static string QualifyBundleItemId(string id)
        {
            if (id.StartsWith("("))
                return id;
            if (int.TryParse(id, out int n) && n < 0)
                return id;
            return "(O)" + id;
        }

        /// <summary>The per-ingredient donated flags (verified: <c>CommunityCenter.bundles.TryGetValue(id, out bool[])</c>).</summary>
        public static bool[] DoneFlags(CommunityCenter cc, int bundleId)
        {
            return cc.bundles.TryGetValue(bundleId, out bool[] flags) && flags != null ? flags : Array.Empty<bool>();
        }

        public static bool IsBundleComplete(CommunityCenter cc, int bundleId)
        {
            try
            {
                return cc.isBundleComplete(bundleId);
            }
            catch (Exception)
            {
                return false; // invalid bundle data
            }
        }

        /// <summary>Whether the whole Community Center is done (final flag, all rooms flagged, or all areas complete).</summary>
        public static bool IsCommunityCenterComplete(CommunityCenter? cc)
        {
            // 'ccIsComplete' / hasCompletedCommunityCenter() as used by Content Patcher and CJB Cheats Menu (MIT)
            Farmer master = Game1.MasterPlayer;
            if (master.mailReceived.Contains("ccIsComplete") || master.hasCompletedCommunityCenter())
                return true;
            try
            {
                return cc != null && cc.areAllAreasComplete();
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static JsonObject CommunityCenterState()
        {
            CommunityCenter? cc = GetCommunityCenter();
            var areas = new JsonArray();
            var bundles = new JsonArray();
            if (cc != null)
            {
                for (int i = 0; i < Math.Min(cc.areasComplete.Length, Areas.Length); i++)
                    areas.Add(JsonValue.Create(cc.areasComplete[i]));

                foreach (BundleInfo bundle in ParseBundles())
                {
                    if (bundle.Area == JojaArea)
                        continue; // not part of the Community Center objective

                    bool[] done = DoneFlags(cc, bundle.Id);
                    var items = new JsonArray();
                    for (int i = 0; i < bundle.Ingredients.Count; i++)
                    {
                        BundleIngredient ing = bundle.Ingredients[i];
                        items.Add(new JsonObject
                        {
                            ["id"] = ing.Id,
                            ["qty"] = Math.Max(0, ing.Qty),
                            ["quality"] = Math.Max(0, ing.Quality),
                            ["done"] = i < done.Length && done[i],
                        });
                    }

                    bundles.Add(new JsonObject
                    {
                        ["key"] = bundle.Key,
                        ["name"] = bundle.Name,
                        ["area"] = bundle.Area,
                        ["items"] = items,
                        ["need"] = bundle.Need,
                        ["complete"] = IsBundleComplete(cc, bundle.Id),
                    });
                }
            }

            return new JsonObject
            {
                ["complete"] = IsCommunityCenterComplete(cc),
                ["areas_complete"] = areas,
                ["bundles"] = bundles,
            };
        }

        private static JsonObject EmptyCommunityCenter()
        {
            return new JsonObject
            {
                ["complete"] = false,
                ["areas_complete"] = new JsonArray(),
                ["bundles"] = new JsonArray(),
            };
        }
    }
}
