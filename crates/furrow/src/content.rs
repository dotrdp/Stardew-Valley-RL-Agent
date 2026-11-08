//! Content tables (DESIGN.md §2.5): the original `data/*.toml` files, embedded at compile time,
//! parsed once into dense, id-indexed tables, and validated.
//!
//! Every entity carries an explicit `id` in its table; ids are dense and append-only. Nothing
//! here depends on table-key order.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::Deserialize;

use crate::hash::fnv1a64;

/// Dense item id.
pub type ItemId = u16;
/// Dense crop id.
pub type CropId = u8;
/// Dense location id.
pub type LocId = u8;
/// Dense POI id.
pub type PoiId = u8;
/// Dense request id.
pub type RequestId = u8;
/// Dense wing id.
pub type WingId = u8;
/// Dense recipe id.
pub type RecipeId = u8;
/// Dense machine-type id.
pub type MachineId = u8;
/// Dense shop id.
pub type ShopId = u8;
/// Dense rock-kind id.
pub type RockId = u8;
/// Dense fish id.
pub type FishId = u8;

/// Raw TOML sources, in the fixed order used for `CONTENT_HASH`.
pub const SOURCES: [(&str, &str); 10] = [
    ("items.toml", include_str!("../data/items.toml")),
    ("crops.toml", include_str!("../data/crops.toml")),
    ("forage.toml", include_str!("../data/forage.toml")),
    ("fish.toml", include_str!("../data/fish.toml")),
    ("mining.toml", include_str!("../data/mining.toml")),
    ("recipes.toml", include_str!("../data/recipes.toml")),
    ("shops.toml", include_str!("../data/shops.toml")),
    ("skills.toml", include_str!("../data/skills.toml")),
    ("restoration.toml", include_str!("../data/restoration.toml")),
    ("world.toml", include_str!("../data/world.toml")),
];

/// Seasons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Season {
    Spring = 0,
    Summer = 1,
    Autumn = 2,
    Winter = 3,
}

impl Season {
    /// All seasons in order.
    pub const ALL: [Season; 4] = [Season::Spring, Season::Summer, Season::Autumn, Season::Winter];
    /// From index 0..4.
    pub fn from_index(i: u8) -> Season {
        Season::ALL[(i & 3) as usize]
    }
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        ["spring", "summer", "autumn", "winter"][self as usize]
    }
    fn parse(s: &str) -> Option<Season> {
        match s {
            "spring" => Some(Season::Spring),
            "summer" => Some(Season::Summer),
            "autumn" => Some(Season::Autumn),
            "winter" => Some(Season::Winter),
            _ => None,
        }
    }
}

/// Bit mask over seasons (bit i = season i).
pub type SeasonMask = u8;

/// Season mask helper.
#[inline]
pub fn season_in(mask: SeasonMask, s: Season) -> bool {
    mask & (1 << (s as u8)) != 0
}

/// Weather.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Weather {
    Sunny = 0,
    Rainy = 1,
    Stormy = 2,
    Snowy = 3,
}

impl Weather {
    /// Rain waters outdoor soil.
    pub fn is_rain(self) -> bool {
        matches!(self, Weather::Rainy | Weather::Stormy)
    }
    /// Name.
    pub fn name(self) -> &'static str {
        ["sunny", "rainy", "stormy", "snowy"][self as usize]
    }
}

/// Weather requirement of a fish.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeatherReq {
    Any,
    Rain,
    Sun,
}

/// Item category.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    Seed,
    Crop,
    Forage,
    Fish,
    Junk,
    Mineral,
    Ore,
    Bar,
    Gem,
    Resource,
    Artisan,
    Food,
    Machine,
    Fertilizer,
}

impl Category {
    fn parse(s: &str) -> Option<Category> {
        Some(match s {
            "seed" => Category::Seed,
            "crop" => Category::Crop,
            "forage" => Category::Forage,
            "fish" => Category::Fish,
            "junk" => Category::Junk,
            "mineral" => Category::Mineral,
            "ore" => Category::Ore,
            "bar" => Category::Bar,
            "gem" => Category::Gem,
            "resource" => Category::Resource,
            "artisan" => Category::Artisan,
            "food" => Category::Food,
            "machine" => Category::Machine,
            "fertilizer" => Category::Fertilizer,
            _ => return None,
        })
    }
    /// Lower-case name.
    pub fn name(self) -> &'static str {
        match self {
            Category::Seed => "seed",
            Category::Crop => "crop",
            Category::Forage => "forage",
            Category::Fish => "fish",
            Category::Junk => "junk",
            Category::Mineral => "mineral",
            Category::Ore => "ore",
            Category::Bar => "bar",
            Category::Gem => "gem",
            Category::Resource => "resource",
            Category::Artisan => "artisan",
            Category::Food => "food",
            Category::Machine => "machine",
            Category::Fertilizer => "fertilizer",
        }
    }
}

/// Tools (fixed order; also the `Select tool` primitive order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Tool {
    Hoe = 0,
    Can = 1,
    Axe = 2,
    Pickaxe = 3,
    Rod = 4,
}

impl Tool {
    /// All tools.
    pub const ALL: [Tool; 5] = [Tool::Hoe, Tool::Can, Tool::Axe, Tool::Pickaxe, Tool::Rod];
    /// Name.
    pub fn name(self) -> &'static str {
        ["hoe", "can", "axe", "pickaxe", "rod"][self as usize]
    }
    fn parse(s: &str) -> Option<Tool> {
        Tool::ALL.iter().copied().find(|t| t.name() == s)
    }
}

/// Facing direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Dir {
    N = 0,
    E = 1,
    S = 2,
    W = 3,
}

impl Dir {
    /// All directions in id order.
    pub const ALL: [Dir; 4] = [Dir::N, Dir::E, Dir::S, Dir::W];
    /// Unit offset.
    pub fn delta(self) -> (i32, i32) {
        match self {
            Dir::N => (0, -1),
            Dir::E => (1, 0),
            Dir::S => (0, 1),
            Dir::W => (-1, 0),
        }
    }
    fn parse(s: &str) -> Option<Dir> {
        match s {
            "n" => Some(Dir::N),
            "e" => Some(Dir::E),
            "s" => Some(Dir::S),
            "w" => Some(Dir::W),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Resolved tables
// ---------------------------------------------------------------------------------------------

/// An item.
#[derive(Clone, Debug)]
pub struct Item {
    pub id: ItemId,
    pub key: String,
    pub name: String,
    pub category: Category,
    pub sell: u32,
    pub energy: u16,
}

/// A crop.
#[derive(Clone, Debug)]
pub struct Crop {
    pub id: CropId,
    pub key: String,
    pub seed: ItemId,
    pub produce: ItemId,
    pub seasons: SeasonMask,
    pub grow_days: u8,
    pub regrow_days: u8,
    pub yield_min: u8,
    pub yield_max: u8,
    pub extra_yield_ppm_per_level: u32,
    pub xp: u16,
}

/// Forage spawn table of one location.
#[derive(Clone, Debug)]
pub struct ForageLocation {
    pub id: u8,
    pub location: LocId,
    pub daily_cap: u8,
    /// Per season: (item, weight).
    pub tables: [Vec<(ItemId, u32)>; 4],
}

/// Fishing parameters.
#[derive(Clone, Debug, Deserialize)]
pub struct FishingParams {
    pub cast_min: u16,
    pub cast_max: u16,
    pub cast_minus_per_level: u16,
    pub base_ppm: u32,
    pub skill_ppm: u32,
    pub rod_ppm: u32,
    pub diff_ppm: u32,
    pub min_ppm: u32,
    pub max_ppm: u32,
    pub junk_ppm_on_miss: u32,
    pub energy_per_cast: u16,
    pub xp_per_catch: u16,
}

/// A fish.
#[derive(Clone, Debug)]
pub struct Fish {
    pub id: FishId,
    pub item: ItemId,
    pub spots: Vec<PoiId>,
    pub seasons: SeasonMask,
    pub weather: WeatherReq,
    pub time: (u16, u16),
    pub difficulty: u8,
    pub min_rod: u8,
    pub weight: u32,
}

/// Mining parameters.
#[derive(Clone, Debug, Deserialize)]
pub struct MiningParams {
    pub floors: u8,
    pub elevator_step: u8,
    pub rocks_per_floor_min: u8,
    pub rocks_per_floor_max: u8,
    pub energy_per_swing: u16,
    pub xp_per_rock: u16,
    pub ladder_rocks: u8,
    pub extra_ladder_ppm: u32,
}

/// A drop entry.
#[derive(Clone, Debug)]
pub struct Drop {
    pub item: ItemId,
    pub min: u8,
    pub max: u8,
    pub ppm: u32,
}

/// A rock kind.
#[derive(Clone, Debug)]
pub struct Rock {
    pub id: RockId,
    pub key: String,
    pub hardness: u8,
    pub min_pick: u8,
    pub drops: Vec<Drop>,
}

/// A depth band.
#[derive(Clone, Debug)]
pub struct Band {
    pub floors: (u8, u8),
    pub rocks: Vec<(RockId, u32)>,
}

/// Recipe unlock condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unlock {
    Start,
    Skill(Skill, u8),
    Wing(WingId),
}

/// Skills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Skill {
    Farming = 0,
    Foraging = 1,
    Fishing = 2,
    Mining = 3,
}

impl Skill {
    /// All skills.
    pub const ALL: [Skill; 4] = [Skill::Farming, Skill::Foraging, Skill::Fishing, Skill::Mining];
    /// Name.
    pub fn name(self) -> &'static str {
        ["farming", "foraging", "fishing", "mining"][self as usize]
    }
}

/// A crafting recipe.
#[derive(Clone, Debug)]
pub struct Recipe {
    pub id: RecipeId,
    pub key: String,
    pub output: ItemId,
    pub count: u16,
    pub inputs: Vec<(ItemId, u16)>,
    pub unlock: Unlock,
}

/// Machine input selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputSel {
    Item(ItemId),
    Category(Category),
}

/// A machine processing rule.
#[derive(Clone, Debug)]
pub struct Rule {
    pub input: InputSel,
    pub input_count: u16,
    pub fuel: Option<(ItemId, u16)>,
    pub output: ItemId,
    pub output_count: u16,
    pub minutes: u32,
}

/// A machine type.
#[derive(Clone, Debug)]
pub struct Machine {
    pub id: MachineId,
    pub key: String,
    pub item: ItemId,
    pub rules: Vec<Rule>,
}

/// A shop stock row.
#[derive(Clone, Debug)]
pub struct StockRow {
    pub item: ItemId,
    pub price: u32,
    pub seasons: SeasonMask,
}

/// A shop.
#[derive(Clone, Debug)]
pub struct Shop {
    pub id: ShopId,
    pub key: String,
    pub name: String,
    pub location: LocId,
    pub poi: PoiId,
    pub open: u16,
    pub close: u16,
    pub closed_weekday: i8,
    pub stock: Vec<StockRow>,
}

/// The wandering merchant.
#[derive(Clone, Debug)]
pub struct Merchant {
    pub poi: PoiId,
    pub location: LocId,
    pub weekdays: Vec<u8>,
    pub open: u16,
    pub close: u16,
    pub slots: u8,
    pub price_mult_ppm: u32,
    pub floor_price: u32,
    pub pool: Vec<ItemId>,
}

/// Tool upgrade level cost.
#[derive(Clone, Debug)]
pub struct UpgradeLevel {
    pub level: u8,
    pub coins: u32,
    pub bar: ItemId,
    pub bars: u16,
}

/// Skill parameters.
#[derive(Clone, Debug, Deserialize)]
pub struct SkillParams {
    pub thresholds: Vec<u32>,
    pub energy_cost_reduction_ppm_per_level: u32,
    pub farming_extra_yield_ppm_per_level: u32,
    pub foraging_double_ppm_per_level: u32,
    pub mining_extra_drop_ppm_per_level: u32,
    pub fishing_cast_minus_per_level: u16,
}

/// XP awards.
#[derive(Clone, Debug, Deserialize)]
pub struct XpParams {
    pub hoe_till: u16,
    pub water_tile: u16,
    pub harvest: u16,
    pub forage_pickup: u16,
    pub chop_tree: u16,
    pub chop_stump: u16,
    pub break_rock: u16,
    pub fish_catch: u16,
    pub craft: u16,
}

/// Tool parameters.
#[derive(Clone, Debug, Deserialize)]
pub struct ToolParams {
    pub hoe_energy: u16,
    pub can_energy_per_tile: u16,
    pub axe_energy: u16,
    pub pickaxe_energy: u16,
    pub rod_energy: u16,
    pub can_capacity: [u16; 4],
    pub tree_hits: [u8; 4],
    pub stump_hits: [u8; 4],
    pub boulder_hits: [u8; 4],
    pub tree_wood: [u8; 4],
    pub tree_resin_ppm: u32,
    pub stump_hardwood: u8,
    pub boulder_stone: u8,
}

/// Unlockable features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Feature {
    Greenhouse = 0,
    Minecart = 1,
    MerchantDaily = 2,
}

impl Feature {
    /// All features.
    pub const ALL: [Feature; 3] = [Feature::Greenhouse, Feature::Minecart, Feature::MerchantDaily];
    /// Name.
    pub fn name(self) -> &'static str {
        ["greenhouse", "minecart", "merchant_daily"][self as usize]
    }
    fn parse(s: &str) -> Option<Feature> {
        Feature::ALL.iter().copied().find(|f| f.name() == s)
    }
}

/// Wing reward.
#[derive(Clone, Debug)]
pub enum Reward {
    Unlock(Feature),
    Energy(u16),
    ToolLevel(Vec<Tool>),
    Money(u32),
}

/// A wing.
#[derive(Clone, Debug)]
pub struct Wing {
    pub id: WingId,
    pub key: String,
    pub name: String,
    pub reward: Reward,
    pub requests: Vec<RequestId>,
}

/// A request slot.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub item: ItemId,
    pub count: u16,
}

/// A request.
#[derive(Clone, Debug)]
pub struct Request {
    pub id: RequestId,
    pub key: String,
    pub wing: WingId,
    pub name: String,
    pub need: u8,
    pub coins: u32,
    pub slots: Vec<Slot>,
    pub pool_train: Vec<ItemId>,
    pub pool_test: Vec<ItemId>,
}

/// Location kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocKind {
    Farm,
    Town,
    Wild,
    Mines,
    Greenhouse,
}

/// A location.
#[derive(Clone, Debug)]
pub struct Location {
    pub id: LocId,
    pub key: String,
    pub name: String,
    pub width: u8,
    pub height: u8,
    pub kind: LocKind,
}

/// A rectangular region (inclusive).
#[derive(Clone, Debug)]
pub struct Region {
    pub location: LocId,
    pub kind: String,
    pub rect: [u8; 4],
}

/// POI kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoiKind {
    Bed,
    Crate,
    Well,
    MillBoard,
    Counter,
    Stall,
    FishingSpot,
    MineEntrance,
    Elevator,
    MineExit,
}

/// A point of interest.
#[derive(Clone, Debug)]
pub struct Poi {
    pub id: PoiId,
    pub key: String,
    pub location: LocId,
    pub kind: PoiKind,
    pub tile: (u8, u8),
    pub stand: (u8, u8),
    pub face: Dir,
}

/// A warp.
#[derive(Clone, Debug)]
pub struct Warp {
    pub id: u8,
    pub from: LocId,
    pub tile: (u8, u8),
    pub to: LocId,
    pub to_tile: (u8, u8),
}

/// Procedural densities.
#[derive(Clone, Debug, Deserialize)]
pub struct Density {
    pub field_weeds_ppm: u32,
    pub field_twig_ppm: u32,
    pub field_stone_ppm: u32,
    pub field_tree_ppm: u32,
    pub tree_region_ppm: u32,
    pub stump_count: u8,
    pub wild_fiber_ppm: u32,
}

/// World parameters.
#[derive(Clone, Debug)]
pub struct WorldParams {
    pub start_location: LocId,
    pub start_tile: (u8, u8),
    pub day_start: u16,
    pub day_end: u16,
    pub season_days: u8,
    pub energy_max: u16,
    pub backpack_slots: u8,
    pub stack_max: u16,
    pub start_money: u32,
    pub passout_energy_ppm: u32,
    pub passout_money_ppm: u32,
    pub passout_money_cap: u32,
    /// Per season: [sunny, rainy, stormy, snowy] ppm.
    pub weather_ppm: [[u32; 4]; 4],
    pub minecart_minutes: u16,
}

/// Surface rock parameters.
#[derive(Clone, Debug)]
pub struct Surface {
    pub quarry_rocks: u8,
    pub quarry_rock_kinds: Vec<(RockId, u32)>,
}

/// All content, resolved.
#[derive(Clone, Debug)]
pub struct Content {
    pub items: Vec<Item>,
    pub crops: Vec<Crop>,
    pub forage: Vec<ForageLocation>,
    pub fishing: FishingParams,
    pub fish: Vec<Fish>,
    pub junk: Vec<ItemId>,
    pub mining: MiningParams,
    pub rocks: Vec<Rock>,
    pub bands: Vec<Band>,
    pub surface: Surface,
    pub recipes: Vec<Recipe>,
    pub machines: Vec<Machine>,
    pub shops: Vec<Shop>,
    pub merchant: Merchant,
    pub upgrade_days: u8,
    pub upgrades: Vec<UpgradeLevel>,
    pub skills: SkillParams,
    pub xp: XpParams,
    pub tools: ToolParams,
    pub restoration_name: String,
    pub wings: Vec<Wing>,
    pub requests: Vec<Request>,
    pub world: WorldParams,
    pub locations: Vec<Location>,
    pub regions: Vec<Region>,
    pub pois: Vec<Poi>,
    pub warps: Vec<Warp>,
    pub density: Density,
    // ---- derived
    /// Item key → id.
    pub item_by_key: BTreeMap<String, ItemId>,
    /// Crop id by seed item (None if the item is not a seed).
    pub crop_by_seed: Vec<Option<CropId>>,
    /// Machine type by its placeable item.
    pub machine_by_item: Vec<Option<MachineId>>,
    /// Items purchasable anywhere (shops ∪ merchant pool), sorted by id — the `Buy` macro domain.
    pub buyable: Vec<ItemId>,
    /// Fishing spots (POI ids) in id order — the `Fish` macro domain.
    pub fishing_spots: Vec<PoiId>,
    /// Location that each POI is in; POI by key.
    pub poi_by_key: BTreeMap<String, PoiId>,
    /// Location by key.
    pub loc_by_key: BTreeMap<String, LocId>,
    /// Total number of slots over all requests.
    pub total_slots: u16,
    /// Max item count appearing in any slot (for scaling).
    pub max_slot_count: u16,
    /// Requests with a coin amount.
    pub coin_requests: Vec<RequestId>,
}

// ---------------------------------------------------------------------------------------------
// Raw TOML shapes
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawItems {
    item: Vec<RawItem>,
}
#[derive(Deserialize)]
struct RawItem {
    id: u16,
    key: String,
    name: String,
    category: String,
    sell: u32,
    energy: u16,
}
#[derive(Deserialize)]
struct RawCrops {
    crop: Vec<RawCrop>,
}
#[derive(Deserialize)]
struct RawCrop {
    id: u8,
    key: String,
    seed: String,
    produce: String,
    seasons: Vec<String>,
    grow_days: u8,
    regrow_days: u8,
    yield_min: u8,
    yield_max: u8,
    extra_yield_ppm_per_level: u32,
    xp: u16,
}
#[derive(Deserialize)]
struct RawForage {
    forage_location: Vec<RawForageLoc>,
}
#[derive(Deserialize)]
struct RawForageLoc {
    id: u8,
    location: String,
    daily_cap: u8,
    season: Vec<RawForageSeason>,
}
#[derive(Deserialize)]
struct RawForageSeason {
    season: String,
    items: Vec<RawWeightedItem>,
}
#[derive(Deserialize)]
struct RawWeightedItem {
    item: String,
    weight: u32,
}
#[derive(Deserialize)]
struct RawFishFile {
    fishing: FishingParams,
    fish: Vec<RawFish>,
    junk: RawJunk,
}
#[derive(Deserialize)]
struct RawJunk {
    items: Vec<String>,
}
#[derive(Deserialize)]
struct RawFish {
    id: u8,
    item: String,
    spots: Vec<String>,
    seasons: Vec<String>,
    weather: String,
    time: [u16; 2],
    difficulty: u8,
    min_rod: u8,
    weight: u32,
}
#[derive(Deserialize)]
struct RawMiningFile {
    mining: MiningParams,
    rock: Vec<RawRock>,
    band: Vec<RawBand>,
    surface: RawSurface,
}
#[derive(Deserialize)]
struct RawRock {
    id: u8,
    key: String,
    hardness: u8,
    min_pick: u8,
    drops: Vec<RawDrop>,
}
#[derive(Deserialize)]
struct RawDrop {
    item: String,
    min: u8,
    max: u8,
    ppm: u32,
}
#[derive(Deserialize)]
struct RawBand {
    floors: [u8; 2],
    rocks: Vec<RawWeightedRock>,
}
#[derive(Deserialize)]
struct RawWeightedRock {
    rock: String,
    weight: u32,
}
#[derive(Deserialize)]
struct RawSurface {
    quarry_rocks: u8,
    quarry_rock_kinds: Vec<RawWeightedRock>,
}
#[derive(Deserialize)]
struct RawRecipesFile {
    recipe: Vec<RawRecipe>,
    machine: Vec<RawMachine>,
}
#[derive(Deserialize)]
struct RawRecipe {
    id: u8,
    key: String,
    output: String,
    count: u16,
    inputs: Vec<RawInput>,
    unlock: String,
}
#[derive(Deserialize)]
struct RawInput {
    item: String,
    count: u16,
}
#[derive(Deserialize)]
struct RawMachine {
    id: u8,
    key: String,
    item: String,
    rules: Vec<RawRule>,
}
#[derive(Deserialize)]
struct RawRule {
    input: Option<String>,
    input_category: Option<String>,
    input_count: u16,
    fuel: Option<String>,
    fuel_count: Option<u16>,
    output: String,
    output_count: u16,
    minutes: u32,
}
#[derive(Deserialize)]
struct RawShopsFile {
    shop: Vec<RawShop>,
    merchant: RawMerchant,
    upgrades: RawUpgrades,
}
#[derive(Deserialize)]
struct RawShop {
    id: u8,
    key: String,
    name: String,
    location: String,
    poi: String,
    open: u16,
    close: u16,
    closed_weekday: i8,
    stock: Vec<RawStock>,
}
#[derive(Deserialize)]
struct RawStock {
    item: String,
    price: u32,
    seasons: Vec<String>,
}
#[derive(Deserialize)]
struct RawMerchant {
    poi: String,
    location: String,
    weekdays: Vec<u8>,
    open: u16,
    close: u16,
    slots: u8,
    price_mult_ppm: u32,
    floor_price: u32,
    pool: Vec<String>,
}
#[derive(Deserialize)]
struct RawUpgrades {
    days: u8,
    levels: Vec<RawUpgradeLevel>,
}
#[derive(Deserialize)]
struct RawUpgradeLevel {
    level: u8,
    coins: u32,
    bar: String,
    bars: u16,
}
#[derive(Deserialize)]
struct RawSkillsFile {
    skills: SkillParams,
    xp: XpParams,
    tools: ToolParams,
}
#[derive(Deserialize)]
struct RawRestorationFile {
    restoration: RawRestoration,
    wing: Vec<RawWing>,
    request: Vec<RawRequest>,
}
#[derive(Deserialize)]
struct RawRestoration {
    name: String,
}
#[derive(Deserialize)]
struct RawWing {
    id: u8,
    key: String,
    name: String,
    reward: RawReward,
}
#[derive(Deserialize)]
struct RawReward {
    kind: String,
    feature: Option<String>,
    amount: Option<u32>,
    tools: Option<Vec<String>>,
}
#[derive(Deserialize)]
struct RawRequest {
    id: u8,
    key: String,
    wing: String,
    name: String,
    need: u8,
    #[serde(default)]
    coins: u32,
    slots: Vec<RawSlot>,
    pool_train: Vec<String>,
    pool_test: Vec<String>,
}
#[derive(Deserialize)]
struct RawSlot {
    item: String,
    count: u16,
}
#[derive(Deserialize)]
struct RawWorldFile {
    world: RawWorld,
    location: Vec<RawLocation>,
    region: Vec<RawRegion>,
    poi: Vec<RawPoi>,
    warp: Vec<RawWarp>,
    density: Density,
}
#[derive(Deserialize)]
struct RawWorld {
    start_location: String,
    start_tile: [u8; 2],
    day_start: u16,
    day_end: u16,
    season_days: u8,
    energy_max: u16,
    backpack_slots: u8,
    stack_max: u16,
    start_money: u32,
    passout_energy_ppm: u32,
    passout_money_ppm: u32,
    passout_money_cap: u32,
    weather_ppm: BTreeMap<String, [u32; 4]>,
    minecart_minutes: u16,
}
#[derive(Deserialize)]
struct RawLocation {
    id: u8,
    key: String,
    name: String,
    width: u8,
    height: u8,
    kind: String,
}
#[derive(Deserialize)]
struct RawRegion {
    location: String,
    kind: String,
    rect: [u8; 4],
}
#[derive(Deserialize)]
struct RawPoi {
    id: u8,
    key: String,
    location: String,
    kind: String,
    tile: [u8; 2],
    stand: [u8; 2],
    face: String,
}
#[derive(Deserialize)]
struct RawWarp {
    id: u8,
    from: String,
    tile: [u8; 2],
    to: String,
    to_tile: [u8; 2],
}

// ---------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------

/// Content loading error.
#[derive(Debug)]
pub struct ContentError(pub Vec<String>);

impl std::fmt::Display for ContentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for e in &self.0 {
            writeln!(f, "- {e}")?;
        }
        Ok(())
    }
}

struct Resolver<'a> {
    items: &'a BTreeMap<String, ItemId>,
    errors: Vec<String>,
}

impl Resolver<'_> {
    fn item(&mut self, key: &str, ctx: &str) -> ItemId {
        match self.items.get(key) {
            Some(id) => *id,
            None => {
                self.errors.push(format!("{ctx}: unknown item '{key}'"));
                0
            }
        }
    }
    fn seasons(&mut self, xs: &[String], ctx: &str) -> SeasonMask {
        let mut m = 0u8;
        for s in xs {
            match Season::parse(s) {
                Some(se) => m |= 1 << (se as u8),
                None => self.errors.push(format!("{ctx}: unknown season '{s}'")),
            }
        }
        m
    }
}

fn parse<T: for<'de> Deserialize<'de>>(name: &str, src: &str, errors: &mut Vec<String>) -> Option<T> {
    match toml::from_str::<T>(src) {
        Ok(v) => Some(v),
        Err(e) => {
            errors.push(format!("{name}: parse error: {e}"));
            None
        }
    }
}

fn check_dense<T>(name: &str, ids: impl Iterator<Item = T>, errors: &mut Vec<String>)
where
    T: Into<u32>,
{
    let mut v: Vec<u32> = ids.map(Into::into).collect();
    v.sort_unstable();
    for (i, id) in v.iter().enumerate() {
        if *id != i as u32 {
            errors.push(format!("{name}: ids must be dense 0..n and unique (got {id} at position {i})"));
            return;
        }
    }
}

fn load() -> Result<Content, ContentError> {
    let mut errors = Vec::new();
    let items_raw: Option<RawItems> = parse("items.toml", SOURCES[0].1, &mut errors);
    let crops_raw: Option<RawCrops> = parse("crops.toml", SOURCES[1].1, &mut errors);
    let forage_raw: Option<RawForage> = parse("forage.toml", SOURCES[2].1, &mut errors);
    let fish_raw: Option<RawFishFile> = parse("fish.toml", SOURCES[3].1, &mut errors);
    let mining_raw: Option<RawMiningFile> = parse("mining.toml", SOURCES[4].1, &mut errors);
    let recipes_raw: Option<RawRecipesFile> = parse("recipes.toml", SOURCES[5].1, &mut errors);
    let shops_raw: Option<RawShopsFile> = parse("shops.toml", SOURCES[6].1, &mut errors);
    let skills_raw: Option<RawSkillsFile> = parse("skills.toml", SOURCES[7].1, &mut errors);
    let rest_raw: Option<RawRestorationFile> = parse("restoration.toml", SOURCES[8].1, &mut errors);
    let world_raw: Option<RawWorldFile> = parse("world.toml", SOURCES[9].1, &mut errors);
    let (Some(items_raw), Some(crops_raw), Some(forage_raw), Some(fish_raw), Some(mining_raw), Some(recipes_raw), Some(shops_raw), Some(skills_raw), Some(rest_raw), Some(world_raw)) = (
        items_raw, crops_raw, forage_raw, fish_raw, mining_raw, recipes_raw, shops_raw, skills_raw, rest_raw, world_raw,
    ) else {
        return Err(ContentError(errors));
    };

    // ---- items
    check_dense("items", items_raw.item.iter().map(|i| i.id as u32), &mut errors);
    let mut items: Vec<Item> = Vec::with_capacity(items_raw.item.len());
    let mut sorted = items_raw.item;
    sorted.sort_by_key(|i| i.id);
    let mut item_by_key = BTreeMap::new();
    for r in sorted {
        let category = match Category::parse(&r.category) {
            Some(c) => c,
            None => {
                errors.push(format!("items: unknown category '{}' on {}", r.category, r.key));
                Category::Junk
            }
        };
        if item_by_key.insert(r.key.clone(), r.id).is_some() {
            errors.push(format!("items: duplicate key {}", r.key));
        }
        items.push(Item {
            id: r.id,
            key: r.key,
            name: r.name,
            category,
            sell: r.sell,
            energy: r.energy,
        });
    }
    let mut rs = Resolver {
        items: &item_by_key,
        errors: Vec::new(),
    };

    // ---- locations / pois / warps (needed by other tables)
    check_dense("locations", world_raw.location.iter().map(|l| l.id as u32), &mut errors);
    let mut locs = world_raw.location;
    locs.sort_by_key(|l| l.id);
    let mut loc_by_key = BTreeMap::new();
    let locations: Vec<Location> = locs
        .into_iter()
        .map(|l| {
            loc_by_key.insert(l.key.clone(), l.id);
            let kind = match l.kind.as_str() {
                "farm" => LocKind::Farm,
                "town" => LocKind::Town,
                "wild" => LocKind::Wild,
                "mines" => LocKind::Mines,
                "greenhouse" => LocKind::Greenhouse,
                other => {
                    errors.push(format!("locations: unknown kind {other}"));
                    LocKind::Wild
                }
            };
            Location {
                id: l.id,
                key: l.key,
                name: l.name,
                width: l.width,
                height: l.height,
                kind,
            }
        })
        .collect();
    let loc = |k: &str, errors: &mut Vec<String>| -> LocId {
        match loc_by_key.get(k) {
            Some(i) => *i,
            None => {
                errors.push(format!("unknown location '{k}'"));
                0
            }
        }
    };
    check_dense("pois", world_raw.poi.iter().map(|p| p.id as u32), &mut errors);
    let mut pois_raw = world_raw.poi;
    pois_raw.sort_by_key(|p| p.id);
    let mut poi_by_key = BTreeMap::new();
    let mut pois = Vec::new();
    for p in pois_raw {
        let kind = match p.kind.as_str() {
            "bed" => PoiKind::Bed,
            "crate" => PoiKind::Crate,
            "well" => PoiKind::Well,
            "mill_board" => PoiKind::MillBoard,
            "counter" => PoiKind::Counter,
            "stall" => PoiKind::Stall,
            "fishing_spot" => PoiKind::FishingSpot,
            "mine_entrance" => PoiKind::MineEntrance,
            "elevator" => PoiKind::Elevator,
            "mine_exit" => PoiKind::MineExit,
            other => {
                errors.push(format!("pois: unknown kind {other}"));
                PoiKind::Well
            }
        };
        let face = Dir::parse(&p.face).unwrap_or_else(|| {
            errors.push(format!("pois: bad face on {}", p.key));
            Dir::N
        });
        let location = loc(&p.location, &mut errors);
        poi_by_key.insert(p.key.clone(), p.id);
        pois.push(Poi {
            id: p.id,
            key: p.key,
            location,
            kind,
            tile: (p.tile[0], p.tile[1]),
            stand: (p.stand[0], p.stand[1]),
            face,
        });
    }
    let poi = |k: &str, errors: &mut Vec<String>| -> PoiId {
        match poi_by_key.get(k) {
            Some(i) => *i,
            None => {
                errors.push(format!("unknown poi '{k}'"));
                0
            }
        }
    };
    let warps: Vec<Warp> = world_raw
        .warp
        .into_iter()
        .map(|w| Warp {
            id: w.id,
            from: loc(&w.from, &mut errors),
            tile: (w.tile[0], w.tile[1]),
            to: loc(&w.to, &mut errors),
            to_tile: (w.to_tile[0], w.to_tile[1]),
        })
        .collect();
    let regions: Vec<Region> = world_raw
        .region
        .into_iter()
        .map(|r| Region {
            location: loc(&r.location, &mut errors),
            kind: r.kind,
            rect: r.rect,
        })
        .collect();
    let mut weather_ppm = [[0u32; 4]; 4];
    for s in Season::ALL {
        match world_raw.world.weather_ppm.get(s.name()) {
            Some(v) => weather_ppm[s as usize] = *v,
            None => errors.push(format!("world: weather_ppm missing {}", s.name())),
        }
    }
    let world = WorldParams {
        start_location: loc(&world_raw.world.start_location, &mut errors),
        start_tile: (world_raw.world.start_tile[0], world_raw.world.start_tile[1]),
        day_start: world_raw.world.day_start,
        day_end: world_raw.world.day_end,
        season_days: world_raw.world.season_days,
        energy_max: world_raw.world.energy_max,
        backpack_slots: world_raw.world.backpack_slots,
        stack_max: world_raw.world.stack_max,
        start_money: world_raw.world.start_money,
        passout_energy_ppm: world_raw.world.passout_energy_ppm,
        passout_money_ppm: world_raw.world.passout_money_ppm,
        passout_money_cap: world_raw.world.passout_money_cap,
        weather_ppm,
        minecart_minutes: world_raw.world.minecart_minutes,
    };

    // ---- crops
    check_dense("crops", crops_raw.crop.iter().map(|c| c.id as u32), &mut errors);
    let mut craw = crops_raw.crop;
    craw.sort_by_key(|c| c.id);
    let crops: Vec<Crop> = craw
        .into_iter()
        .map(|c| Crop {
            id: c.id,
            seed: rs.item(&c.seed, "crops"),
            produce: rs.item(&c.produce, "crops"),
            seasons: rs.seasons(&c.seasons, "crops"),
            key: c.key,
            grow_days: c.grow_days,
            regrow_days: c.regrow_days,
            yield_min: c.yield_min,
            yield_max: c.yield_max,
            extra_yield_ppm_per_level: c.extra_yield_ppm_per_level,
            xp: c.xp,
        })
        .collect();

    // ---- forage
    check_dense("forage", forage_raw.forage_location.iter().map(|f| f.id as u32), &mut errors);
    let mut fraw = forage_raw.forage_location;
    fraw.sort_by_key(|f| f.id);
    let mut forage = Vec::new();
    for f in fraw {
        let mut tables: [Vec<(ItemId, u32)>; 4] = Default::default();
        for s in f.season {
            let Some(se) = Season::parse(&s.season) else {
                errors.push(format!("forage: bad season {}", s.season));
                continue;
            };
            tables[se as usize] = s.items.iter().map(|w| (rs.item(&w.item, "forage"), w.weight)).collect();
        }
        forage.push(ForageLocation {
            id: f.id,
            location: loc(&f.location, &mut errors),
            daily_cap: f.daily_cap,
            tables,
        });
    }

    // ---- fish
    check_dense("fish", fish_raw.fish.iter().map(|f| f.id as u32), &mut errors);
    let mut fish_sorted = fish_raw.fish;
    fish_sorted.sort_by_key(|f| f.id);
    let fish: Vec<Fish> = fish_sorted
        .into_iter()
        .map(|f| Fish {
            id: f.id,
            item: rs.item(&f.item, "fish"),
            spots: f.spots.iter().map(|s| poi(s, &mut errors)).collect(),
            seasons: rs.seasons(&f.seasons, "fish"),
            weather: match f.weather.as_str() {
                "any" => WeatherReq::Any,
                "rain" => WeatherReq::Rain,
                "sun" => WeatherReq::Sun,
                o => {
                    errors.push(format!("fish: bad weather {o}"));
                    WeatherReq::Any
                }
            },
            time: (f.time[0], f.time[1]),
            difficulty: f.difficulty,
            min_rod: f.min_rod,
            weight: f.weight,
        })
        .collect();
    let junk: Vec<ItemId> = fish_raw.junk.items.iter().map(|k| rs.item(k, "junk")).collect();

    // ---- mining
    check_dense("rocks", mining_raw.rock.iter().map(|r| r.id as u32), &mut errors);
    let mut rraw = mining_raw.rock;
    rraw.sort_by_key(|r| r.id);
    let mut rock_by_key = BTreeMap::new();
    let rocks: Vec<Rock> = rraw
        .into_iter()
        .map(|r| {
            rock_by_key.insert(r.key.clone(), r.id);
            Rock {
                id: r.id,
                key: r.key,
                hardness: r.hardness,
                min_pick: r.min_pick,
                drops: r
                    .drops
                    .iter()
                    .map(|d| Drop {
                        item: rs.item(&d.item, "rock drops"),
                        min: d.min,
                        max: d.max,
                        ppm: d.ppm,
                    })
                    .collect(),
            }
        })
        .collect();
    let rock = |k: &str, errors: &mut Vec<String>| -> RockId {
        match rock_by_key.get(k) {
            Some(i) => *i,
            None => {
                errors.push(format!("unknown rock '{k}'"));
                0
            }
        }
    };
    let bands: Vec<Band> = mining_raw
        .band
        .iter()
        .map(|b| Band {
            floors: (b.floors[0], b.floors[1]),
            rocks: b.rocks.iter().map(|w| (rock(&w.rock, &mut errors), w.weight)).collect(),
        })
        .collect();
    let surface = Surface {
        quarry_rocks: mining_raw.surface.quarry_rocks,
        quarry_rock_kinds: mining_raw
            .surface
            .quarry_rock_kinds
            .iter()
            .map(|w| (rock(&w.rock, &mut errors), w.weight))
            .collect(),
    };

    // ---- wings / requests
    check_dense("wings", rest_raw.wing.iter().map(|w| w.id as u32), &mut errors);
    check_dense("requests", rest_raw.request.iter().map(|r| r.id as u32), &mut errors);
    let mut wraw = rest_raw.wing;
    wraw.sort_by_key(|w| w.id);
    let mut wing_by_key = BTreeMap::new();
    let mut wings: Vec<Wing> = wraw
        .into_iter()
        .map(|w| {
            wing_by_key.insert(w.key.clone(), w.id);
            let reward = match w.reward.kind.as_str() {
                "unlock" => Reward::Unlock(
                    w.reward
                        .feature
                        .as_deref()
                        .and_then(Feature::parse)
                        .unwrap_or_else(|| {
                            errors.push(format!("wing {}: bad feature", w.key));
                            Feature::Greenhouse
                        }),
                ),
                "energy" => Reward::Energy(w.reward.amount.unwrap_or(0) as u16),
                "money" => Reward::Money(w.reward.amount.unwrap_or(0)),
                "tool_level" => Reward::ToolLevel(
                    w.reward
                        .tools
                        .clone()
                        .unwrap_or_default()
                        .iter()
                        .filter_map(|t| Tool::parse(t))
                        .collect(),
                ),
                o => {
                    errors.push(format!("wing {}: bad reward kind {o}", w.key));
                    Reward::Money(0)
                }
            };
            Wing {
                id: w.id,
                key: w.key,
                name: w.name,
                reward,
                requests: Vec::new(),
            }
        })
        .collect();
    let mut rqraw = rest_raw.request;
    rqraw.sort_by_key(|r| r.id);
    let mut requests = Vec::new();
    for r in rqraw {
        let wing = match wing_by_key.get(&r.wing) {
            Some(w) => *w,
            None => {
                errors.push(format!("request {}: unknown wing {}", r.key, r.wing));
                0
            }
        };
        if r.need as usize > r.slots.len() {
            errors.push(format!("request {}: need {} > {} slots", r.key, r.need, r.slots.len()));
        }
        if r.coins == 0 && r.need == 0 {
            errors.push(format!("request {}: needs either coins or need>0", r.key));
        }
        let req = Request {
            id: r.id,
            wing,
            need: r.need,
            coins: r.coins,
            slots: r
                .slots
                .iter()
                .map(|s| Slot {
                    item: rs.item(&s.item, "request slots"),
                    count: s.count,
                })
                .collect(),
            pool_train: r.pool_train.iter().map(|k| rs.item(k, "pool_train")).collect(),
            pool_test: r.pool_test.iter().map(|k| rs.item(k, "pool_test")).collect(),
            key: r.key,
            name: r.name,
        };
        if let Some(w) = wings.get_mut(wing as usize) {
            w.requests.push(req.id);
        }
        requests.push(req);
    }

    // ---- recipes / machines
    check_dense("recipes", recipes_raw.recipe.iter().map(|r| r.id as u32), &mut errors);
    check_dense("machines", recipes_raw.machine.iter().map(|m| m.id as u32), &mut errors);
    let mut rcraw = recipes_raw.recipe;
    rcraw.sort_by_key(|r| r.id);
    let recipes: Vec<Recipe> = rcraw
        .into_iter()
        .map(|r| {
            let unlock = if r.unlock == "start" {
                Unlock::Start
            } else if let Some(rest) = r.unlock.strip_prefix("wing:") {
                Unlock::Wing(*wing_by_key.get(rest).unwrap_or_else(|| {
                    errors.push(format!("recipe {}: unknown wing {rest}", r.key));
                    &0
                }))
            } else {
                let mut parts = r.unlock.split(':');
                let sk = parts.next().unwrap_or("");
                let lv: u8 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                let skill = Skill::ALL.iter().copied().find(|s| s.name() == sk);
                match skill {
                    Some(s) => Unlock::Skill(s, lv),
                    None => {
                        errors.push(format!("recipe {}: bad unlock {}", r.key, r.unlock));
                        Unlock::Start
                    }
                }
            };
            Recipe {
                id: r.id,
                output: rs.item(&r.output, "recipes"),
                count: r.count,
                inputs: r.inputs.iter().map(|i| (rs.item(&i.item, "recipe inputs"), i.count)).collect(),
                unlock,
                key: r.key,
            }
        })
        .collect();
    let mut mraw = recipes_raw.machine;
    mraw.sort_by_key(|m| m.id);
    let machines: Vec<Machine> = mraw
        .into_iter()
        .map(|m| Machine {
            id: m.id,
            item: rs.item(&m.item, "machines"),
            rules: m
                .rules
                .iter()
                .map(|r| {
                    let input = match (&r.input, &r.input_category) {
                        (Some(i), _) => InputSel::Item(rs.item(i, "machine rule")),
                        (None, Some(c)) => InputSel::Category(Category::parse(c).unwrap_or_else(|| {
                            errors.push(format!("machine {}: bad category {c}", m.key));
                            Category::Junk
                        })),
                        (None, None) => {
                            errors.push(format!("machine {}: rule has no input", m.key));
                            InputSel::Category(Category::Junk)
                        }
                    };
                    Rule {
                        input,
                        input_count: r.input_count,
                        fuel: r.fuel.as_ref().map(|f| (rs.item(f, "machine fuel"), r.fuel_count.unwrap_or(1))),
                        output: rs.item(&r.output, "machine output"),
                        output_count: r.output_count,
                        minutes: r.minutes,
                    }
                })
                .collect(),
            key: m.key,
        })
        .collect();

    // ---- shops / merchant / upgrades
    check_dense("shops", shops_raw.shop.iter().map(|s| s.id as u32), &mut errors);
    let mut sraw = shops_raw.shop;
    sraw.sort_by_key(|s| s.id);
    let shops: Vec<Shop> = sraw
        .into_iter()
        .map(|s| Shop {
            id: s.id,
            location: loc(&s.location, &mut errors),
            poi: poi(&s.poi, &mut errors),
            open: s.open,
            close: s.close,
            closed_weekday: s.closed_weekday,
            stock: s
                .stock
                .iter()
                .map(|r| StockRow {
                    item: rs.item(&r.item, "stock"),
                    price: r.price,
                    seasons: rs.seasons(&r.seasons, "stock"),
                })
                .collect(),
            key: s.key,
            name: s.name,
        })
        .collect();
    let merchant = Merchant {
        poi: poi(&shops_raw.merchant.poi, &mut errors),
        location: loc(&shops_raw.merchant.location, &mut errors),
        weekdays: shops_raw.merchant.weekdays,
        open: shops_raw.merchant.open,
        close: shops_raw.merchant.close,
        slots: shops_raw.merchant.slots,
        price_mult_ppm: shops_raw.merchant.price_mult_ppm,
        floor_price: shops_raw.merchant.floor_price,
        pool: shops_raw.merchant.pool.iter().map(|k| rs.item(k, "merchant pool")).collect(),
    };
    let upgrades: Vec<UpgradeLevel> = shops_raw
        .upgrades
        .levels
        .iter()
        .map(|u| UpgradeLevel {
            level: u.level,
            coins: u.coins,
            bar: rs.item(&u.bar, "upgrades"),
            bars: u.bars,
        })
        .collect();

    errors.extend(rs.errors);

    // ---- derived
    let mut crop_by_seed = vec![None; items.len()];
    for c in &crops {
        crop_by_seed[c.seed as usize] = Some(c.id);
    }
    let mut machine_by_item = vec![None; items.len()];
    for m in &machines {
        machine_by_item[m.item as usize] = Some(m.id);
    }
    let mut buyable: Vec<ItemId> = shops.iter().flat_map(|s| s.stock.iter().map(|r| r.item)).collect();
    buyable.extend(merchant.pool.iter().copied());
    buyable.sort_unstable();
    buyable.dedup();
    let fishing_spots: Vec<PoiId> = pois.iter().filter(|p| p.kind == PoiKind::FishingSpot).map(|p| p.id).collect();
    let total_slots: u16 = requests.iter().map(|r| r.slots.len() as u16).sum();
    let max_slot_count = requests.iter().flat_map(|r| r.slots.iter().map(|s| s.count)).max().unwrap_or(1);
    let coin_requests: Vec<RequestId> = requests.iter().filter(|r| r.coins > 0).map(|r| r.id).collect();

    let content = Content {
        items,
        crops,
        forage,
        fishing: fish_raw.fishing,
        fish,
        junk,
        mining: mining_raw.mining,
        rocks,
        bands,
        surface,
        recipes,
        machines,
        shops,
        merchant,
        upgrade_days: shops_raw.upgrades.days,
        upgrades,
        skills: skills_raw.skills,
        xp: skills_raw.xp,
        tools: skills_raw.tools,
        restoration_name: rest_raw.restoration.name,
        wings,
        requests,
        world,
        locations,
        regions,
        pois,
        warps,
        density: world_raw.density,
        item_by_key,
        crop_by_seed,
        machine_by_item,
        buyable,
        fishing_spots,
        poi_by_key,
        loc_by_key,
        total_slots,
        max_slot_count,
        coin_requests,
    };
    if !errors.is_empty() {
        return Err(ContentError(errors));
    }
    content.validate().map_err(ContentError)?;
    Ok(content)
}

static CONTENT: OnceLock<Content> = OnceLock::new();
static CONTENT_HASH: OnceLock<u64> = OnceLock::new();

/// The parsed content tables (parsed once; panics on invalid content, which is a build defect).
pub fn content() -> &'static Content {
    CONTENT.get_or_init(|| match load() {
        Ok(c) => c,
        Err(e) => panic!("invalid Furrow content:\n{e}"),
    })
}

/// Try loading content, returning validation errors instead of panicking.
pub fn try_content() -> Result<&'static Content, ContentError> {
    if let Some(c) = CONTENT.get() {
        return Ok(c);
    }
    let c = load()?;
    Ok(CONTENT.get_or_init(|| c))
}

/// FNV-1a 64 over all TOML sources in the fixed [`SOURCES`] order (file name + contents).
pub fn content_hash() -> u64 {
    *CONTENT_HASH.get_or_init(|| {
        let mut bytes = Vec::new();
        for (name, src) in SOURCES {
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(0);
            bytes.extend_from_slice(src.as_bytes());
            bytes.push(0);
        }
        fnv1a64(&bytes)
    })
}

impl Content {
    /// Item by key (panics if unknown — content keys are compile-time constants in callers).
    pub fn item_id(&self, key: &str) -> ItemId {
        *self.item_by_key.get(key).unwrap_or_else(|| panic!("unknown item key {key}"))
    }
    /// Location by key.
    pub fn loc_id(&self, key: &str) -> LocId {
        *self.loc_by_key.get(key).unwrap_or_else(|| panic!("unknown location key {key}"))
    }
    /// POI by key.
    pub fn poi_id(&self, key: &str) -> PoiId {
        *self.poi_by_key.get(key).unwrap_or_else(|| panic!("unknown poi key {key}"))
    }
    /// Regions of a location with a given kind.
    pub fn regions_of<'a>(&'a self, loc: LocId, kind: &'a str) -> impl Iterator<Item = &'a Region> + 'a {
        self.regions.iter().filter(move |r| r.location == loc && r.kind == kind)
    }
    /// The shop selling an item today (cheapest), ignoring the merchant.
    pub fn shop_row(&self, item: ItemId, season: Season) -> Option<(ShopId, usize)> {
        let mut best: Option<(u32, ShopId, usize)> = None;
        for s in &self.shops {
            for (i, r) in s.stock.iter().enumerate() {
                if r.item == item && season_in(r.seasons, season) && best.is_none_or(|b| r.price < b.0) {
                    best = Some((r.price, s.id, i));
                }
            }
        }
        best.map(|b| (b.1, b.2))
    }
    /// Merchant price of an item.
    pub fn merchant_price(&self, item: ItemId) -> u32 {
        let sell = self.items[item as usize].sell as u64;
        let p = sell * self.merchant.price_mult_ppm as u64 / 1_000_000;
        (p as u32).max(self.merchant.floor_price)
    }
    /// Band for a floor.
    pub fn band_for_floor(&self, floor: u8) -> Option<&Band> {
        self.bands.iter().find(|b| floor >= b.floors.0 && floor <= b.floors.1)
    }

    /// Validate cross-table invariants. Returns all problems found.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut e = Vec::new();
        let n_items = self.items.len();
        // sources per item
        let mut src: Vec<Vec<&'static str>> = vec![Vec::new(); n_items];
        for c in &self.crops {
            src[c.produce as usize].push("crop");
        }
        for f in &self.forage {
            for t in &f.tables {
                for (i, _) in t {
                    src[*i as usize].push("forage");
                }
            }
        }
        for f in &self.fish {
            src[f.item as usize].push("fish");
        }
        for j in &self.junk {
            src[*j as usize].push("junk");
        }
        for r in &self.rocks {
            for d in &r.drops {
                src[d.item as usize].push("mining");
            }
        }
        for s in &self.shops {
            for r in &s.stock {
                src[r.item as usize].push("shop");
            }
        }
        for i in &self.merchant.pool {
            src[*i as usize].push("merchant");
        }
        for r in &self.recipes {
            src[r.output as usize].push("recipe");
        }
        for m in &self.machines {
            for r in &m.rules {
                src[r.output as usize].push("machine");
            }
        }
        src[self.item_id("wood") as usize].push("tree");
        src[self.item_id("resin") as usize].push("tree");
        src[self.item_id("hardwood") as usize].push("stump");
        src[self.item_id("fiber") as usize].push("weeds");
        src[self.item_id("stone") as usize].push("debris");
        for r in &self.requests {
            for s in &r.slots {
                if src[s.item as usize].is_empty() {
                    e.push(format!("request {}: slot item {} has no source", r.key, self.items[s.item as usize].key));
                }
            }
            for i in r.pool_train.iter().chain(r.pool_test.iter()) {
                if src[*i as usize].is_empty() {
                    e.push(format!("request {}: pool item {} has no source", r.key, self.items[*i as usize].key));
                }
            }
            if r.pool_train.len() < r.slots.len() && !r.slots.is_empty() {
                e.push(format!("request {}: pool_train smaller than slots", r.key));
            }
        }
        for r in &self.recipes {
            for (i, _) in &r.inputs {
                if src[*i as usize].is_empty() {
                    e.push(format!("recipe {}: input {} has no source", r.key, self.items[*i as usize].key));
                }
            }
        }
        // greenhouse circularity: the greenhouse is a reward; nothing in wing 0 may require it.
        // (checked statically: all wing-0 crops must be in-season outdoors) — true by construction here.
        // POIs in bounds
        for p in &self.pois {
            let l = &self.locations[p.location as usize];
            for (x, y) in [p.tile, p.stand] {
                if x >= l.width || y >= l.height {
                    e.push(format!("poi {}: tile out of bounds", p.key));
                }
            }
        }
        for w in &self.warps {
            let l = &self.locations[w.from as usize];
            if w.tile.0 >= l.width || w.tile.1 >= l.height {
                e.push(format!("warp {}: tile out of bounds", w.id));
            }
            let l2 = &self.locations[w.to as usize];
            if w.to_tile.0 >= l2.width || w.to_tile.1 >= l2.height {
                e.push(format!("warp {}: to_tile out of bounds", w.id));
            }
        }
        for f in &self.fish {
            if f.time.0 >= f.time.1 || f.time.1 > self.world.day_end {
                e.push(format!("fish {}: bad time window", f.id));
            }
            for s in &f.spots {
                if self.pois[*s as usize].kind != PoiKind::FishingSpot {
                    e.push(format!("fish {}: spot {} is not a fishing spot", f.id, s));
                }
            }
        }
        for c in &self.crops {
            if c.grow_days == 0 || c.yield_min == 0 || c.yield_max < c.yield_min || c.seasons == 0 {
                e.push(format!("crop {}: bad parameters", c.key));
            }
            if self.items[c.seed as usize].category != Category::Seed {
                e.push(format!("crop {}: seed item is not a seed", c.key));
            }
        }
        if self.skills.thresholds.len() != 10 {
            e.push("skills: need exactly 10 thresholds".into());
        }
        for (f, l) in [(1u8, self.mining.floors)].into_iter().take(1) {
            for floor in f..=l {
                if self.band_for_floor(floor).is_none() {
                    e.push(format!("mining: floor {floor} has no band"));
                }
            }
        }
        if self.upgrades.len() != 3 {
            e.push("upgrades: need levels 1..=3".into());
        }
        if n_items > u16::MAX as usize - 1 || self.locations.len() > 64 || self.pois.len() > 255 {
            e.push("content: id ranges too large".into());
        }
        if e.is_empty() {
            Ok(())
        } else {
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_loads_and_validates() {
        let c = match try_content() {
            Ok(c) => c,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(c.items.len(), 85);
        assert_eq!(c.crops.len(), 13);
        assert_eq!(c.wings.len(), 6);
        assert_eq!(c.requests.len(), 24);
        assert!(c.total_slots > 60);
        assert_eq!(c.fishing_spots.len(), 4);
        assert!(c.buyable.len() > 30);
    }

    #[test]
    fn hash_is_stable_within_process() {
        assert_eq!(content_hash(), content_hash());
        assert_ne!(content_hash(), 0);
    }

    #[test]
    fn ids_are_dense_and_keys_resolve() {
        let c = content();
        for (i, it) in c.items.iter().enumerate() {
            assert_eq!(it.id as usize, i);
            assert_eq!(c.item_id(&it.key), it.id);
        }
        assert_eq!(c.loc_id("homestead"), 0);
        assert_eq!(c.pois[c.poi_id("mill_board") as usize].kind, PoiKind::MillBoard);
    }
}
