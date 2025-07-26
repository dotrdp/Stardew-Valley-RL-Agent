from API import read_msgpack_base64
from colorama import Fore


defaults = {
    "Building":  "󰆦",
    "normal":    "·",
    "player":    "",
    "Chest":     "󰜦",
    "Stone":     "",
    "Weeds":     "󰹧",
    "Twig":      "󱐚",
    "Tree":      "",
    "door":      "󰠚",
    "NpcBarrier":"▫",
    "Bed":       "",
    "something": "▫",
    "PetBowl":   "󰊎",
    "shipbin":   "",
    "Seed Spot": "",
    "NoS":       "✕",
    "Grass":     "󱔐",
    "bar":       "▫",
    "Buildable": "󰦻",
    "NF":        "󰆷",
    "NFNS":      "✕",
    "notbuild":  "✕",
    "Stone Owl": "󰏒",
"Artifact Spot": "",
"debugmarker":   "",
    "cbuilding": "",
    "mail":      "󰛮",
    "f":         "·",
  "NFF":         "·",
"debug_red":     "✕",
    "Water":     "",
}

def get_logic(type):
   lgc = {
        "cbuilding": {"collision": True, "blocks_crops": True},
        "Building": {"collision": True, "blocks_crops": True},
        "Water": {"collision": True, "blocks_crops": True},
        "Stone": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Pickaxe"},
        "Weeds": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Scythe"},
        "Tree": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Axe", "health": 10},
        "Twig": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Axe"},
        "Seed Spot": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Hoe"},
        "Chest": {"collision": True},
        "door": {"collision": True, "collisionwarp": True},
        "shipbin": {"collision": True, "sellplace": True, "blocks_crops": True},
        "PetBowl": {"collision": True, "blocks_crops": True},
        "notbuild": {"collision": True, "blocks_crops": True},
        "NFNS": {"collision": True, "blocks_crops": True},
        "Stone Owl": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Pickaxe"},
        "Grass": {"blocks_crops": True, "breakable": True, "tool": "Scythe"},
        "Artifact Spot": {"collision": True, "blocks_crops": True, "breakable": True, "tool": "Hoe"},
        "mail": {"collision": True, "blocks_crops": True},
        }
   if type in lgc:
       return lgc[type]
   else:
        return {}
    

defaults["Building"] = Fore.MAGENTA + defaults["Building"] + Fore.RESET
defaults["cbuilding"] = Fore.LIGHTCYAN_EX + defaults["cbuilding"] + Fore.RESET
defaults["Chest"] = Fore.BLUE + defaults["Chest"] + Fore.RESET
defaults["player"] = Fore.RED + defaults["player"] + Fore.RESET
defaults["normal"] = Fore.LIGHTBLACK_EX + defaults["normal"] + Fore.RESET
defaults["something"] = Fore.LIGHTBLACK_EX + defaults["something"] + Fore.RESET
defaults["Weeds"] = Fore.GREEN + defaults["Weeds"] + Fore.RESET
defaults["Tree"] = Fore.GREEN + defaults["Tree"] + Fore.RESET
defaults["Twig"] = Fore.LIGHTGREEN_EX + defaults["Twig"] + Fore.RESET
defaults["Seed Spot"] = Fore.RED + defaults["Seed Spot"] + Fore.RESET
defaults["Grass"] = Fore.LIGHTBLACK_EX + defaults["Grass"] + Fore.RESET
defaults["NoS"] = Fore.LIGHTBLACK_EX + defaults["NoS"] + Fore.RESET
defaults["NpcBarrier"] = Fore.RED + defaults["NpcBarrier"] + Fore.RESET
defaults["bar"] = Fore.RED + defaults["bar"] + Fore.RESET
defaults["notbuild"] = Fore.BLACK + defaults["notbuild"] + Fore.RESET
defaults["Stone Owl"] = Fore.RED + defaults["Stone Owl"] + Fore.RESET
defaults["Artifact Spot"] = Fore.RED + defaults["Artifact Spot"] + Fore.RESET
defaults["debugmarker"] = Fore.RED + defaults["debugmarker"] + Fore.RESET
defaults["debug_red"] = Fore.RED + defaults["debug_red"] + Fore.RESET
defaults["Water"] = Fore.BLUE + defaults["Water"] + Fore.RESET



known_properties = {
    "NpcBarrier": ["NPCBarrier", "NoSpawn"],
    "bar": ["NPCBarrier"],
    "Building": ["WallID"],
    "Bed": ["DefaultBedPosition", "FloorID"],
    "NoS": ["NoSpawn"],
    "Buildable": ["Buildable"],
    "NF": ["NoFurniture"],
    "NFNS": ["NoFurniture", "NoSpawn"],
    "f": ["FloorID"],
    "NFF": ["NoFurniture", "FloorID"],
}

class building():
    def __init__(self, x, y):
        self.origin = (x, y)
        self.build = {}

    def fill(self, x1, y1, x2, y2, type):
        x1 += self.origin[0]
        y1 += self.origin[1]
        x2 += self.origin[0]
        y2 += self.origin[1]
        for x in range(x1, x2):
            for y in range(y1, y2):
                self.build[str(x)+","+str(y)] = Tile(x, y, type)
    def set(self, x, y, type):
        x += self.origin[0]
        y += self.origin[1]
        self.build[str(x)+","+str(y)] = Tile(x, y, type)

    def inject(self, grid):
        g = grid
        for tile in self.build:
            x, y = map(int, tile.split(","))
            g[x][y] = self.build[tile]
        return g 

class Farmhouse():
    def __init__(self, x, y):
        self.building = building(x, y)
        self.building.fill(0, 0, 9, 3, "cbuilding") # main body
        self.building.set(8, 3, "cbuilding") # porch
        self.building.fill(0, 4, 9, 5, "cbuilding") # stairs
        self.building.fill(0, 3, 1, 4, "cbuilding") # porch
        self.building.fill(4, 4, 7, 5, "normal") # stairs
        self.building.set(5, 2, "door") # door
        self.building.set(9, 4, "mail") # mail box
        self.build = self.building.build

class Greenhouse():
    def __init__(self, x, y):
        self.building = building(x, y)
        self.building.set(0, 0, "cbuilding")
        self.build = self.building.build

class shipbin():
    def __init__(self, x, y):
        self.building = building(x, y)
        self.building.fill(-1, 0, 1, 1, "shipbin")
        self.build = self.building.build

class PetBowl():
    def __init__(self, x, y):
        self.building = building(x, y)
        self.building.set(0, 0, "PetBowl")
        self.build = self.building.build

buildings = {
    "Farmhouse": Farmhouse,
    "Greenhouse": Greenhouse,
    "Shipping Bin": shipbin,  
    "Pet Bowl": PetBowl,

}
        
# 󰆚 cow

class Tile():
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type
        self.tilesheet = None
        self.defaults = defaults
        self.properties = get_logic(type)

    def __str__(self): 
        if self.type in self.defaults:
            return self.defaults[self.type]
        print(f"Unknown tile type: {self.type}")
        return "?"


class Map():
    def __init__(self, api):
        self.api = api
        self.defaults = defaults
        self.readmsgpack = read_msgpack_base64
        self.known_properties = known_properties
        self.buildings = buildings

    def get_building(self, buildname):
        if buildname in self.buildings:
            return self.buildings[buildname]
        raise ValueError(f"Unknown building type: {buildname}")
        

    def properties_logic(self, keys):
        for property in self.known_properties:
            if self.known_properties[property] == list(keys):
                return property
        self.api.logger.log(f"Unknown property: {keys}", "WARNING")
        return "something"

    def get_data(self):
        map = self.map
        buildings, back = map["Layers"]["Buildings"]["Tiles"], map["Layers"]["Back"]["Tiles"]
        other_buildings = map["Buildings"]
        objects = map["Objects"]
        tileprops: dict = map["TileProperties"]
        number_of_x = back[-1]["X"]
        number_of_y = back[-1]["Y"]
        result = [[Tile(x, y, "normal") for y in range(number_of_y+1)] for x in range(number_of_x+1)]
        player = self.readmsgpack(self.api.reflection(function="getproperty", args=["player", "Tile"])["Base64_binary"])
        tilesheets = {}

        for chunk in map["TileSheets"]:
           for data in chunk["Properties"]:
               if data.startswith("@TileIndex@"):
                   split = data.replace("@TileIndex@", "")
                   id, label = split.split("@")
                   value = chunk["Properties"][data]
                   id = str(id)
                   if id not in tilesheets:
                     tilesheets[id] = {} 
                   tilesheets[id][label] = value
              
        for prop in tileprops["Back"]:
            x, y = prop.split(",")
            value = tileprops["Back"][prop]
            prop_type = "normal"
            if "Buildable" in value:
                if value["Buildable"] == "f":
                    prop_type = "notbuild"
            else:
                prop_type = self.properties_logic(value.keys())
            result[int(x)][int(y)] = Tile(int(x), int(y), prop_type)

        for build in other_buildings:
            x, y = build["Position"].split(",")
            x, y = int(x), int(y)
            building_target = build["BuildingType"]
            building = self.get_building(building_target)
            building = building(x, y)
            result = building.building.inject(result) # type: ignore
            build_type = build["Type"]
            self.api.logger.log(f"Building {building_target} at {x},{y} ({build_type})", "DEBUG")
            if build_type == "ShippingBin":
                build_type = "shipbin"

        for tile in buildings:
            x, y = tile["X"], tile["Y"]
            result[x][y] = Tile(x, y, "Building")
 
        #for warp in warps:
        #    x, y = warp["TargetX"], warp["TargetY"]
        #    result[x][y] = Tile(x, y, "Warp")
        for object in objects:
            x,y = object["Position"]["X"], object["Position"]["Y"]
            object_type = object["Name"]
            if object["MinutesUntilReady"] == 1 and object_type == "Weeds":
                object_type = "Tree"
            result[x][y] = Tile(x, y, object_type)

        if "TerrainFeatures" in map:
            for object in map["TerrainFeatures"]:
                x, y = object["Position"]["X"], object["Position"]["Y"]
                object_type = object["Type"]
                result[x][y] = Tile(x, y, object_type)

        for label in list(map["Layers"].keys()):
            layer = map["Layers"][label]["Tiles"]
            for tile in layer:
                # if tile["TileSheet"] != "untitled tile sheet":
                #     if tile["TileSheet"] != "Paths":
                #         print(f"TileSheet: {tile['TileSheet']}")
                if tile["Properties"] != {}:
                    if "Order" in tile["Properties"]:
                        x, y = tile["X"], tile["Y"]
                        if result[x][y].type == "normal":
                            result[x][y].type = "something"
                        
                id = tile["TileIndex"]
                x, y = tile["X"], tile["Y"]
                id = str(id)
                if id in list(tilesheets.keys()):
                    result[x][y].tilesheet = tilesheets[id]
                    if "Water" in tilesheets[id]:
                        result[x][y].type = "Water"
                else:
                    pass # There are a bunch of unknown tilesheets which I don't know what could they be. They're just everywhere.
                        
        player_x, player_y = int(player["_Field_X"]), int(player["_Field_Y"])
        result[player_x][player_y] = Tile(player_x, player_y, "player")


        self.api.logger.log("Generated map data from API", "DEBUG")
        self.api.logger.log(f"Grid size: {len(result)}x{len(result[0])} ({str(len(result)*len(result[0]))} tiles)", "INFO")
                
        return result
    
        

    @property
    def map(self):
        return self.api.map()["MapData"]

    def __str__(self):
        data = self.get_data()
        transposed_data = list(zip(*data))  
        # transposed_data = data
        result = ""
        for row in transposed_data:
            for tile in row:
                result += str(tile)
            result += "\n"
        return result


