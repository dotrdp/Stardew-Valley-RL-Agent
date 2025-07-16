from API import StardewModdingAPI, read_msgpack_base64
from colorama import Fore

defaults = {
    "building":  "󰆦",
    "normal":    "·",
    "player":    "",
    "Chest":     "󰜦",
}
defaults["building"] = Fore.MAGENTA + defaults["building"] + Fore.RESET
defaults["Chest"] = Fore.BLUE + defaults["Chest"] + Fore.RESET
defaults["player"] = Fore.GREEN + defaults["player"] + Fore.RESET


class Tile():
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.properties = {}
        self.type = type
        self.defaults = defaults

    def __str__(self): #type: ignore
        if self.type in self.defaults:
            return self.defaults[self.type]
        print(f"Unknown tile type: {self.type}")
        return "?"


class grid():
    def __init__(self, api):
        self.api = api
        self.defaults = defaults
        self.readmsgpack = read_msgpack_base64

    def get_data(self):
        map = self.map
        buildings, back = map["Layers"]["Buildings"]["Tiles"], map["Layers"]["Back"]["Tiles"]
        objects = map["Objects"]
        number_of_x = back[-1]["X"]
        number_of_y = back[-1]["Y"]
        result = [[Tile(x, y, "normal") for y in range(number_of_y+1)] for x in range(number_of_x+1)]
        player = self.readmsgpack(self.api.reflection(function="getproperty", args=["player", "Tile"])["Base64_binary"])
        player_x, player_y = int(player["_Field_X"]), int(player["_Field_Y"])
        result[player_x][player_y] = Tile(player_x, player_y, "player")
        for tile in buildings:
            x, y = tile["X"], tile["Y"]
            result[x][y] = Tile(x, y, "building")
        for object in objects:
            x,y = object["Position"]["X"], object["Position"]["Y"]
            result[x][y] = Tile(x, y, object["Type"])
        return result
        

    @property
    def map(self):
        return self.api.map()["MapData"]

    @property
    def size(self):
        return {"x": self.map["MapSize"]["Width"], 
                "y": self.map["MapSize"]["Height"]}
   
    def tile(self, pos, type="normal"):
        self.map
    
    def __str__(self):
        data = self.get_data()
        result = ""
        for row in data:
            for tile in row:
                result += str(tile)
            result += "\n"
        return result


class Map():
    def __init__(self, api):
        self.api = api
    
    @property
    def grid(self):
        return grid(self.api)
    
    @property
    def size(self):
        return self.map_data["MapSize"]

    @property
    def map_data(self):
        return self.api.map()["MapData"]

api = StardewModdingAPI(method="ssh+tty")
a = Map(api)
print(a.grid)
