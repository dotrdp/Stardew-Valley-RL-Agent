from API import StardewModdingAPI

defaults = {
    "wall": "▐",
    "wood": "🪵",
    "rock": "🪨",
    "tree": "🌲",
    "npc": "🧑‍🦯",
    "grass": "🌿",
    "water": "💧",
    "path": "👣",
    "normal": " ,",
    "floor": "a",
}

class Tile():
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.properties = {}
        self.type = type
        self.defaults = defaults

    def __str__(self): #type: ignore
        if self.type == "building":
            return self.defaults["wall"]
        elif self.type == "tree":
            return self.defaults["tree"]
        elif self.type == "rock":
            return self.defaults["rock"]
        elif self.type == "npc":
            return self.defaults["npc"]
        elif self.type == "grass":
            return self.defaults["grass"]
        elif self.type == "water":
            return self.defaults["water"]
        elif self.type == "path":
            return self.defaults["path"]
        return self.defaults["normal"]

class grid():
    def __init__(self, api):
        self.api = api
        self.defaults = defaults

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
        output = ""
        map = self.map
        buildings, back = map["Layers"]["Buildings"]["Tiles"], map["Layers"]["Back"]["Tiles"]
        MaxIndex = len(back)
        current = 0
        for tiled in back:
            x, y = tiled["X"], tiled["Y"]
            type = "normal"
            if tiled["TileSheet"] == "walls_and_floors":
                type = "floor"
            for maybe in buildings:
                if maybe["X"] == x and maybe["Y"] == y:
                    type = "building"
            output += Tile(x, y, type).__str__()
                    
            current += 1
            if current % map["MapSize"]["Height"] == 0:
                output += "\n"
        return output


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
