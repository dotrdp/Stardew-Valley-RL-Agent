from API import StardewModdingAPI, read_msgpack_base64
from ENV import environment
from logger import Logger

METHOD = "ssh+tty"
class key:
    w = "w"
    a = "a"
    s = "s"
    d = "d"
    c = "c"
    x = "x"
    e = "e"

class Item():
    def __init__(self,environment, type, name, id, slot):
        self.type = type
        self.name = name
        self.id = id
        self.slot = slot
        self.environment = environment
    def __str__(self):
        return f"{self.slot} {self.name} ({self.type} - id: {self.id})" 
    def __call__(self, use=False):
        self.environment.logger.log(f"Selecting item {self.name} ({self.type}) in slot {self.slot}", "DEBUG")
        self.environment.game_instance.reflection(function="setproperty", args=["player", "CurrentToolIndex", self.slot])
        if use != False:
            self.environment.logger.log(f"Using item {self.name} ({self.type})", "DEBUG")
            return self.environment.world_action(["c", 100])

class inventory():
    def __init__(self, collection_items, environment: environment):
        self.items = []
        self.slots = list(range(12))
        self.environment = environment
        self.environment.logger.log("INSTANTIATING INV, refer to inv.selected instead\nunless strictly changing tool", "WARNING")
        for item in collection_items:
            slot = self.slots.pop(0)
            self.slots = self.slots[0:]
            if item == None:
                self.items.append(Item(self.environment, "Empty", "Empty", "Empty", slot))
                continue
            name = item["Type"]
            type, id = item["ToString"].split(".")[1], item["ToString"].split(".")[-1]
            self.items.append(Item(self.environment, type, name, id, slot))
        self.environment.logger.log(("\n"+", ".join(str(item) for item in self.items)), "INFO")
        if self.items[-1] != Item(self.environment, "Empty", "Empty", "Empty", self.items[-1].slot):
            self.environment.logger.log("quick inventory is full", "WARNING")
    def __str__(self):
        return ", ".join(str(item) for item in self.items)

    @property
    def selected(self):
        res = self.environment.game_instance.reflection(function="getproperty", args=["player", "CurrentToolIndex"])
        if res["Success"]:
            return self.items[res["Result"]]
        else:
            self.items[0]()
            return self.items[0]

class player():
    def __init__(self, environment: environment, loglevel: str = "CRITICAL"):
        self.environment = environment
        self.r = self.environment.game_instance.__getattribute__("reflection") # USEFUL API FOR READING GAME DATA
        self.read_msgpack_base64 = read_msgpack_base64
        self.MaxItems = 12
        self.logger = Logger(loglevel)
        self.logger.log("Player instance created", "INFO")
        

    def wrap_result(self, result: dict):
        if result["Success"]:
            return self.read_msgpack_base64(result["Base64_binary"])
        else:
            self.logger.log(f"Error: {result['Error']}", "ERROR")
            self.logger.log("This is a critical error, the game may not work properly", "CRITICAL")
            raise Exception(f"Error: {result['Error']}")
    def gp(self, args):
        return self.wrap_result(self.r(function="getproperty", args=args))
    def gf(self, args):
        return self.wrap_result(self.r(function="getfield", args=args))

    @property
    def stamina(self):
        self.logger.log("Requesting player stamina", "DEBUG")
        return self.gp(["player", "stamina"])

    @property
    def health(self):
        self.logger.log("Requesting player health", "DEBUG")
        return self.wrap_result(self.r(function="getfield", args=["player", "health"]))

    @property
    def money(self):
        self.logger.log("Requesting player money", "DEBUG")
        return self.gp(["player", "Money"])

    @property
    def location(self):
        self.logger.log("Requesting player location", "DEBUG")
        return self.wrap_result(self.r(function="getproperty", args=["player", "currentLocation"]))["NameOrUniqueName"]

    @property
    def position(self):
        res = self.wrap_result(self.r(function="getproperty", args=["player", "Tile"]))
        self.logger.log(f"Player position requested: {res['_Field_X']}, {res['_Field_Y']}", "DEBUG")
        self.logger.log("player.position res" + str(res), "DEBUG")
        return (res["_Field_X"], res["_Field_Y"])
    
    @property
    def inventory(self):
        collection_items = self.wrap_result(self.r(function="getproperty", args=["player", "Items"]))["_Collection_Items"]
        return inventory(collection_items, self.environment)

    def normal_action(self, key, durationms):
        return self.environment.world_action([key, durationms])

    def close_dialogue(self):
        self.logger.log("Closing dialogue", "INFO")
        self.logger.log("Invoking exitActiveMenu\nIt requires some special movement sometimes", "WARNING")
        return self.environment.game_instance.reflection(function="invokemethod", args=["game1", "exitActiveMenu"])

