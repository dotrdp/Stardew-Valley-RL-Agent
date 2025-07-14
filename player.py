from API import StardewModdingAPI, read_msgpack_base64
from ENV import environment

METHOD = "ssh+tty"

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
        self.environment.game_instance.reflection(function="setproperty", args=["player", "CurrentToolIndex", self.slot])
        if use != False:
            return self.environment.world_action(["c", 100])

class inventory():
    def __init__(self, collection_items, environment):
        self.items = []
        self.slots = list(range(12))
        self.environment = environment
        for item in collection_items:
            slot = self.slots.pop(0)
            self.slots = self.slots[0:]
            if item == None:
                self.items.append(Item(self.environment, "Empty", "Empty", "Empty", slot))
                continue
            name = item["Type"]
            type, id = item["ToString"].split(".")[1], item["ToString"].split(".")[-1]
            self.items.append(Item(self.environment, type, name, id, slot))
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
    def __init__(self, environment):
        self.environment = environment
        self.r = self.environment.game_instance.__getattribute__("reflection") # USEFUL API FOR READING GAME DATA
        self.read_msgpack_base64 = read_msgpack_base64
        self.MaxItems = 12
        

    def wrap_result(self, result: dict):
        if result["Success"]:
            return self.read_msgpack_base64(result["Base64_binary"])
        else:
            print(result)
            raise Exception(f"Error: {result['Error']}")
    def gp(self, args):
        return self.wrap_result(self.r(function="getproperty", args=args))
    def gf(self, args):
        return self.wrap_result(self.r(function="getfield", args=args))

    @property
    def stamina(self):
        return self.gp(["player", "stamina"])

    @property
    def health(self):
        return self.wrap_result(self.r(function="getfield", args=["player", "health"]))

    @property
    def money(self):
        return self.gp(["player", "Money"])

    @property
    def location(self):
        return self.wrap_result(self.r(function="getproperty", args=["player", "currentLocation"]))["NameOrUniqueName"]

    @property
    def position(self):
        return self.gp(["player", "Position"])
    
    @property
    def inventory(self):
        collection_items = self.wrap_result(self.r(function="getproperty", args=["player", "Items"]))["_Collection_Items"]
        return inventory(collection_items, self.environment)

game_environment = environment(StardewModdingAPI(method=METHOD))
player_agent = player(game_environment)
