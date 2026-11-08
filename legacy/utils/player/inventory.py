from ..logger import Logger

class Item():
    def __init__(self,environment, type, name, id, slot, log_level):
        self.type = type
        self.name = name
        self.id = id
        self.slot = slot
        self.logger = Logger(log_level)
        self.environment = environment
    def __str__(self):
        return f"{self.slot} {self.name} ({self.type} - id: {self.id})" 
    def __call__(self, use=False):
        self.logger.log(f"Selecting item {self.name} ({self.type}) in slot {self.slot}", "DEBUG")
        self.environment.game_instance.reflection(function="setproperty", args=["player", "CurrentToolIndex", self.slot])
        if use != False:
            self.logger.log(f"Using item {self.name} ({self.type})", "DEBUG")
            return self.environment.world_action(["c", 100])

class inventory():
    def __init__(self, collection_items, environment, loglevel):
        self.items = []
        self.slots = list(range(12))
        self.environment = environment
        self.logger = Logger(loglevel)
        self.lv = loglevel
        self.logger.log("INSTANTIATING INV, note that you can save Item instances since those act as pointers", "WARNING")
        for item in collection_items:
            slot = self.slots.pop(0)
            self.slots = self.slots[0:]
            if item == None:
                self.items.append(Item(self.environment, "Empty", "Empty", "Empty", slot, self.lv))
                continue
            name = item["Type"]
            type, id = item["ToString"].split(".")[1], item["ToString"].split(".")[-1]
            self.items.append(Item(self.environment, type, name, id, slot, self.lv))
        self.logger.log(("\n"+", ".join(str(item) for item in self.items)), "INFO")
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

