from API import read_msgpack_base64
from ENV import environment
import networkx as nx
import time

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
    def __init__(self, environment: environment):
        self.environment = environment
        self.r = self.environment.game_instance.__getattribute__("reflection") # USEFUL API FOR READING GAME DATA
        self.read_msgpack_base64 = read_msgpack_base64
        self.MaxItems = 12
        self.logger = self.environment.logger
        self.path = None
        self.logger.log("Player instance created", "INFO")
        self.antithesis = {
            "a": "d",
            "d": "a",
            "w": "s",
            "s": "w"
        }

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
    def speed(self):
        self.logger.log("Requesting player speed", "DEBUG")
        return self.wrap_result(self.r(function="getproperty", args=["player", "speed"]))

    def walk1(self, speed):
        return int(1000 / speed)
        

    @property
    def money(self):
        self.logger.log("Requesting player money", "DEBUG")
        return self.gp(["player", "Money"])

    @property
    def location(self):
        self.logger.log("Requesting player location", "DEBUG")
        return self.wrap_result(self.r(function="getproperty", args=["player", "currentLocation"]))["NameOrUniqueName"]

    @property
    def position(self) -> tuple:
        self.logger.log("Requesting player position", "DEBUG")
        res = self.wrap_result(self.r(function="getproperty", args=["player", "Tile"]))
        self.logger.log(f"Player position: {res['_Field_X']}, {res['_Field_Y']}", "INFO")
        return (res["_Field_X"], res["_Field_Y"])
    
    @property
    def inventory(self) -> inventory:
        collection_items = self.wrap_result(self.r(function="getproperty", args=["player", "Items"]))["_Collection_Items"]
        return inventory(collection_items, self.environment)

    def normal_action(self, key, durationms):
        return self.environment.world_action([key, durationms])

    def walk_to(self, x, y):
        self.logger.log(f"Walking to position ({x}, {y})", "INFO")
        xi, yi = self.position
        xi, yi = int(xi), int(yi)
        collision_graph = self.environment.get_collision_graph()
        self.logger.log("getting shortest path", "DEBUG")
        path = nx.shortest_path(collision_graph, source=(xi, yi), target=(x, y))
        self.logger.log(f"Shortest path found: {path}", "INFO")
        path = self.optimize_path(path)
        self.logger.log(f"Optimized path: {path}", "INFO")
        self.logger.log("Following path", "DEBUG")
        self.path = path
        expected = None
        for point in path:
            self.logger.log(f"Walking to point {point}", "DEBUG")
            player = self.environment.game_instance.reflection(function="getproperty", args=["game1", "player"])["Result"]["Properties"]
            xi, yi = (str(player["Tile"]).replace("Vector2: ", "").replace("}", "").replace("{X:","").replace("Y","").replace(":", "").split(" "))  # "Vector2: {X: 1 Y: 1]"
            xi, yi = int(xi), int(yi)
            self.logger.log(f"Current position: ({xi}, {yi})", "DEBUG")
            if expected is not None and expected != (xi, yi):
                self.walk_to(x, y)
                break
            speed = player["speed"]
            walk1 = self.walk1(speed)
            xp, yp = point
            if (xp, yp) == (xi, yi):
                continue
            distance = ((xp - xi) ** 2 + (yp - yi) ** 2) ** 0.5
            duration = int(distance * walk1)
            key = None
            if xp > xi:
                key = "d"
            elif xp < xi:
                key = "a"
            elif yp > yi:
                key = "s"
            elif yp < yi:
                key = "w"
            self.normal_action(key, duration)
            time.sleep(duration / 1000)  # Convert milliseconds to seconds
            expected = (xp, yp)
        xi, yi = self.position
        if (xi, yi) != (x, y):
            self.walk_to(x, y)
        self.path = None
            
            

    def optimize_path(self, path: list):
        self.logger.log("Optimizing path", "DEBUG")
        if not path:
            self.logger.log("Path is empty, returning empty path", "CRITICAL")
            return []
        optimized_path = []
        next_point = None
        previous_slope = (path[0][1] - path[1][1]) / (path[0][0] - path[1][0]) if path[0][0] != path[1][0] else float('inf')
        for point in path:
            next_point = path[path.index(point) + 1] if path.index(point) + 1 < len(path) else point
            slope = (point[1] - next_point[1]) / (point[0] - next_point[0]) if point[0] != next_point[0] else float('inf')
            if slope != previous_slope:
                optimized_path.append(point)
                previous_slope = slope
        if path[-1] not in optimized_path:
            optimized_path.append(path[-1])
        return optimized_path



    def close_dialogue(self):
        self.logger.log("Closing dialogue", "INFO")
        self.logger.log("Invoking exitActiveMenu\nIt requires some special movement sometimes", "WARNING")
        return self.environment.game_instance.reflection(function="invokemethod", args=["game1", "exitActiveMenu"])
