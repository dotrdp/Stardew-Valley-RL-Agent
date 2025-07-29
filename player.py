from API import read_msgpack_base64
from ENV import environment
import networkx as nx
import time

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
        self.environment.logger.log("INSTANTIATING INV, note that you can save Item instances since those act as pointers", "WARNING")
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
        self.expected = None
        self.nconvs = 0
        self.failed_convs = 0
        self.cutscenes_quickfix()
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

    @property
    def can_move(self):
        self.logger.log("Checking if player can move", "DEBUG")
        envstate = self.environment.envstate # returns a dict with environment properties
        res = self.environment.world_env.isAvailable(envstate) # returns a list [bool, missing_properties_if_any]  
        if res[0]:
            self.logger.log("Player can move", "DEBUG")
            return True
        else:
            self.logger.log(f"Player cannot move, missing properties: {res[1]}", "WARNING")
            return False

    def normal_action(self, key, durationms):
        return self.environment.world_action([key, durationms])

    def cutscenes_quickfix(self):
        self.environment.game_instance.skip_events(False)
        self.environment.game_instance.skip_events(True)
        self.environment.game_instance.reflection(function="invokemethod", args=["game1", "exitActiveMenu"])

    def walk_to(self, x, y):
        self.cutscenes_quickfix()
        self.logger.log(f"Walking to position ({x}, {y})", "INFO")
        xi, yi = self.position
        xi, yi = int(xi), int(yi)
        if (xi, yi) == (x, y):
            self.logger.log(f"Already at target position ({x}, {y})", "DEBUG")
            return
        collision_graph = self.environment.get_collision_graph()
        self.logger.log("getting shortest path", "DEBUG")
        if ((x-xi) ** 2 + (y-yi) ** 2) ** 0.5 == 1:
            self.logger.log("Target is close enough, walking directly", "DEBUG")
            path = [(x, y)]
        else:
            path = nx.shortest_path(collision_graph, source=(xi, yi), target=(x, y))
        self.logger.log(f"Shortest path found: {path}", "INFO")
        path = self.optimize_path(path)
        self.logger.log(f"Optimized path: {path}", "INFO")
        self.logger.log("Following path", "DEBUG")
        self.path = path
        for point in path:
            self.logger.log(f"Walking to point {point}", "DEBUG")
            player = self.r(function="getproperty", args=["game1", "player"])["Result"]["Properties"]
            xi, yi = (str(player["Tile"]).replace("Vector2: ", "").replace("}", "").replace("{X:","").replace("Y","").replace(":", "").split(" "))  # "Vector2: {X: 1 Y: 1]"
            xi, yi = int(xi), int(yi)
            self.logger.log(f"Current position: ({xi}, {yi})", "DEBUG")
            if self.expected != None and self.expected != (xi, yi):
                self.nconvs += 1
                self.logger.log(f"Position mismatch: self.expected {self.expected}, got ({xi}, {yi})", "WARNING")
                if self.nconvs > 5:
                    self.logger.log("Too many consecutive position mismatches, modifying spatial state", "DEBUG")
                    gp = self.environment.get_energy_graph()
                    pt = nx.dijkstra_path(gp, source=(xi, yi), target=(x, y))
                    pont = pt[1]
                    if len(pt) < 2:
                        pont = pt[0]
                    xd, yd = pont
                    self.environment.draw_learned_tile(xd, yd, "Building")
                    self.nconvs = 0
                    self.follow_energy_path(pt) # type: ignore
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
            action: list = self.normal_action(key, duration) # type: ignore
            if action[0] == False:
                self.failed_convs += 1
                if self.failed_convs > 4:
                    self.failed_convs = 0
                    self.nconvs = 0
                    self.logger.log("spatial state is not modifiable\n probably in a cutscene or stuck somewhere", "CRITICAL")
                    return

            self.expected = (xp, yp)
            self.logger.log(f"set expected to {self.expected}", "DEBUG")
            interval = duration/8
            cachedx, cachedy = None, None
            for _ in range(8):
                xi, yi = self.position
                cachedx, cachedy = xi, yi
                if (xi, yi) == self.expected:    
                    self.logger.log(f"Reached point {self.expected}", "DEBUG")
                    break
                time.sleep(interval / 1000)  # Convert milliseconds to seconds
                if (cachedx, cachedy) == (xi, yi):
                    break # player is not moving, therefore no point in waiting
        xi, yi = self.position
        if (xi, yi) != (x, y):
            self.walk_to(x, y)

        self.path = None
            

    def optimize_path(self, path: list):
        self.logger.log("Optimizing path", "DEBUG")
        if not path:
            self.logger.log("Path is empty, returning empty path", "CRITICAL")
            return []
        if len(path) <= 1:
            self.logger.log("Path has less than 2 points, returning original path", "DEBUG")
            return path
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
    def face_direction(self, key: str):
        dirs = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3
        }
        if key not in dirs:
            self.logger.log(f"Invalid direction: {key}", "ERROR")
        self.r(function="invokemethod", args=["player", "faceDirection", dirs[key]])

    def follow_energy_path(self, path: list):
        current_target = None
        tools = {
            "Pickaxe": None,
            "Axe": None,
            "Scythe": None

        }
        self.logger.log("Following energy path", "DEBUG")
        for item in self.inventory.items:
            if item.name in tools:
                tools[item.name] = item
            elif item.name == "MeleeWeapon":
                tools["Scythe"] = item
        for point in path:
            x, y = point
            point_properties = self.environment.spatial_state[x][y].properties
            if "tool" in point_properties:
                tool = point_properties["tool"]
                current_target = path[path.index(point) - 1]
                self.walk_to(current_target[0], current_target[1])
                key = "error"
                if x > current_target[0]:
                    key = "d"
                elif x < current_target[0]:
                    key = "a"
                elif y > current_target[1]:
                    key = "s"
                elif y < current_target[1]:
                    key = "w"
                self.face_direction(key)
                if tool in tools and tools[tool] is not None:
                    self.logger.log(f"Using tool {tools[tool].name} on point {point}", "DEBUG") # type: ignore
                    if "health" in point_properties:
                        health = point_properties["health"]
                        for i in range(health):
                            self.logger.log(f"Dealing damage to point {point} ({i+1}/{health})", "DEBUG")
                            tools[tool](use=True) # type: ignore
                    else:
                        tools[tool](use=True) # type: ignore
                    self.environment.update_spatial_state()
                    if self.logger.level == 0: # DEBUG level is 0, sorry for the magic numbers
                        self.environment.print_path(path[path.index(point):]) # this calls print, therefore it couldnt be blocked otherwise
                    time.sleep(0.1)
                else:
                    self.logger.log(f"No tool found for {tool}, skipping", "WARNING")

    def close_dialogue(self):
        self.logger.log("Closing dialogue", "INFO")
        self.logger.log("Invoking exitActiveMenu\nIt requires some special movement sometimes", "WARNING")
        return self.environment.game_instance.reflection(function="invokemethod", args=["game1", "exitActiveMenu"])
