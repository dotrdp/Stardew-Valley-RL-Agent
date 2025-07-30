import networkx as nx
import time
from retry import retry
import asyncio
from dotenv import dotenv_values
from logger import Logger

from ..api.API import read_msgpack_base64
from .inventory import inventory, Item

prefs = dotenv_values(".env") 
log_level_player = prefs.get("debug_level_player", "ERROR")
debug_level = prefs.get("debug_level", "ERROR")
if debug_level != "ERROR":
    log_level_player = debug_level

attempts = {
            "api_call": 4, # number of attempts to interact with the environment or game instance
            "walk_to": 3, # number of attempts to walk_to a position
            "follow_energy_path": 2, # number of attempts to follow an energy path
            "assume_wall": 5 # number of attempts to assume that the player is running into a wall
        } # reducing these yields faster actions, but may lead to errors

class player():
    def __init__(self, environment):

        self.environment = environment
        self.r = self.environment.game_instance.__getattribute__("reflection") # USEFUL API FOR READING GAME DATA
        self.read_msgpack_base64 = read_msgpack_base64
        # api related

        self.MaxItems = 12 # there are ways to increase this, but we will stick with this for now

        self.logger = Logger(log_level_player) # logger for the player class 
        self.log_level_player = log_level_player # log level for the player class, used in the logger

        self.cutscenes_quickfix() # ensure that it is not stuck in a cutscene, although this is not guaranteed to work in all cases

        # behavior related 
        self.attempts = attempts # number of attempts for various actions
        self.likely_running_into_wall = 0
        self.logger.log("Player instance created", "INFO")

        # environment related
        self.Moving = False


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
    def stamina(self) -> int: # Note that it can be regenerated using food
        self.logger.log("Requesting player stamina", "DEBUG")
        return self.gp(["player", "stamina"])

    @property
    def health(self) -> float: # shouldn't be an issue since there are barely any ways to die in the game
        self.logger.log("Requesting player health", "DEBUG")
        return self.wrap_result(self.r(function="getfield", args=["player", "health"]))

    @property
    def speed(self) -> float: # how many tiles per second the player can walk
        '''
        Returns the player speed in tiles per second
        '''
        self.logger.log("Requesting player speed", "DEBUG")
        return self.wrap_result(self.r(function="getproperty", args=["player", "speed"]))

    def walk1(self, speed: float) -> int: # conversion to miliseconds and int
        '''
        Returns the time in milliseconds it takes to walk 1 tile
        '''
        return int(1000 / speed)

    @property
    def money(self) -> int:
        self.logger.log("Requesting player money", "DEBUG")
        return self.gp(["player", "Money"])

    @property
    def location(self) -> str:
        '''
        Returns the current location of the player, e.g. "Farm", "Town", "Beach"
        '''
        self.logger.log("Requesting player location", "DEBUG")
        return self.wrap_result(self.r(function="getproperty", args=["player", "currentLocation"]))["NameOrUniqueName"]

    @property
    def position(self) -> tuple[float, float]:
        '''
        Returns the tile position of the player in the current location
        Not global position
        '''
        self.logger.log("Requesting player position", "DEBUG")
        res = self.wrap_result(self.r(function="getproperty", args=["player", "Tile"]))
        self.logger.log(f"Player position: {int(res['_Field_X'])}, {int(res['_Field_Y'])}", "INFO")
        return (int(res["_Field_X"]), int(res["_Field_Y"]))
    
    @property
    def inventory(self) -> inventory:
        '''
        Returns the inventory of the player, which is a collection of items
        to get a list of items, use player.inventory.items
        each item is a pointer to an Item instance, gathering the inventory is heavy
        so refer to pointer of items(the item class)
        '''
        collection_items = self.wrap_result(self.r(function="getproperty", args=["player", "Items"]))["_Collection_Items"]
        return inventory(collection_items, self.environment, self.log_level_player)

    @property
    def can_move(self) -> bool:
        envstate = self.environment.envstate # returns a dict with environment properties
        res = self.environment.world_env.isAvailable(envstate) # returns a list [bool, missing_properties_if_any]  
        if res[0]:
            return True
        else:
            return False

    def normal_action(self, key: str, durationms: int) -> list: 
        '''
        Returns a list either [True] or [False, missing_properties]
        where True means the action was successful and False means it failed
        '''
        return self.environment.world_action([key, durationms])

    def cutscenes_quickfix(self) -> None:
        self.environment.game_instance.skip_events(False)
        self.environment.game_instance.skip_events(True)
        self.close_dialogue()
        self.logger.log("Cutscenes quickfix applied", "DEBUG")

    def get_direction_key(self, target_position: tuple[int, int]) -> str:
        '''
        Returns the key to press to move to the target position
        '''
        x, y = target_position
        xi, yi = self.position
        if x > xi:
            return "d"
        elif x < xi:
            return "a"
        elif y > yi:
            return "s"
        elif y < yi:
            return "w"
        else:
            return ""

    @retry(tries=attempts["api_call"])
    async def single_step(self, target_position: tuple[int, int]) -> Exception | None: # will raise an exception if the environment is not available
        '''
        bound to fail often or frame-dependent, use walk_to instead to guarantee success
        [NOTE] Won't check if target position is reachable, it will just try to move there
        '''
        if self.can_move == False:
            self.logger.log("environment not available", "WARNING")
            raise Exception("Player cannot move, environment is not available")
        x, y = target_position
        xi, yi = self.position
        key = self.get_direction_key(target_position)
        if key == "":
            return None  # already at the target position, no need to move
        distance = ((x - xi) ** 2 + (y - yi) ** 2) ** 0.5
        duration = int(distance * self.walk1(self.speed))
        success: list = self.normal_action(key, duration)
        if not success[0]:
            raise Exception(f"Failed to move to position ({x}, {y})\nwith missing properties: {success[1]}") 

    def check_if_reachable_by_breaking(self, target_position: tuple[int, int]) -> bool:
        '''
        Checks if the target position is reachable by breaking obstacles in the way
        Returns True if reachable, False otherwise
        '''
        self.logger.log(f"Checking if position {target_position} is reachable by breaking", "DEBUG")
        energy_graph = self.environment.get_energy_graph()
        try:
            res = nx.has_path(energy_graph, self.position, target_position)
        except nx.NetworkXNoPath:
            self.logger.log(f"Position {target_position} is NOT reachable by breaking", "DEBUG")
            return False
        if res:
            self.logger.log(f"Position {target_position} is reachable by breaking", "DEBUG")
            return True
        else:
            return False
    def check_if_reachable_by_walking(self, target_position: tuple[int, int]) -> bool:
        '''
        Checks if the target position is reachable by walking
        Returns True if reachable, False otherwise
        '''
        self.logger.log(f"Checking if position {target_position} is reachable by walking", "DEBUG")
        strictly_collision_graph = self.environment.get_collision_graph()
        try:
            res = nx.has_path(strictly_collision_graph, self.position, target_position)
        except nx.NetworkXNoPath:
            self.logger.log(f"Position {target_position} is NOT reachable by walking", "DEBUG")
            return False
        if res:
            self.logger.log(f"Position {target_position} is reachable by walking", "DEBUG")
            return True
        else:
            return False

    def wait_until_pos_or_not_moving(self, target_position: tuple[int, int]) -> bool:
        '''
        Waits until the player reaches the target position or stops moving
        Returns True if the player reached the target position, False if the player stopped moving
        '''
        previous_position = self.position
        recurrence = 0
        x, y = target_position
        while True:
            current_position = self.position
            if current_position == (x, y):
                self.logger.log(f"Player reached target position {target_position}", "INFO")
                return True
            if current_position == previous_position:
                self.logger.log(f"Player stopped moving at position {current_position}", "WARNING")
                if recurrence <= 2:
                    self.logger.log(f"Player is almost certainly running into a wall", "DEBUG")
                    self.likely_running_into_wall += 1
                return False
            previous_position = current_position
            recurrence += 1
            time.sleep(0.05)

    @retry(tries=attempts["walk_to"])
    def walk_to(self, target_position: tuple[int, int], allow_breaking: bool = False) -> Exception | None:
        self.Walking = True

        # edge cases
        self.cutscenes_quickfix()
        current_position = self.position
        if current_position == target_position:
            self.logger.log(f"Already at target position {target_position}, no need to walk", "DEBUG")
            return
        try:
            strictly_collision_graph = self.environment.get_collision_graph()
            path = nx.shortest_path(strictly_collision_graph, current_position, target_position)
        except nx.NetworkXNoPath:
            try:
                energy_graph = self.environment.get_energy_graph()
                path = nx.dijkstra_path(energy_graph, current_position, target_position)
            except nx.NetworkXNoPath:
                self.logger.log(f"Target position {target_position} is not reachable by walking or breaking", "ERROR")
                raise Exception(f"Target position {target_position} is not reachable by walking or breaking")
            if allow_breaking: # back to breaking logs u go!
                self.logger.log(f"Target position {target_position} is reachable by breaking, but not by walking, following energy path", "DEBUG")
                self.follow_energy_path(path) #type: ignore
                return
            else:
                self.logger.log(f"Target position {target_position} is not reachable by walking, but reachable by breaking, but allow_breaking is False", "ERROR")
                return
        # edge cases

        optimized_path = self.optimize_path(path) #type: ignore
        # if len(optimized_path) >= 1:
        #     optimized_path = optimized_path[1:]  # remove the first point, since it is the current position
        # path logic 
        if self.logger.level == 0:  # DEBUG level is 0, sorry for the magic numbers
            self.environment.print_path(optimized_path)  # this calls print, therefore it couldnt be blocked otherwise

        for current_target in optimized_path:
            self.logger.log(f"current target position: {current_target}", "INFO")

            asyncio.run(self.single_step(current_target)) # this will raise an exception if the environment is not available, including retries
            success = self.wait_until_pos_or_not_moving(current_target)  # wait until the player reaches the target position

            # retry walking to the target position if it failed
            for _ in range(self.attempts["walk_to"]):
                asyncio.run(self.single_step(current_target))  # try to walk to the target position again
                success = self.wait_until_pos_or_not_moving(current_target)
                if success:
                    break
            # retry walking to the target position if it failed

            if self.likely_running_into_wall >= self.attempts["assume_wall"]:
                self.likely_running_into_wall = 0
                graph = None
                try:
                    graph = self.environment.get_collision_graph()
                except nx.NetworkXNoPath:
                    graph = self.environment.get_energy_graph()
                if graph is None:
                    self.logger.log("Failed to get collision graph, cannot assume wall", "ERROR")
                    raise Exception("Failed to get collision graph, cannot assume wall")
                path = nx.shortest_path(graph, self.position, target_position)
                self.logger.log(f"Player is likely running into a wall, assuming wall at position {path[1]}", "WARNING")
                self.environment.draw_learned_tile(path[1], "Building") # mark the tile as a wall in the environment
                graph = self.environment.get_energy_graph() # update the graph
                path = nx.dijkstra_path(graph, self.position, target_position) # get the new path
                self.follow_energy_path(path) # type: ignore
                if path[1] == target_position:
                    self.logger.log(f"Target position {target_position} is blocked by a wall", "ERROR")
                    raise Exception(f"Target position {target_position} is blocked by a wall")
                continue


            if self.position != current_target:
                self.logger.log(f"Failed to walk to target position {current_target}, current position is {self.position}", "ERROR")
                raise Exception(f"Failed to walk to target position {current_target}, current position is {self.position}")
            else:
                self.like_running_into_wall = 0  # reset the counter if the player reached the target position
                       
    def optimize_path(self, path: list[tuple]) -> list[tuple]:
        '''
        Guarantees that the path only counts turns, therefore
        reducing unnecesary convolution over points that are in a straight line
        '''
        self.logger.log("Optimizing path", "DEBUG")
        if not path:
            self.logger.log("Path is empty, returning empty path", "CRITICAL")
            return []
        if len(path) <= 1:
            self.logger.log("Path has less than 2 points, returning original path", "DEBUG")
            return path
        optimized_path = []
        next_point = None
        previous_slope = (path[0][1] - path[1][1]) / (path[0][0] - path[1][0]) if path[0][0] != path[1][0] else float('inf')  # kinda like the derivative
        for point in path:
            next_point = path[path.index(point) + 1] if path.index(point) + 1 < len(path) else point
            slope = (point[1] - next_point[1]) / (point[0] - next_point[0]) if point[0] != next_point[0] else float('inf')
            if slope != previous_slope:
                optimized_path.append(point)
                previous_slope = slope
        if path[-1] not in optimized_path:
            optimized_path.append(path[-1])
        return optimized_path

    def face_direction(self, target_position: tuple[int, int]) -> Exception | None:
        dirs = {
            "w": 0,
            "d": 1,
            "s": 2,
            "a": 3
        }
        key = self.get_direction_key(target_position)
        if key not in dirs:
            self.logger.log(f"target_position is the same as the player", "ERROR")
        distance = ((target_position[0] - self.position[0]) ** 2 + (target_position[1] - self.position[1]) ** 2) ** 0.5
        assert distance == 1, "Target position must be adjacent to current position, not diagonally"
        res = self.r(function="invokemethod", args=["player", "faceDirection", dirs[key]])
        if not res["Success"]:
            self.logger.log(f"Failed to face direction {key}: {res['Error']}", "ERROR")
            raise Exception(f"Failed to face direction {key}: {res['Error']}")
    
    def break_next_tile(self, target_position: tuple[int, int], tool: Item) -> Exception | None:
        '''
        Breaks an obstacle at the target position
        '''
        current_position = self.position
        assert target_position != current_position, "Target position cannot be the same as current position"
        distance = ((target_position[0] - current_position[0]) ** 2 + (target_position[1] - current_position[1]) ** 2) ** 0.5
        assert distance == 1, "Target position must be adjacent to current position"
        self.logger.log(f"Breaking tile at position {target_position} with tool {tool.name}", "DEBUG")
        self.face_direction(target_position)
        tool(use=True)  # use the tool to break the tile


    @retry(tries=attempts["follow_energy_path"])
    def follow_energy_path(self, path: list):

        self.moving = True
        # tool pointers
        current_target = None
        tools = {"Pickaxe": None, "Axe": None, "Scythe": None}
        self.logger.log("Following energy path", "DEBUG")
        for item in self.inventory.items:
            if item.name in tools:
                tools[item.name] = item
            elif item.name == "MeleeWeapon":
                tools["Scythe"] = item
        # tool pointers

        for point in path:
            point_properties = self.environment.spatial_state[point[0]][point[1]].properties
            if "tool" in point_properties:

                # edge case
                if point_properties["tool"] not in tools:
                    self.logger.log(f"Tool {point_properties['tool']} not found in inventory, skipping point {point}", "WARNING")
                    continue
                # edge case
                
                # utils
                tool = point_properties["tool"]
                current_target = path[path.index(point) - 1] 
                # utils

                self.walk_to(current_target, allow_breaking=True) # last walkable point in theory

                # edge case
                if "health" in point_properties:
                    health = point_properties["health"]
                    self.face_direction(point)
                    for i in range(health):
                        self.logger.log(f"Dealing damage to point {point} ({i+1}/{health})", "DEBUG")
                        tools[tool](use=True) # type: ignore
                        self.environment.update_spatial_state()
                        x, y = point
                        if not "collision" in self.environment.spatial_state[x][y].properties:
                            break
                    self.environment.update_spatial_state()
                    if self.logger.level == 0:
                        self.environment.print_path(path[path.index(point):])
                    continue
                # edge case

                self.break_next_tile(point, tools[tool]) # type: ignore
                self.environment.update_spatial_state()
                if self.logger.level == 0: # DEBUG level is 0, sorry for the magic numbers
                    self.environment.print_path(path[path.index(point):]) # this calls print, therefore it couldnt be blocked otherwise
                time.sleep(0.1)
        self.Moving = False 

    def close_dialogue(self):
        self.logger.log("Closing dialogue", "INFO")
        self.logger.log("Invoking exitActiveMenu\nIt requires some special movement sometimes", "WARNING")
        return self.environment.game_instance.reflection(function="invokemethod", args=["game1", "exitActiveMenu"])
