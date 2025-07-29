from map_wrapper import Map, Tile
from logger import Logger
import networkx as nx
from dotenv import dotenv_values

prefs = dotenv_values(".env")
loglevel = prefs.get("debug_level_environment", "ERROR")
debug_level = prefs.get("debug_level", "ERROR")
if debug_level != "ERROR":
    loglevel = debug_level


def check_wanted_conditions_in_env(wanted, environment):
    for prop in wanted[:]:
        if environment[prop] != False:
            wanted.remove(prop)
    if len(wanted) == 0:
        return [True]
    else:
        return [False, wanted]

class world_action():
    def __init__(self, stardew_modding_api):
        self.game_instance = stardew_modding_api
        self.wanted_conditions = ["IsPlayerFree", "CanPlayerMove", "IsWorldReady"]
        self.check_wanted_conditions_in_env = check_wanted_conditions_in_env

    def isAvailable(self, res) -> list:
        if res != None:
            return self.check_wanted_conditions_in_env(self.wanted_conditions, res)
        else:
            self.game_instance.logger.log("Environment state is None", "CRITICAL")
            return [False, self.wanted_conditions]
        
    def __call__(self, action):
        key, ms = action
        return self.game_instance.hold_key(key, ms)
        
class environment():
    def __init__(self, stardew_modding_api, loglevel = loglevel):
        self.game_instance = stardew_modding_api 
        self.world_env = world_action(self.game_instance)
        self.map = Map(self.game_instance)
        self.spatial_state = self.map.get_data()
        self.learned_spatial_features = {}
        self.logger = Logger(loglevel)
        self.logger.log("Environment initialized", "INFO")

    def world_action(self, action):
        self.logger.log(f"trying to perform action{action} on world action environment", "DEBUG")
        self.logger.log("Checking if the world action is available", "INFO")
        res = self.world_env.isAvailable(self.envstate)
        if res[0] == True: 
            self.logger.log("World action is available, executing action", "INFO")
            self.world_env(action)
        elif res[0] == False:
            self.logger.log(f"World action is not available, missing conditions: {res[1]}", "WARNING")
        return res
    @property
    def envstate(self) -> dict:
        res = {
            "IsPlayerFree": False,
            "CanPlayerMove": False,
            "IsWorldReady": False}
        res["IsPlayerFree"] = self.game_instance.reflection(function="getproperty", args=["Context", "IsPlayerFree"])["Result"]
        res["CanPlayerMove"] = self.game_instance.reflection(function="getproperty", args=["Context", "CanPlayerMove"])["Result"]
        res["IsWorldReady"] = self.game_instance.reflection(function="getproperty", args=["Context", "IsWorldReady"])["Result"]
        self.logger.log(f"Environment state: {res}", "DEBUG")
        return res
    def update_spatial_state(self):
        self.logger.log("Updating spatial state", "DEBUG")
        self.spatial_state = self.map.get_data()
        if self.learned_spatial_features != {}:
            for i, v in self.learned_spatial_features.items():
                x, y = i
                self.spatial_state[x][y] = v
        self.logger.log("Spatial state updated", "DEBUG")
    def get_collision_graph(self):
        graph = nx.Graph()
        for x in range(len(self.spatial_state)):
            for y in range(len(self.spatial_state[x])):
                tile = (x, y)
                if "collision" in self.spatial_state[x][y].properties:
                    continue
                neighbors = []
                if x+1 <= len(self.spatial_state) - 1:
                    neighbors.append((x+1, y))
                if x-1 >= 0:
                    neighbors.append((x-1, y))
                if y+1 <= len(self.spatial_state[x]) - 1:
                    neighbors.append((x, y+1))
                if y-1 >= 0:
                    neighbors.append((x, y-1))
                for neighbor in neighbors:
                    if "collision" not in self.spatial_state[neighbor[0]][neighbor[1]].properties:
                        
                        graph.add_edge(tile, neighbor, weight=1)
        self.logger.log("Walkable graph created", "DEBUG")
        return graph

    def get_energy_graph(self):
        graph = nx.Graph()
        rows = len(self.spatial_state) - 1 
        for x in range(rows):
            cols = len(self.spatial_state[x]) - 1
            for y in range(cols):
                tile = (x, y)

                props = self.spatial_state[x][y].properties

                if "collision" in props and "tool" not in props:
                    continue
                graph.add_node(tile)
                neighbor_positions = [
                    (min(x+1, len(self.spatial_state)), y),
                    (x, min(y+1, cols)),
                    (max(x-1, 0), y),
                    (x, max(y-1, 0))
                ]
                for nx_pos, ny_pos in neighbor_positions:
                    neighbor_props = getattr(self.spatial_state[nx_pos][ny_pos], "properties", [])
                    neighbor = (nx_pos, ny_pos)
                    if "collision" in neighbor_props:
                        if "tool" in props:
                            if "health" in props:
                                graph.add_edge(tile, neighbor, weight=props["health"])
                                continue
                            graph.add_edge(tile, neighbor, weight=2)
                    else:
                        graph.add_edge(tile, neighbor, weight=1)
            # Ensure graph is connected
        if nx.is_connected(graph):
            self.logger.log("Energy graph is connected", "DEBUG")
        self.logger.log("Energy graph created and connected", "DEBUG")
        return graph

    
    def print_path(self, path):
        self.logger.log("Drawing path on the map", "DEBUG")
        copy = self.spatial_state.copy()
        for point in path:
            x, y = point
            type = copy[x][y].type
            if type == "player":
                continue
            if "tool" in copy[x][y].properties:
                if "collision" in copy[x][y].properties:
                    copy[x][y].type = "debug_red"
                    continue
            copy[x][y].type = "debugmarker"
        res = ""
        copy = list(zip(*copy))  # Transpose the matrix for easier printing
        for x in range(len(copy)):
            for y in range(len(copy[x])):
                tile = copy[x][y]
                res += str(tile)
            res += "\n"
        print(res)

    def draw_learned_tile(self, x, y, type):
        self.logger.log(f"Drawing tile at ({x}, {y}) with type {type}", "DEBUG")
        self.spatial_state[x][y] = Tile(x, y, type) 
        self.learned_spatial_features[(x, y)] = Tile(x, y, type) 
