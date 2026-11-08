from dotenv import dotenv_values

from ..logger import Logger
from .world_action import world_action
from .environment_graphs import Env_Graphs
from .path_utils import print_path as pt
from .map_wrapper import Map, Tile


prefs = dotenv_values(".env")
loglevel = prefs.get("debug_level_environment", "ERROR")
debug_level = prefs.get("debug_level", "ERROR")
if debug_level != "ERROR":
    loglevel = debug_level


       
class environment():
    def __init__(self, stardew_modding_api, loglevel = loglevel):
        self.game_instance = stardew_modding_api 
        self.world_env = world_action(self.game_instance)
        self.map = Map(self.game_instance)
        self.spatial_state = self.map.get_data()
        self.time, self.season, self.raining, self.snow = self.map.time, self.map.season, self.map.raining, self.map.snowing
        self.learned_spatial_features = {}
        self.previous_spatial_state = {}
        self.logger = Logger(loglevel)
        self.env_graphs = Env_Graphs(self.spatial_state, self.logger)
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
            "IsWorldReady": False,
            "time": self.game_instance.reflection(function="getproperty", args=["Context", "Time"])["Result"],
            }
        res["IsPlayerFree"] = self.game_instance.reflection(function="getproperty", args=["Context", "IsPlayerFree"])["Result"]
        res["CanPlayerMove"] = self.game_instance.reflection(function="getproperty", args=["Context", "CanPlayerMove"])["Result"]
        res["IsWorldReady"] = self.game_instance.reflection(function="getproperty", args=["Context", "IsWorldReady"])["Result"]
        self.logger.log(f"Environment state: {res}", "DEBUG")
        return res

    def update_spatial_state(self):
        self.logger.log("Updating spatial state", "DEBUG")
        self.previous_spatial_state = self.spatial_state.copy()
        self.spatial_state = self.map.get_data()
        self.time, self.season, self.raining, self.snow = self.map.time, self.map.season, self.map.raining, self.map.snowing
        if self.learned_spatial_features != {}:
            for i, v in self.learned_spatial_features.items():
                x, y = i
                self.spatial_state[x][y] = v
        self.env_graphs.update_spatial_state(self.spatial_state)
        self.logger.log("Spatial state updated", "DEBUG")

    def get_collision_graph(self):
        return self.env_graphs.get_collision_graph()
    def get_energy_graph(self):
        return self.env_graphs.get_energy_graph()

    def print_path(self, path) -> None:
        pt(self.spatial_state, path)

    def draw_learned_tile(self, position: tuple[int, int], type: str):
        x, y = position
        self.learned_spatial_features[x][y] = Tile(x, y, type)
        self.spatial_state[x][y] = Tile(x, y, type)

