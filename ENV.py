from map_wrapper import Map
from logger import Logger


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

    def isAvailable(self, res):
        if res != None:
            return self.check_wanted_conditions_in_env(self.wanted_conditions, res)
        else:
            print("The environment state is non existent.")
            return [False, self.wanted_conditions]
        
    def __call__(self, action):
        key, ms = action
        return self.game_instance.hold_key(key, ms)
        
class environment():
    def __init__(self, stardew_modding_api, loglevel = "CRITICAL"):
        self.game_instance = stardew_modding_api 
        self.world_env = world_action(self.game_instance)
        self.map = Map(self.game_instance)
        self.logger = Logger(loglevel)
        self.logger.log("Environment initialized", "INFO")

    def world_action(self, action):
        res = self.world_env.isAvailable(self.envstate())
        self.logger.log("Checking if the world action is available", "INFO")
        if res[0] == True: 
             self.logger.log("World action is available, executing action", "INFO")
             return self.world_env(action)
        elif res[0] == False:
            self.logger.log(f"World action is not available, missing conditions: {res[1]}", "WARNING")
            return res
    def envstate(self) -> dict:
        res = {
            "IsPlayerFree": False,
            "CanPlayerMove": False,
            "IsWorldReady": False}
        res["IsPlayerFree"] = self.game_instance.reflection(function="getproperty", args=["Context", "IsPlayerFree"])["Result"]
        res["CanPlayerMove"] = self.game_instance.reflection(function="getproperty", args=["Context", "CanPlayerMove"])["Result"]
        res["IsWorldReady"] = self.game_instance.reflection(function="getproperty", args=["Context", "IsWorldReady"])["Result"]
        return res
        
