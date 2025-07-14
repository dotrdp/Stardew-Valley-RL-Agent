from API import StardewModdingAPI

def check_wanted_conditions_in_env(wanted, environment):
    for prop in wanted[:]:
        if environment[prop] != False:
            wanted.remove(prop)
    if len(wanted) == 0:
        return [True]
    else:
        return [False, wanted]

class normal_movement():
    def __init__(self, stardew_modding_api):
        self.game_instance = stardew_modding_api

    def check_normal_movement(self, res):
        wanted = ["IsPlayerFree", "CanPlayerMove", "IsWorldReady"]
        for prop in wanted[:]:
            if res[prop] != False:
                wanted.remove(prop)
        if len(wanted) == 0:
            return True
        else:
            return [False, wanted]

    def __call__(self, action):
        key, ms = action
        return self.game_instance.hold_key(key, ms)
        
class environment():
    def __init__(self, stardew_modding_api):
        self.game_instance = stardew_modding_api 
        self.nmv = normal_movement(stardew_modding_api)
    def __call__(self, target, action):
        if target == "normal_movement":
            res = self.nmv.check_normal_movement(self.envstate())
            if res == True: 
                return self.nmv(action)
            elif res[0] == False:
               print("the environment couldn't be interacted using normal movement")
               print("missing properties:", res[1])
            
        else:
            return "Unknown target."
    def envstate(self) -> dict:
        res = {
            "IsPlayerFree": False,
            "CanPlayerMove": False,
            "IsWorldReady": False}
        res["IsPlayerFree"] = self.game_instance.reflection(function="GetProperty", args=["Context", "IsPlayerFree"])["Result"]
        res["CanPlayerMove"] = self.game_instance.reflection(function="GetProperty", args=["Context", "CanPlayerMove"])["Result"]
        res["IsWorldReady"] = self.game_instance.reflection(function="GetProperty", args=["Context", "IsWorldReady"])["Result"]
        return res

    def __getattr__(self, name):
        if name == "state":
            return self.state()
        else:
            raise AttributeError(f"'environment' has no attribute '{name}'")
        
env = environment(StardewModdingAPI(method="ssh+tty"))
env("normal_movement", "up")
