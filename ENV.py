from API import StardewModdingAPI

class normal_movement():
    def __init__(self, stardew_modding_api):
        self.game_instance = stardew_modding_api
    def __call__(self, action):
        res = self.game_instance.reflection(function="GetProperty", args=["Context", "IsPlayerFree"])
        
class environment():
    def __init__(self, stardew_modding_api):
        self.game_instance = stardew_modding_api 
        self.normal_movement = normal_movement(stardew_modding_api)
    def __call__(self, target, action):
        if target == "normal_movement":
            return self.normal_movement(action)
        else:
            return "Unknown target."
        
        
env = environment(StardewModdingAPI(method="ssh+tty"))
env("normal_movement", "up")
