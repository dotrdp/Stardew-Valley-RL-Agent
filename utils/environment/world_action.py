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
 
