import torch
from ENV import environment
from player import player 
from API import StardewModdingAPI
from map_wrapper import Map

# [NOTE] set your method here
METHOD = "ssh+tty"
LOG_LEVEL = "DEBUG"
# [DEBUG]: 0
# [INFO]: 1
# [WARNING]: 2
# [ERROR]: 3
# [FATAL]: 4 ;[NOTE] to call fatal set the string to "CRITICAL"

api = StardewModdingAPI(method=METHOD, loglevel=LOG_LEVEL)
game_environment = environment(api, loglevel=LOG_LEVEL)
player_agent = player(game_environment)
player_agent.normal_action("a", 1000)

class RewardSystem:
    def __init__(self, env: environment, map = None):
        self.env = env
        self.cached_state = map 

    def get_reward(self, state, action, map=None):
        if map != None:
            self.cached_state = map
            self.env.logger.log("RewardSystem: setting provided map as state", "DEBUG")
        if self.cached_state == None:
            self.env.logger.log("RewardSystem: requires a new map to draw a reward\notherwise it will try to use it's previous cached state as a state", "WARNING")
            self.env.logger.log("RewardSystem: previous state is None", "CRITICAL")
            self.env.logger.log("RewardSystem: cached state is None, instantiating a new Map class\nPLEASE DON'T LET IT TO DO THIS, CACHE THE MAP AS MUCH AS POSSIBLE", "WARNING")
            self.cached_state = Map(api)
        self.env.logger.log("RewardSystem: using the current map as a state", "DEBUG")
        



        


class Agent:
    def __init__(self, env: environment, player: player):
        self.env = env
        self.player = player

    def act(self, state):
        # Convert the state to a tensor
        pass

        # Get the action from the player



