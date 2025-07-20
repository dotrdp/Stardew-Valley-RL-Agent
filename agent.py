import torch
from ENV import environment
from player import player 
from API import StardewModdingAPI

# [NOTE] set your method here
METHOD = "ssh+tty"
LOG_LEVEL = "DEBUG"
# [DEBUG]: 0
# [INFO]: 1
# [WARNING]: 2
# [ERROR]: 3
# [FATAL]: 4 ;[NOTE] to call fatal set the string to "CRITICAL"

game_environment = environment(StardewModdingAPI(method=METHOD, loglevel=LOG_LEVEL), loglevel=LOG_LEVEL)
player_agent = player(game_environment)
print(game_environment.map.grid)
player_agent.normal_action("a", 1000)

class RewardSystem:
    def __init__(self, env: environment):
        self.env = env

    def get_reward(self, state, action):
        pass
        


class Agent:
    def __init__(self, env: environment, player: player):
        self.env = env
        self.player = player

    def act(self, state):
        # Convert the state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get the action from the player



