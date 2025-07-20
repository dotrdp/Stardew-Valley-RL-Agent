import torch
from ENV import environment
from player import player 
from API import StardewModdingAPI

# [NOTE] set your method here
METHOD = "ssh+tty"

game_environment = environment(StardewModdingAPI(method=METHOD))
player_agent = player(game_environment)
print(game_environment.map.grid)

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



