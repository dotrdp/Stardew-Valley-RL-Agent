import torch
import lightning as L 
from utils import environment, player
from muon import MuonWithAuxAdam # give it some cutting edge shall we?, seems to be better than all mighty adam, besides it is currently trending and it's main use case is around RL and neural networks, not yet implemented in pytorch lightning hence this package

# figured out in the past LSTMs were used for RL, but they're just outdated transformers
item_rewards = {

}
class RewardSystem:
    def __init__(self, env: environment, map=None):
        self.env = env

    def get_items(self, player):
        pass
        

class NN(L.LightningModule):
    def __init__(self, env: environment, player: player):
        super().__init__()


class Agent:
    def __init__(self, env: environment, player: player):
        self.env = env
        self.player = player

    def act(self, state):
        # Convert the state to a tensor
        pass

        # Get the action from the player
