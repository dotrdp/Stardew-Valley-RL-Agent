import torch.nn as nn
import lightning as L
from utils import environment, player, StardewModdingAPI
from ai_utils import SimpleLSTM, env_index, get_state_embedding
from muon import MuonWithAuxAdam # give it some cutting edge shall we?, seems to be better than all mighty adam, besides it is currently trending and it's main use case is around RL and neural networks, not yet implemented in pytorch lightning hence this package

# figured out in the past LSTMs were used for RL, but they're just outdated transformers

# we doing LSTM ig, but let's give it attention please.

# pretty much X -> Embedding -> torch built-in LSTM -> attention -> linear layer -> output + skip connection(layer norm)

api = StardewModdingAPI()
environment = environment(api)
player = player(environment)
get_state_embedding(environment, player)

class LSTM_attn(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = SimpleLSTM(input_dim, hidden_dim) # this thing was adapted from lightning documentation, it comes with MLP and uses the torch built-in LSTM, which should be fairly good

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=2) # figured out heads aint that useful

        self.fc = nn.Sequential(
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        y, hidden, memory = self.lstm(x)
        # watch me use these 3 as QKV
        attn_output, _ = self.attention(y, hidden, memory)

        output = self.fc(attn_output) + attn_output

        return output

    def configure_optimizers(self):
        muon_hidden_weights = [p for p in self.lstm.rnn.parameters() if p.ndim >= 2]
        muon_hidden_gains_biases = [p for p in self.lstm.rnn.parameters() if p.ndim < 2]
        adam = [*self.attention.parameters(), *self.fc.parameters()]
        param_groups = [
            dict(params=muon_hidden_weights, use_muon=True,
                 lr=0.02, weight_decay=0.01),
            dict(params=muon_hidden_gains_biases+adam, use_muon=False,
                 lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
        return optimizer
# just some simple LSTM with attention, using my beloved muon optimizer, props to the guys who implemented it

class Agent:
    def __init__(self, env: environment, player: player):
        self.env = env
        self.player = player

    def act(self, state):
        # Convert the state to a tensor
        pass

        # Get the action from the player
