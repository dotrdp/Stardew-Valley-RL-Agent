import torch.nn as nn
import torch.nn.functional as F

# code adapted from lightning documentation 

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, ninp=200, nhid=200, nlayers=2, dropout=0.2):
        super().__init__()
        self.input_size = input_size 
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(input_size, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, input_size)

        self.nlayers = nlayers
        self.nhid = nhid
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output) # great, fully packaged LSTm with MLP built-in
        decoded = decoded.view(-1, self.input_size)
        return F.relu(decoded), hidden # we don't want softmax here 

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, bsz, self.nhid),
            weight.new_zeros(self.nlayers, bsz, self.nhid),
        )
