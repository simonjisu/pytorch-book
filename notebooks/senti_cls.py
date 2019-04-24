import torch
import torch.nn as nn

class SentimentCls(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size,
                 num_layers=3, batch_first=True, bidirec=True):
        super(SentimentCls, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.n_direct = 2 if bidirec else 1
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.rnn_layer = nn.LSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=batch_first,
                                 bidirectional=bidirec)
        self.linear = nn.Linear(self.n_direct*hidden_size, output_size)

    def forward(self, x):
        embeded = self.embedding_layer(x)
        hidden, cell = self.init_hiddens(x.size(0), self.hidden_size, device=x.device)
        output, (hidden, cell) = self.rnn_layer(embeded, (hidden, cell))
        last_hidden = torch.cat([h for h in hidden[-self.n_direct:]], dim=1)
        scores = self.linear(last_hidden)
        return scores.view(-1)
    
    def init_hiddens(self, batch_size, hidden_size, device):
        hidden = torch.zeros(self.n_direct*self.n_layers, batch_size, hidden_size)
        cell = torch.zeros(self.n_direct*self.n_layers, batch_size, hidden_size)
        return hidden.to(device), cell.to(device)