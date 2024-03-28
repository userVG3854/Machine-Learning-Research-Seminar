import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=True, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.bn = nn.BatchNorm1d(hidden_size * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        # Permute the input to have the shape (number_players, number_frames, number_features)
        x = x.permute(1, 0, 2)
        out, _ = self.rnn(x)
        # Reshape the output for batch normalization
        out = out.contiguous().view(-1, out.size(-1))
        out = self.bn(out)
        out = nn.functional.dropout(out, p=self.dropout, training=self.training)
        # Reshape the output back to its original shape
        out = out.view(x.size(0), x.size(1), -1)
        out = self.fc(out)
        # Permute the output back to the original shape (number_frames, number_players, number_features)
        out = out.permute(1, 0, 2)
        return out