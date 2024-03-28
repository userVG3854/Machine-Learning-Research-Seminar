import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Reshape the output to have the same shape as the input
        out = out.contiguous().view(-1, self.hidden_size * (2 if self.bidirectional else 1))
        out = self.fc(out)
        out = out.view(x.size(0), x.size(1), -1)
        return out