import torch.nn as nn
import torch
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """    
    """
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.fc_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)
        output = self.fc_o(context)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    """
    
    """
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, nhead)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn(x2))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class DiffusionTransformer(nn.Module):
    """The main model class, which includes a stack of encoder layers and a final linear layer.
    The concept of diffusion models is explained in the paper "Denoising Diffusion Probabilistic Models" by Ho et al. (2020).
    We can find an implementation on: https://github.com/hojonathanho/diffusion
    """
    def __init__(self, d_model, nhead, num_encoder_layers, d_ff, dropout=0.1, max_len=5000):
        super(DiffusionTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff, dropout=dropout) for _ in range(num_encoder_layers)])
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.final_layer = nn.Linear(d_model, d_model)

    def forward(self, src):
        src = self.pos_encoding(src)
        for layer in self.encoder_layers:
            src = layer(src)
        output = self.final_layer(src)
        return output
