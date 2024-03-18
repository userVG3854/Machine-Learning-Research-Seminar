import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SocialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(SocialEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Define linear layers for query, key, and value projections
        self.linear_q = nn.Linear(input_dim, hidden_dim)
        self.linear_k = nn.Linear(input_dim, hidden_dim)
        self.linear_v = nn.Linear(input_dim, hidden_dim)

        # Define linear layer for output projection
        self.linear_out = nn.Linear(hidden_dim, input_dim)

        # Multi-head attention mechanism
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)

        # Feedforward network
        self.feedforward = FeedForward(hidden_dim)

        # Layer normalization
        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: input data (batch_size, seq_len, input_dim)

        # Project input for query, key, and value
        query = self.linear_q(x)
        key = self.linear_k(x)
        value = self.linear_v(x)

        # Multi-head attention
        attn_output = self.attention(query, key, value)

        # Add residual connection and apply layer normalization
        x = self.layernorm1(x + self.dropout(attn_output))

        # Feedforward network
        ff_output = self.feedforward(x)

        # Add residual connection and apply layer normalization
        x = self.layernorm2(x + self.dropout(ff_output))

        # Project output back to input dimension
        encoded_output = self.linear_out(x)

        return encoded_output

# Define Multi-Head Attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear layers for query, key, and value projections
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)

        # Linear layer for output projection
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

        # Attention dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        # Linear projections
        Q = self.linear_q(query)
        K = self.linear_k(key)
        V = self.linear_v(value)

        # Split heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention score calculation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())

        # Apply softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention to value
        output = torch.matmul(attention_weights, V)

        # Concatenate heads
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)

        # Linear projection
        output = self.linear_out(output)

        return output

# Define FeedForward network
class FeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.linear1(x)))
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = F.relu(self.linear1(src))
        src = src + self.dropout(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = F.relu(self.linear1(tgt))
        tgt = tgt + self.dropout(tgt2)
        return tgt

class ResNetTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(ResNetTransformer, self).__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, tgt):
        memory = src
        for i, layer in enumerate(self.encoder_layers):
            residual = memory
            memory = layer(memory)
            memory += residual if i % 2 == 0 else 0  # Add residual connection every 2 layers

        output = tgt
        for j, layer in enumerate(self.decoder_layers):
            residual = output
            output = layer(output, memory)
            output += residual if j % 2 == 0 else 0  # Add residual connection every 2 layers

        return self.norm(output)



class DenoisingLayer(nn.Module):
    def __init__(self, input_dim, noise_scale):
        super(DenoisingLayer, self).__init__()
        self.noise_scale = noise_scale

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_scale
        noisy_x = x + noise
        return noisy_x, noise

class CNNDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNDecoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x + F.relu(self.conv2(x))
        x = x + F.relu(self.conv3(x))
        return x


class DiffusionTransformer(nn.Module):
    """The main model class, which includes a stack of encoder layers and a final linear layer.
    The concept of diffusion models is explained in the paper "Denoising Diffusion Probabilistic Models" by Ho et al. (2020).
    We can find an implementation on: https://github.com/hojonathanho/diffusion
    """
    def __init__(self, d_model, nhead, num_encoder_layers, d_ff, dropout=0.1, max_len=5000, noise_scale=0.1):
        super(DiffusionTransformer, self).__init__()
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        self.social_encoder = SocialEncoder(d_model, d_model, num_encoder_layers, nhead, dropout)
        self.resnet_transformer = ResNetTransformer(d_model, nhead, num_encoder_layers)
        self.denoising_layer = DenoisingLayer(d_model, noise_scale)
        self.final_layer = nn.Linear(d_model, d_model)
        self.cnn_decoder = CNNDecoder(d_model, d_model)

    def forward(self, src):
        src = self.pos_encoding(src)
        src = self.social_encoder(src)
        noisy_src, noise = self.denoising_layer(src)
        tgt = noisy_src  # For simplicity, we use the noisy input for both the encoder and decoder.
        output = self.resnet_transformer(src, tgt)
        output = self.final_layer(output)
        output = self.cnn_decoder(output.transpose(1, 2)).transpose(1, 2)
        return output