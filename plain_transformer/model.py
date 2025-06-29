import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    Adds information about position in the sequence to the input embeddings.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, embedding_dim]
        x = x + self.pe[:x.size(0), :]
        return x


class PlainTransformer(nn.Module):
    """
    Custom Transformer architecture for time series forecasting.
    """
    def __init__(self, opt, input_dim):
        super(PlainTransformer, self).__init__()
        
        self.opt = opt
        
        # Feature embedding
        self.embedding = nn.Linear(input_dim, opt.embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(opt.embedding_dim)
        
        if opt.use_torch_transformer:
            # Use PyTorch's built-in transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=opt.embedding_dim,
                nhead=opt.num_heads,
                dim_feedforward=opt.embedding_dim * 4,
                dropout=opt.dropout,
                batch_first=False,
                activation='gelu'  # Using GELU for potentially better performance
            )
            
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=opt.num_encoder_layers,
                norm=nn.LayerNorm(opt.embedding_dim)
            )
        else:
            # Implement custom transformer encoder layers
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(opt) 
                for _ in range(opt.num_encoder_layers)
            ])
            self.norm = nn.LayerNorm(opt.embedding_dim)
        
        # Output layer
        self.output_layer = nn.Linear(opt.embedding_dim, opt.num_classes)
        
    def forward(self, src):
        # src: [batch_size, seq_len, input_dim]
        
        # Transpose for transformer input: [seq_len, batch_size, input_dim]
        src = src.permute(1, 0, 2)
        
        # Embedding
        src = self.embedding(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        if self.opt.use_torch_transformer:
            # Transformer encoder (PyTorch implementation)
            output = self.transformer_encoder(src)
        else:
            # Custom transformer implementation
            for layer in self.layers:
                src = layer(src)
            output = self.norm(src)
        
        # Take the output of the last token for classification
        output = output[-1]
        
        # Output layer
        output = self.output_layer(output)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Custom implementation of transformer encoder layer.
    """
    def __init__(self, opt):
        super(TransformerEncoderLayer, self).__init__()
        
        d_model = opt.embedding_dim
        
        # Self-attention
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model, 
            num_heads=opt.num_heads, 
            dropout=opt.dropout
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(opt.dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(opt.dropout)
        self.dropout2 = nn.Dropout(opt.dropout)
        
        # Activation function
        self.activation = nn.GELU()
        
    def forward(self, src):
        # Self-attention block
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2)
        src = src + self.dropout1(src2)
        
        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = self.activation(src2)
        src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        
        return src


class MultiHeadAttention(nn.Module):
    """
    Custom implementation of multi-head attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # Get dimensions
        seq_len, batch_size, embed_dim = query.size()
        
        # Linear projections and reshape for multi-head attention
        q = self.query(query).view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        k = self.key(key).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        v = self.value(value).view(-1, batch_size, self.num_heads, self.head_dim).transpose(0, 2)
        
        # q: [num_heads, seq_len, batch_size, head_dim]
        # k, v: [num_heads, key_len, batch_size, head_dim]
        
        # Compute attention scores
        q = q.transpose(1, 2)  # [num_heads, batch_size, seq_len, head_dim]
        k = k.transpose(1, 2).transpose(2, 3)  # [num_heads, batch_size, head_dim, key_len]
        
        # Scaled dot-product attention
        # [num_heads, batch_size, seq_len, key_len]
        scores = torch.matmul(q, k) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
            
        # Apply padding mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        # Apply softmax and dropout
        attention = self.dropout(torch.softmax(scores, dim=-1))
        
        # Compute output
        v = v.transpose(1, 2)  # [num_heads, batch_size, key_len, head_dim]
        output = torch.matmul(attention, v)  # [num_heads, batch_size, seq_len, head_dim]
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).transpose(0, 1).contiguous().view(seq_len, batch_size, embed_dim)
        output = self.output_proj(output)
        
        return output