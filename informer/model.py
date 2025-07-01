import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Ensure sample_k is at least 1 and not larger than L_K
        sample_k = max(1, min(sample_k, L_K))
        n_top = max(1, min(n_top, L_Q))

        if sample_k >= L_K:
            # If we're sampling all keys, just use full attention
            Q_K = torch.matmul(Q, K.transpose(-2, -1))
            M_top = torch.arange(n_top).unsqueeze(0).unsqueeze(0).expand(B, H, -1).to(Q.device)
            return Q_K[:, :, :n_top, :], M_top

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)

        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Compute M measure
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), sample_k)
        
        # Get top queries
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            # For causal masking, use cumulative sum
            if L_Q == L_V:
                context = V.cumsum(dim=-2)
            else:
                # If lengths don't match, use mean
                V_sum = V.mean(dim=-2)
                context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        
        # Ensure proper indexing with device placement
        batch_idx = torch.arange(B, device=V.device)[:, None, None]
        head_idx = torch.arange(H, device=V.device)[None, :, None]
        
        context_in[batch_idx, head_idx, index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V], device=V.device)/L_V).type_as(attn)
            attns[batch_idx, head_idx, index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # Ensure we have reasonable values for U_part and u
        U_part = max(1, min(self.factor * int(np.ceil(np.log(L_K)).item()), L_K))
        u = max(1, min(self.factor * int(np.ceil(np.log(L_Q)).item()), L_Q))
        
        # If sequence is too short, fall back to standard attention
        if L_K <= self.factor or L_Q <= self.factor:
            # Standard attention for short sequences
            scale = self.scale or 1./math.sqrt(D)
            scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            context = torch.matmul(attn, values)
            return context.transpose(2,1).contiguous(), attn
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        scale = self.scale or 1./math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # Create causal mask
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        
        # Ensure index is on the right device and has proper shape
        if index.device != device:
            index = index.to(device)
            
        indicator = _mask_ex[torch.arange(B, device=device)[:, None, None],
                             torch.arange(H, device=device)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask


class Informer(nn.Module):
    def __init__(self, opt, input_dim):
        super(Informer, self).__init__()
        self.opt = opt
        
        # Feature embedding
        self.embedding = nn.Linear(input_dim, opt.embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(opt.embedding_dim)
        
        # Encoder
        # Use mask_flag=False for classification tasks (no causal masking needed)
        Attn = ProbAttention(False, opt.factor, attention_dropout=opt.dropout, output_attention=False)
        
        encoder_layers = []
        conv_layers = []
        
        for l in range(opt.num_encoder_layers):
            encoder_layers.append(
                EncoderLayer(
                    AttentionLayer(Attn, opt.embedding_dim, opt.num_heads, mix=False),
                    opt.embedding_dim,
                    opt.embedding_dim * 4,
                    dropout=opt.dropout,
                    activation='gelu'
                )
            )
            
            # Add conv layers for distilling (except for the last layer)
            if opt.distil and l < opt.num_encoder_layers - 1:
                conv_layers.append(ConvLayer(opt.embedding_dim))
        
        self.encoder = Encoder(
            encoder_layers,
            conv_layers if opt.distil else None,
            norm_layer=torch.nn.LayerNorm(opt.embedding_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(opt.dropout),
            nn.Linear(opt.embedding_dim, opt.num_classes)
        )
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        
        # Embedding
        x = self.embedding(x)
        
        # Add positional encoding
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embedding_dim]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]
        
        # Encoder
        enc_out, _ = self.encoder(x, attn_mask=None)
        
        # Classification
        enc_out = enc_out.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len]
        output = self.classifier(enc_out)
        
        return output