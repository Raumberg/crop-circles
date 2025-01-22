import math
import torch

from models.entmax import EntmaxAlpha, Sparsemax
from einops import rearrange, repeat

import torch
import random
import numpy as np

class GSH(torch.nn.Module):
    """
    Generalized Sparse Hopfield module.

    This module implements a generalized sparse Hopfield mechanism for attention.

    Parameters
    ----------
    scale : float, optional
        Parameter for scaling attention scores in the attention mechanism.
    dropout : float, optional
        Dropout ratio (default is 0.1).
    actv : str, optional
        Activation function to use ('softmax', 'sparsemax', or 'entmax'; default is 'sparsemax').
    """
    def __init__(self, scale=None, dropout=0.1, actv='sparsemax'):
        super(GSH, self).__init__()
        self.scale = scale
        self.dropout = torch.nn.Dropout(dropout)

        if actv == 'softmax':
            self.softmax = torch.nn.Softmax(dim=-1)
        elif actv == 'sparsemax':
            self.softmax = Sparsemax()
        else:
            self.softmax = EntmaxAlpha()
  
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        self.scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(self.softmax(self.scale * scores))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()
  

class GSHLayer(torch.nn.Module):
    """
    Generalized Sparse Hopfield (GSH) layer.

    This layer implements the GSH mechanism for multi-head self-attention.

    Parameters
    ----------
    d_model : int
        Number of hidden features.
    n_heads : int
        Number of heads.
    d_keys : int, optional
        Dimensionality of the projected queries and keys in the multi-head self-attention mechanism.
    d_values : int, optional
        Dimensionality of the projected values for each head in the multi-head self-attention layer.
    mix : bool, optional
        Whether to mix the outputs in the model's forward pass (default is True).
    dropout : float, optional
        Dropout ratio (default is 0.1).
    hopfield : bool, optional
        True if using Hopfield attention, False if using classical Transformer attention (default is True).
    """
    def __init__(self, d_model=512, n_heads=8, d_keys=None, d_values=None, mix=True, dropout=0.1, actv='entmax', hopfield=True):
        super(GSHLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.hopfield = hopfield

        self.inner_attention = GSH(scale=None, dropout=dropout, actv=actv)
        self.query_projection = torch.nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = torch.nn.Linear(d_model, d_keys * n_heads)

        if self.hopfield:
            self.value_projection = torch.nn.Linear(d_keys * n_heads, d_keys * n_heads)
        else:
            self.value_projection = torch.nn.Linear(d_model, d_values * n_heads)

        self.out_projection = torch.nn.Linear(d_values * n_heads, d_model)
        self.norm = torch.nn.SELU()
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)

        if self.hopfield:
            values = self.key_projection(values)
            values = self.value_projection(values).view(B, S, H, -1)
            values = values.view(B, S, H, -1)
        else:
            values = self.value_projection(values).view(B, S, H, -1)
            values = values.view(B, S, H, -1) 

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)
  

class BAModule (torch.nn.Module):
    """
    BAModule for attention-based feature aggregation.

    This module aggregates features using attention mechanisms and pooling.

    Parameters
    ----------
    n_pool : int, optional
        Number of pooling operations (default is 10).
    factor : int, optional
        Factor for pooling (default is 8).
    d_model : int, optional
        Number of hidden features (default is 512).
    n_heads : int, optional
        Number of heads (default is 8).
    d_ff : int, optional
        Dimensionality of the feed-forward layer (default is 4 times d_model).
    dropout : float, optional
        Dropout ratio (default is 0.1).
    actv : str, optional
        Activation function to use ('entmax' is default).
    hopfield : bool, optional
        True if using Hopfield attention, False if using classical Transformer attention (default is True).
    
    Input
    ------
    x : torch.Tensor
        Patched embedded tabular data of shape (batch size, embedding dimension, number of patches, d_model).

    Output
    ------
    torch.Tensor
        Patched embedded tabular data of shape (batch size, emb_dim, n_pool, d_model).
    """
    def __init__(self, n_pool=10, factor=8, d_model=512, n_heads=8, d_ff=None, dropout=0.1, actv='entmax', hopfield=True):
        super(BAModule, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.feat_attn = GSHLayer(d_model, n_heads, dropout=dropout, actv=actv, hopfield=hopfield)
        self.emb_pooling = GSHLayer(d_model, n_heads, dropout=dropout, actv=actv, hopfield=hopfield)
        self.emb_attn = GSHLayer(d_model, n_heads, dropout=dropout, actv=actv, hopfield=hopfield)
        self.pooling = torch.nn.Parameter(torch.randn(n_pool, factor, d_model))
        self.dropout = torch.nn.Dropout(dropout)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.norm4 = torch.nn.LayerNorm(d_model)

        self.MLP1 = torch.nn.Sequential(torch.nn.Linear(d_model, d_ff), torch.nn.GELU(),
                                        torch.nn.Linear(d_ff, d_model))
        self.MLP2 = torch.nn.Sequential(torch.nn.Linear(d_model, d_ff),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(d_ff, d_model))

    def forward(self, x):
        # Cross Feature: Directly apply MSA to each feature patch
        batch = x.shape[0]
        feat_in = rearrange(x, 'b emb_dim n_pool d_model -> (b emb_dim) n_pool d_model')
        feat_enc = self.feat_attn(
            feat_in, feat_in, feat_in
        )
        feat_out = feat_in + self.dropout(feat_enc)
        feat_out = self.norm1(feat_out)
        feat_out = feat_out + self.dropout(self.MLP1(feat_out))
        emb_in = self.norm2(feat_out)

        # Cross Embedding: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        emb_send = rearrange(emb_in, '(b emb_dim) n_pool d_model -> (b n_pool) emb_dim d_model', b=batch)
        pooling_send = repeat(self.pooling, 'n_pool factor d_model -> (repeat n_pool) factor d_model', repeat=batch)
        emb_buffer = self.emb_pooling(pooling_send, emb_send, emb_send)
        emb_receive = self.emb_attn(emb_send, emb_buffer, emb_buffer)
        emb_out = emb_send + self.dropout(emb_receive)
        emb_out = self.norm3(emb_out)
        emb_out = emb_out + self.dropout(self.MLP2(emb_out))
        emb_out = self.norm4(emb_out)

        final_out = rearrange(emb_out, '(b n_pool) emb_dim d_model -> b emb_dim n_pool d_model', b=batch)
        return final_out