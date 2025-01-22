# Based on arxiv article "SAINT: Improved Neural Networks for Tabular Datavia Row Attention and Contrastive Pre-Training"
# by Gowthami Somepalli, Micah Goldblum, Avi Schwarzschild, C. Bayan Bruss and Tom Goldstein
# Link: https://arxiv.org/pdf/2106.01342

import torch
import torch.nn as nn

from ..deeplayers.embeddings import Abberation
from ..deeplayers.transformers import PositionwiseFeedForward, MultiHeadedAttention, EncoderLayer, Encoder, MultiHeadedIntersampleAttention

class SaintLayer(nn.Module):
    """
    Saint layer. Stacks the self attention and the intersample attention
    """
    def __init__(self, SelfAttn: nn.Module, IntersampleAttn: nn.Module, size: int):
        """
        Args:
            SelfAttn (nn.Module): Encoder layer of the self attention 
            IntersampleAttn (nn.Module): Encoder layer of intersample attention 
            size (int): Number of features, this is required by LayerNorm
        """
        super(SaintLayer, self).__init__()
        self.SelfAttn = SelfAttn                          # multi-head attention
        self.IntersampleAttn = IntersampleAttn            # multi-head interasample attention
        self.size = size

    def forward(self, x: torch.Tensor):
        x = self.SelfAttn(x)
        x = self.IntersampleAttn(x)
        return x

class SAINT():
    def __init__(
            self, 
            num_heads: int, 
            embed_dim: int, 
            num_layers: int, 
            feedforward_dim: int, 
            dropout: int, 
            feedforward_dropout: int
            ) -> None:
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.ff_dim = feedforward_dim
        self.dropout = dropout
        self.ff_dropout = feedforward_dropout

    def new(self):
        feedforward = PositionwiseFeedForward(d_model = self.embed_dim, d_ff = self.ff_dim, dropout = self.ff_dropout)
        self_attn = MultiHeadedAttention(h = self.num_heads, d_model = self.embed_dim)
        multihead_attn = EncoderLayer(size = self.embed_dim, self_attn = self_attn, feed_forward = feedforward, dropout = self.dropout)
        self_inter_attn = MultiHeadedIntersampleAttention(
            h = self.num_heads,
            d_model = self.embed_dim,
            dropout = self.dropout
        )
        intersample_attn = EncoderLayer(size = self.embed_dim, self_attn = self_inter_attn, feed_forward = feedforward, dropout = self.dropout)
        
        layer = SaintLayer(multihead_attn, intersample_attn, size = self.embed_dim)
        encoder = Encoder(layer = layer, N = self.num_layers)
        for p in encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return encoder