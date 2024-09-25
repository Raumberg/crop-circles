import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange

from typing import callable

class Temporality(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class GEGLU(nn.Module):
    """
    Gated GELU
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)
    
class Abberation(nn.Module):
    def __init__(
        self,
        dim,
        n_streams = 8,
        dim_stream = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_stream * n_streams
        self.n_streams = n_streams
        self.scale = dim_stream ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.n_streams
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'bn(hd)->bhnd', h=h), (q, k, v))
        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'bhnd->bn(hd)', h = h)
        return self.to_out(out)
    
class Perceptron(nn.Module):
    def __init__(self, dims):
        super(Perceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

class Riegel(nn.Module):
    def __init__(
            self, 
            num_tokens, 
            dim, 
            n_feats, 
            depth, 
            n_streams, 
            dim_stream, 
            attn_dropout, 
            ff_dropout,
            style='col'
        ):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(n_feats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Temporality(Abberation(
                        dim, 
                        n_streams=n_streams, 
                        dim_stream=dim_stream, 
                        dropout=attn_dropout))),
                    PreNorm(dim, Temporality(FeedForward(
                        dim, 
                        dropout=ff_dropout))),
                    PreNorm(dim*n_feats, Temporality(Abberation(
                        dim*n_feats, 
                        n_streams=n_streams, 
                        dim_stream=64, 
                        dropout=attn_dropout))),
                    PreNorm(dim*n_feats, Temporality(FeedForward(
                        dim*n_feats, 
                        dropout=ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*n_feats, Temporality(Abberation(
                        dim*n_feats, 
                        n_streams=n_streams, 
                        dim_stream=64, 
                        dropout=attn_dropout))),
                    PreNorm(dim*n_feats, Temporality(FeedForward(
                        dim*n_feats, 
                        dropout = ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, mask = None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers: 
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'bnd->1b(nd)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1b(nb)->bnd', n = n)
        else:
             for attn1, ff1 in self.layers:
                x = rearrange(x, 'bnd->1b(nd)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1b(nd)->bnd', n = n)
        return x
