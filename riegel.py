import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import math

class Abstract(nn.Module):
    def __init__(self, cats: int, dim: int, lr: float = 0.01, use_attention: bool = False, att_dim: int = 0):
        super(Abstract, self).__init__()
        self.embedding = nn.Embedding(cats, dim)
        self.att_dim = att_dim
        if use_attention:
            self.attention = AbstractAttention(dim, attention_dim=att_dim)

    def forward(self, x: Tensor):
        x = self.embedding(x)
        if self.use_attention:
            x = self.attention

    def extract(self):
        return self.embedding.weight.detach().numpy()
    
class AbstractAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super(AbstractAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, attention_dim)
        self.key_linear = nn.Linear(input_dim, attention_dim)
        self.value_linear = nn.Linear(input_dim, attention_dim)

    def forward(self, x: Tensor):
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        attention_scores = torch.matmul(query, key.T) / math.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=1)

        output = attention_weights * value
        return output.sum(dim=1)
