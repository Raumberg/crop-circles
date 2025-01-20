import torch
import torch.nn as nn


# --- NORMS ---
# Source: https://github.com/bzhangGo/rmsnorm
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-5, bias=False):
        """Root Mean Square Layer Normalization

        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class ScaleNorm(nn.Module):
    """
    Sources:
    - https://github.com/tnq177/transformers_without_tears/blob/25026061979916afb193274438f7097945acf9bc/layers.py#L132
    - https://github.com/tnq177/transformers_without_tears/blob/6b2726cd9e6e642d976ae73b9f696d9d7ff4b395/layers.py#L157
    """

    def __init__(self, d: int, eps: float = 1e-5, clamp: bool = False) -> None:
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(d ** 0.5))
        self.eps = eps
        self.clamp = clamp

    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True)
        norms = norms.clamp(min=self.eps) if self.clamp else norms + self.eps
        return self.scale * x / norms

# --- NORMS ---