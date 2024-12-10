import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import deeplib as lib
from deeplib import deepmodules as dm
from deeplib import augmentations as aug

def attenuated_kaiming_uniform_(tensor, a=math.sqrt(5), scale=1., mode='fan_in', nonlinearity='leaky_relu'):
    """
    Initializes a tensor using an attenuated Kaiming uniform distribution.

    This initialization method is designed to keep the scale of the gradients
    approximately the same in all layers, which helps in training deep networks.

    Args:
        tensor (Tensor): The tensor to initialize.
        a (float): The negative slope of the rectifier used after this layer (default: sqrt(5)).
        scale (float): Scaling factor for the standard deviation (default: 1.0).
        mode (str): Either 'fan_in' (default) or 'fan_out'.
        nonlinearity (str): The non-linear function (default: 'leaky_relu').

    Returns:
        Tensor: The initialized tensor.
    """
    fan = nn_init._calculate_correct_fan(tensor, mode)
    gain = nn_init.calculate_gain(nonlinearity, a)
    std = gain * scale / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class Tokenizer(nn.Module):
    """
    Tokenizer module for converting numerical and categorical features into token embeddings.

    Attributes:
        category_offsets (Tensor): Offsets for categorical features.
        category_embeddings (nn.Embedding): Embedding layer for categorical features.
        weight (nn.Parameter): Weight parameter for numerical features.
        weight2 (nn.Parameter): Second weight parameter for numerical features.
        bias (nn.Parameter): Bias parameter.
        bias2 (nn.Parameter): Second bias parameter.
    """

    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.weight2 = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        self.bias2 = nn.Parameter(Tensor(d_bias, d_token)) if bias else None

        attenuated_kaiming_uniform_(self.weight)
        attenuated_kaiming_uniform_(self.weight2)
        nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        nn_init.kaiming_uniform_(self.bias2, a=math.sqrt(5))


    @property
    def n_tokens(self) -> int:
        """
        Returns the number of tokens.

        Returns:
            int: The number of tokens.
        """
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor) -> Tensor:
        """
        Forward pass through the Tokenizer.

        Args:
            x_num (Tensor): Input tensor of numerical features.

        Returns:
            Tensor: Token embeddings.
        """
        assert x_num is not None
        if isinstance(x_num, list):
            x_num = x_num[0].float()
        x1 = self.weight[None] * x_num[:, :, None] + self.bias[None]
        x2 = self.weight2[None] * x_num[:, :, None] + self.bias2[None]
        # print(x2)
        # print(x2.shape)
        return x1 * torch.tanh(x2)



class MultiheadAttention(nn.Module):
    """
    Multihead attention module.

    Attributes:
        W_q (nn.Linear): Query projection layer.
        W_k (nn.Linear): Key projection layer.
        W_v (nn.Linear): Value projection layer.
        W_out (nn.Linear): Output projection layer (optional).
        n_heads (int): Number of attention heads.
        dropout (nn.Dropout): Dropout layer (optional).
    """
    def __init__(
        self, d: int, n_heads: int, dropout: float, init_scale: float = 0.01
    ) -> None:
        """
        Initializes the MultiheadAttention module.

        Args:
            d (int): Dimension of input features.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            init_scale (float): Initialization scale (default: 0.01).
        """
        if n_heads > 1:
            assert d % n_heads == 0
        # assert initialization in ['xavier', 'kaiming']
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for i, m in enumerate([self.W_q, self.W_k, self.W_v]):
            attenuated_kaiming_uniform_(m.weight, scale=init_scale)
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            attenuated_kaiming_uniform_(self.W_out.weight)
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        """
        Reshapes the input tensor for multihead attention.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reshaped tensor.
        """
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )
    
    def get_attention_mask(self, input_shape, device):
        """
        Generates an attention mask for the input tensor.

        Args:
            input_shape (tuple): Shape of the input tensor.
            device (torch.device): Device on which the tensor is located.

        Returns:
            Tensor: Attention mask.
        """
        bs, _, seq_len = input_shape
        seq_ids = torch.arange(seq_len, device=device)
        attention_mask = seq_ids[None, None, :].repeat(bs, seq_len, 1) <= seq_ids[None, :, None]
        # attention_mask = seq_ids[None, :].repeat(seq_len, 1) <= seq_ids[:, None]
        attention_mask = (1.0 - attention_mask.float()) * -1e4
        return attention_mask

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        """
        Forward pass through the MultiheadAttention module.

        Args:
            x_q (Tensor): Query tensor.
            x_kv (Tensor): Key-Value tensor.
            key_compression (nn.Linear): Key compression layer (optional).
            value_compression (nn.Linear): Value compression layer (optional).

        Returns:
            Tensor: Attention output.
        """
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_scores = q @ k.transpose(1, 2) / math.sqrt(d_head_key) # b f f
        masks = self.get_attention_mask(attention_scores.shape, attention_scores.device)
        attention = F.softmax(attention_scores + masks, dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class RiegelRing(nn.Module):
    """
    Riegel Ring module with all parameters initialized by kaiming uniform.

    Attributes:
        tokenizer (Tokenizer): Tokenizer module.
        n_categories (int): Number of categories.
        shared_kv_compression (nn.Linear): Shared key-value compression layer (optional).
        layers (nn.ModuleList): List of transformer layers.
        activation (nn.Module): Activation function.
        last_activation (nn.Module): Last activation function.
        prenormalization (bool): Whether to use pre-normalization.
        last_normalization (nn.Module): Last normalization layer (optional).
        ffn_dropout (float): Dropout rate for feed-forward network.
        residual_dropout (float): Dropout rate for residual connections.
        head (nn.Linear): Output linear layer.
        last_fc (nn.Linear): Last fully connected layer.
    """
    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        prenormalization: bool,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
        init_scale: float = 0.1,
    ) -> None:
        """
        Initializes the RiegelRing module.

        Args:
            d_numerical (int): Dimension of numerical features.
            categories (List[int]): List of categorical feature sizes.
            token_bias (bool): Whether to use bias in the tokenizer.
            n_layers (int): Number of transformer layers.
            d_token (int): Dimension of token embeddings.
            n_heads (int): Number of attention heads.
            attention_dropout (float): Dropout rate for attention layers.
            ffn_dropout (float): Dropout rate for feed-forward networks.
            residual_dropout (float): Dropout rate for residual connections.
            prenormalization (bool): Whether to use pre-normalization.
            kv_compression (float): Compression ratio for key-value pairs (optional).
            kv_compression_sharing (str): Type of key-value compression sharing (optional).
            d_out (int): Dimension of the output layer.
            init_scale (float): Initialization scale (default: 0.1).
        """
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        # assert initialization in ['xavier', 'kaiming']
        n_tokens = d_numerical + len(categories) if categories is not None else d_numerical
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        self.n_categories = 0 if categories is None else len(categories)

        def make_kv_compression():
            """
            Creates a key-value compression layer.

            Returns:
                nn.Linear: Key-value compression layer.
            """
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            """
            Creates a layer normalization module.

            Returns:
                nn.LayerNorm: Layer normalization module.
            """
            return nn.LayerNorm(d_token)

        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    # Attenuated Initialization
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, init_scale=init_scale
                    ),
                    'linear0': nn.Linear(d_token, d_token * 2),
                    'norm1': make_normalization(),
                }
            )
            # Attenuated Initialization
            attenuated_kaiming_uniform_(layer['linear0'].weight, scale=init_scale)
            nn_init.zeros_(layer['linear0'].bias)

            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = dm.get_activation_fn('tanglu')
        self.last_activation = nn.PReLU()
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        # Attenuated Initialization
        self.head = nn.Linear(d_token, d_out)
        attenuated_kaiming_uniform_(self.head.weight)
        # nn_init.zeros_(self.head.bias)
        self.last_fc = nn.Linear(n_tokens, 1) # b f d -> b 1 d eintransform
        # self.last_fc = nn.Linear(d_numerical, 1)
        attenuated_kaiming_uniform_(self.last_fc.weight)
        # nn_init.zeros_(self.last_fc.bias)
        

    def _get_kv_compressions(self, layer):
        """
        Determine the key and value compression modules for a given layer.

        This method checks if a shared key-value compression is set for the entire
        model. If not, it falls back to the individual key and value compressions
        defined in the layer configuration.

        Args:
            layer (dict): A dictionary representing the configuration of a layer.

        Returns:
            tuple: A tuple containing the key and value compression modules.
        """
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        """
        Prepare the input for a residual connection, optionally applying normalization.

        If prenormalization is enabled, this method applies the corresponding
        normalization layer to the input.

        Args:
            x (Tensor): The input tensor.
            layer (dict): A dictionary representing the configuration of a layer.
            norm_idx (int): The index of the normalization layer to use.

        Returns:
            Tensor: The potentially normalized input tensor for the residual connection.
        """
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        """
        Complete a residual connection by adding the residual and optionally applying normalization.

        This method adds the residual to the input and applies dropout if specified.
        If prenormalization is not enabled, it applies the corresponding normalization
        layer to the output.

        Args:
            x (Tensor): The original input tensor.
            x_residual (Tensor): The residual tensor to be added.
            layer (dict): A dictionary representing the configuration of a layer.
            norm_idx (int): The index of the normalization layer to use.

        Returns:
            Tensor: The output tensor after completing the residual connection.
        """
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x
    
    def forward(self, x_num: Tensor, mixup: bool=False, beta=0.5, mtype='feat_mix') -> Tensor:
        """
        Forward pass through the Riegel Ring, with optional mixup augmentation.

        This method processes the input through a series of layers, applying
        attention, activation functions, and residual connections. It supports
        various mixup techniques for data augmentation.

        Args:
            x_num (Tensor): The input tensor, which can be a single tensor or a list of tensors.
            mixup (bool, optional): Whether to apply mixup augmentation. Defaults to False.
            beta (float, optional): The beta parameter for the mixup distribution. Defaults to 0.5.
            mtype (str, optional): The type of mixup to apply ('naive', 'feat_mix', or 'hidden_mix'). Defaults to 'feat'.

        Returns:
            Tensor: The output tensor after processing through the network. If mixup is applied, 
                    also returns feature masks and shuffled IDs for the mixup.
        """
        if mtype == 'naive': # naive mixup
            x_num, feat_masks, shuffled_ids = aug.mixup_data(x_num, beta=beta)

        if isinstance(x_num, list):
            x = x_num[0].float()
        else:
            x = x_num.float()

        x = self.tokenizer(x)

        if mixup and mtype != 'naive':
            mixup_func = {
                'feat_mix': aug.batch_feat_shuffle,
                'hidden_mix': aug.batch_dim_shuffle,
            }[mtype]
            x, feat_masks, shuffled_ids = mixup_func(x, beta=beta)

        for layer_idx, layer in enumerate(self.layers):
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)  # Residual expects [batch_size, seq_len, d_token]
            x_residual = layer['attention'](                # Attention expects [batch_size, seq_len, d_token] 
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            x = self._end_residual(x, x_residual, layer, 0)
            # reglu
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        x = self.last_fc(x.transpose(1,2))[:,:,0] # b f d -> b d eintransform
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        if mixup:
            return x, feat_masks, shuffled_ids
        return x

    def _needs_wd(name):
        return all(x not in name for x in ['tokenizer', '.norm', '.bias'])
    
    def setup(
            self
        ) -> None:
        """
        On review!
        """
        parameters_with_wd = [v for k, v in self.named_parameters() if self._needs_wd(k)]
        parameters_without_wd = [v for k, v in self.named_parameters() if not self._needs_wd(k)]
        params = [
                {'params': parameters_with_wd},
                {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(params=params, lr=0.001)
        return self
    
    def inject(
            self,
            loader: torch.utils.data.DataLoader,
            epochs: int = 50,
        ) -> None:
        """
        On review! 
        """
        if not hasattr(self, 'optimizer'):
            self.setup()
        for epoch in range(1, epochs):
            self.train()
            pbar = Progbar(target=len(loader), width=50)
            for iteration, batch in enumerate(loader):
                inputs, targets = batch
                targets = targets.float().squeeze()
                self.optimizer.zero_grad()
                loss = self.loss_fn(self(batch), targets)
                loss.backward()
                self.optimizer.step()
                pbar.update(iteration + 1, [("Loss", loss.item())])
