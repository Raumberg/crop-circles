import torch
import numpy as np

class Augmenter:
    """
    A class to perform various data augmentation techniques on tensor data.
    """

    def __init__(self):
        pass

    def batch_feat_shuffle(X: torch.Tensor, beta: float = 0.5):
        """
        Performs feature-wise mixup on the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_features) or (batch_size, num_features, num_dimensions).
            beta (float): The parameter for the beta distribution used to determine shuffle rates.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The augmented tensor after feature-wise mixup.
                - torch.Tensor: The feature masks used for mixing.
                - np.ndarray: The shuffled sample indices.
        """
        b, f = X.shape[0], X.shape[1]
        shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
        feat_masks = np.random.random(size=(b, f)) > shuffle_rates  # b f
        feat_masks = torch.from_numpy(feat_masks).to(X.device)

        shuffled_sample_ids = np.random.permutation(b)

        X_shuffled = X[shuffled_sample_ids]
        feat_masks = feat_masks.unsqueeze(-1) if X.ndim == 3 else feat_masks
        X_mixup = feat_masks * X + ~feat_masks * X_shuffled

        return X_mixup, feat_masks.squeeze(-1), shuffled_sample_ids

    def batch_dim_shuffle(X: torch.Tensor, beta: float = 0.5):
        """
        Performs dimension-wise mixup on the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_features, num_dimensions).
            beta (float): The parameter for the beta distribution used to determine shuffle rates.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The augmented tensor after dimension-wise mixup.
                - torch.Tensor: The shuffle rates used for mixing.
                - np.ndarray: The shuffled sample indices.
        """
        b, f, d = X.shape
        shuffle_rates = np.random.beta(beta, beta, size=(b, 1))
        dim_masks = np.random.random(size=(b, d)) < shuffle_rates  # b d
        dim_masks = torch.from_numpy(dim_masks).to(X.device)

        shuffled_sample_ids = np.random.permutation(b)

        X_shuffled = X[shuffled_sample_ids]
        dim_masks = dim_masks.unsqueeze(1)  # b 1 d
        X_mixup = dim_masks * X + ~dim_masks * X_shuffled

        return X_mixup, torch.from_numpy(shuffle_rates[:, 0]).float().to(X.device), shuffled_sample_ids

    @staticmethod
    def mixup_data(X: torch.Tensor, beta: float = 0.5):
        """
        Performs naive mixup on the input tensor.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, num_features).
            beta (float): The parameter for the beta distribution used to determine mixing ratio.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The augmented tensor after naive mixup.
                - float: The mixing ratio (lambda).
                - np.ndarray: The shuffled sample indices.
        """
        b, f = X.shape
        lam = np.random.beta(beta, beta)
        shuffle_sample_ids = np.random.permutation(b)
        mixed_X = lam * X + (1 - lam) * X[shuffle_sample_ids]

        return mixed_X, lam, shuffle_sample_ids
