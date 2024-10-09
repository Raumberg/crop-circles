import torch
import torch.nn as nn


class CategoricalEmbedding(nn.Module):
    """
    Embedding of categorical features using NN embedding layer.
    N embeddings will be created for N categorical feature
    """

    def __init__(self, no_cat, embed_dim, feature=None):
        """
        Args:
            no_cat (int): number of categories for the feature
            embed_dim (int): dimension of embedding
            feature (str): name of categorical feature vector
        """
        super(CategoricalEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            no_cat,
            embed_dim
        )
        self.feature = feature

    def forward(self, x):
        return self.embedding(x)


class NumericalEmbedding(nn.Module):
    """
    Embedding class of each Numerical feature using NN Linear \
    layer of size (1 x embed_dim) and then Relu non-linearity. 
    N embeddings will be created for N numerical features
    """

    def __init__(self, embed_dim, feature=None):
        """
        Args:
            embed_dim (int): dimension of embedding
            feature (str): name of numerical feature vector
        """
        super(NumericalEmbedding, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU())
        self.feature = feature

    def forward(self, x):
        return self.linear(x) 


class Abberation(nn.Module):
    """
    Abberation is a distortion of data (preliminary, tabular) to a combination of one with categorical and numerical vector fields
    """

    def __init__(self, embed_dim, no_num, no_cat, cats):
        super(Abberation, self).__init__()
        assert no_cat == len(cats)

        self.embed_dim = embed_dim

        self.cat_embedding = nn.ModuleList()
        for cat in cats:
            self.cat_embedding.append(
                CategoricalEmbedding(cat, embed_dim)
            )

        self.num_embedding = nn.ModuleList()
        for i in range(no_num):
            self.num_embedding.append(
                NumericalEmbedding(embed_dim)
            )

        self.no_num = no_num
        self.no_cat = no_cat

    def forward(self, x):
        batchsize = x.shape[0]

        output = []

        for i, layer in enumerate(self.cat_embedding):
            output.append(layer(x[:, i].long()))

        for i, layer in enumerate(self.num_embedding):
            output.append(layer(x[:, self.no_cat + i].unsqueeze(1).float()))

        data = torch.stack(output, dim=1)  # -> [batch_size, N, embedding_size]

        return data
