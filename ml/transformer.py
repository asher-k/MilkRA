import torch.nn as nn
from functools import partial


class ViT(nn.Module):
    """
    Vision Transformer.
    """
    def __init__(self, chw, n_dims, n_heads):
        super(ViT, self).__init__()

        # Non-layer attributes
        self.chw = chw  # for ViT use SubdivTransform in data.py to obtain true chw
        self.n_dims = n_dims  # Dimensionality of image embeddings
        self.n_heads = n_heads  # No. of Attention Heads

        # Embeddings
        self.linear_embedding = nn.Linear(0, self.n_dims)
        self.positional_embedding = partial(_sinusoidal_positional_embeddings, n=self.n_dims)

    def forward(self, x):
        """
        Performs a forward pass of the ViT.

        :param x: Tensor data sample
        :return: Int predicted class
        """
        # Compute embeddings of x
        embedded_x = self.linear_embedding(x)
        position_x = self.positional_embedding(x)
        x = embedded_x + position_x

        # Encoder; MHA


def _sinusoidal_positional_embeddings(n, p):
    """
    Computes sinusoidal positional embeddings per Vaswani et al.

    :param n: Int dimensionality of the positional embeddings
    :param p: Int index to obtain the embedding for
    :return: n-Dimensionality embedding at position p
    """

    return p
