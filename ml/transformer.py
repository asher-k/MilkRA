import numpy as np
import torch
import torch.nn as nn
from functools import partial


class ViT(nn.Module):
    """
    Vision Transformer. Use SubdivTransform when loading from Dataset to ensure samples are divided into sub-regions.
    """
    def __init__(self, sd, n_dims, n_heads, n_blocks, n_classes):
        super(ViT, self).__init__()

        # Non-layer attributes
        self.sd = sd  # (Num Subdivisions, Subdiv dimensions)
        self.n_dims = n_dims  # Dimensionality of image embeddings
        self.n_heads = n_heads  # No. of Attention Heads
        self.n_blocks = n_blocks  # No. of Attention blocks
        self.n_classes = n_classes

        # Embeddings & Tokens
        self.linear_embedding = nn.Linear(self.sd[1], self.n_dims)
        self.positional_embedding = partial(_sinusoidal_positional_embeddings, n=self.n_dims)
        self.v_class = nn.Parameter(torch.rand(1, self.n_dims))  # classification token

        # Encoder
        self.attn_blocks = nn.ModuleList([AttentionBlock(self.n_dims, self.n_heads) for i in range(0, self.n_blocks)])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.n_dims, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Performs a forward pass of the ViT.

        :param x: Tensor data sample
        :return: Int predicted class, unused auxiliary parameter (changed to return mean attentions by layer)
        """
        # Compute embeddings of x and append our class token
        position_x = self.positional_embedding(x)
        x = self.linear_embedding(x)
        x = x + position_x
        x = torch.stack([torch.vstack((self.v_class, x[i])) for i in range(len(x))])  # stack classification tokens w/ x

        # Encoder
        attns = []
        for block in self.attn_blocks:
            x, _attn = block(x)
            attns.append(_attn)
        x = x[:, 0]  # Extract classification token

        # Classifier
        prediction = self.classifier(x)
        return prediction, attns


class AttentionBlock(nn.Module):
    """
    Attention Block; composed of layer normalizations, MHSA, residual connections and an MLP.
    """
    def __init__(self, n_dims, n_heads, dense_size=4):
        super(AttentionBlock, self).__init__()

        # Non-layer attributes
        self.n_dims = n_dims
        self.n_heads = n_heads
        self.dense_size = dense_size

        # Layers
        self.lnorm1 = nn.LayerNorm(self.n_dims)
        self.mhsa = MultiHeadSelfAttention(self.n_dims, self.n_heads)

        self.lnorm2 = nn.LayerNorm(self.n_dims)
        self.lin1 = nn.Linear(self.n_dims, self.n_dims * self.dense_size)  # TODO: update number of dims
        self.gelu = nn.GELU()
        self.lin2 = nn.Linear(self.n_dims * self.dense_size, self.n_dims)

    def forward(self, x):
        """
        Performs a forward pass through the attention block.

        :param x: Tensor data sample
        :return: Transformed sample
        """
        inner_x = self.lnorm1(x)
        inner_x, _attns = self.mhsa(inner_x)
        x_bp = inner_x + x  # Propagate original embeddings through residual connection, store as a breakpoint for later

        inner_x = self.lin1(self.lnorm2(x_bp))  # MLP
        inner_x = self.lin2(self.gelu(inner_x))
        x = inner_x + x_bp  # Second residual connection
        return x, _attns


class MultiHeadSelfAttention(nn.Module):
    """
    Custom implementation of a Multi-Headed self attention layer.
    """
    def __init__(self, n_dims, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.ndims = n_dims
        self.nheads = n_heads
        self.hdims = int(self.ndims / self.nheads)
        assert self.ndims % self.nheads == 0  # Otherwise unable to properly divide embedding dimensions into the heads

        self.q_vectors = nn.ModuleList([nn.Linear(self.hdims, self.hdims) for i in range(0, self.nheads)])
        self.k_vectors = nn.ModuleList([nn.Linear(self.hdims, self.hdims) for i in range(0, self.nheads)])
        self.v_vectors = nn.ModuleList([nn.Linear(self.hdims, self.hdims) for i in range(0, self.nheads)])
        self.sm = nn.Softmax(dim=-1)  # Perform SM on the innermost dimension (ie attentions)

    def forward(self, x):
        """
        Performs a forward pass through the attention layer.
        Input: (B, C, D)
        Output: (B, C, D)

        :param x: Tensor batch of data samples
        :return: Transformed sample
        """
        results = []
        attns = [[] for i in range(len(x))]
        for i, sample in enumerate(x):
            s_results = []
            for h in range(self.nheads):
                q, k, v = self.q_vectors[h], self.k_vectors[h], self.v_vectors[h]
                window = sample[:, h * self.hdims: (h+1) * self.hdims]  # Obtain embedding window for the hth head
                q, k, v = q(window), k(window), v(window)
                a = self.sm(q @ k.T / (self.hdims ** 0.5))  # Product of the queries & keys, plus scaling, masking & SM
                attns[i].append(a)
                a = a @ v  # Then obtain product with the values
                s_results.append(a)
            attns[i] = torch.mean(torch.stack([a for a in attns[i]]), dim=0)  # compute mean attention over heads
            results.append(torch.hstack(s_results))
        results = [torch.unsqueeze(r, dim=0) for r in results]  # Expand out the final dimension to match input shape
        return torch.cat(results), attns


def _sinusoidal_positional_embeddings(x, n=None):
    """
    Computes sinusoidal positional embeddings over a batch per Vaswani et al.

    :param x: Tensor Batch of subdivided samples to compute positional embeddings over; ensures consistency with other
    torch.nn layers
    :param n: Int dimensionality of the positional embeddings
    :return: n-Dimensionality embeddings for each sample in x
    """
    def inner_func(i, j, d=n):
        """
        Computes positional embedding at the provided timesteps and position.

        :param i: Int Timestep within the sequence
        :param j: Int Position with the embedding
        :param d: Int dimensionality of the embedding
        :return:
        """
        if i % 2 == 0:
            return np.sin(i / (10000 ** (j / d)))
        return np.cos(i / (10000 ** ((j-1) / d)))

    bs, n_subdivs = x.shape[0], x.shape[1]
    embs = torch.zeros((bs, n_subdivs, n))
    for eid, sample in enumerate(embs):
        for sid, _ in enumerate(sample):
            subdiv = torch.Tensor([inner_func(sid, j) for j in range(0, n)])
            embs[eid][sid] = subdiv
    return embs
