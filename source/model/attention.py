from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(seq_length: int, embed_dim: int) -> torch.Tensor:
    """
    Positional encoding as described in "Attention is all you need" (https://arxiv.org/pdf/1706.03762.pdf)
    [seq_length, embed_dim]
    """
    pe = torch.zeros(seq_length, embed_dim)
    
    pos = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(1)
    div = (10000 ** (2 * torch.arange(1, embed_dim, 2, dtype=torch.float32) / embed_dim))
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)

    return pe

class VanillaSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super(VanillaSelfAttention, self).__init__()
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_dim, dtype=x.dtype))

        attention_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, v)

        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 seq_length: int = 1024,
                 dropout: float = 0.2,):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim

        self.positional_encoding = positional_encoding(seq_length=seq_length, embed_dim=embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention = VanillaSelfAttention(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        [batch, seq_length, embed_dim] -> [batch, seq_length, embed_dim]
        """
        batch, seq_length, embed_dim = x.shape
        print(x.shape)
        x = x + self.positional_encoding[:seq_length, :].repeat(batch, 1, 1)
        x = x + self.attention(x)
        x = self.layer_norm(x)
        x = x + self.feed_forward(x)
        return x
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = torch.rand([1, 352, 1024])
    pe = positional_encoding(512, 512)
    # pe = positional_encoding_2(1024, 1)
    print(pe.shape)
    print(pe)
    plt.pcolormesh(pe)
    plt.colorbar()
    plt.ylabel("Sequence Length")
    plt.xlabel("Embedding Dimension")
    plt.show()

    plt.close()