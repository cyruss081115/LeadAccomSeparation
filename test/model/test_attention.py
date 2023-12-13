import sys, os
sys.path.append(os.getcwd())

import unittest
import torch
import torch.nn as nn

from source.model.attention import (
    VanillaSelfAttention,
    TransformerBlock,
    positional_encoding
)


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self):
        seq_length = 10
        embed_dim = 64

        pe = positional_encoding(seq_length=seq_length, embed_dim=embed_dim)

        self.assertIsInstance(pe, torch.Tensor)
        self.assertEqual(pe.shape, (seq_length, embed_dim))

class TestVanillaSelfAttention(unittest.TestCase):
    def test_init_is_nn_Module(self):
        embed_dim = 64

        attention = VanillaSelfAttention(embed_dim=embed_dim)

        self.assertIsInstance(attention, nn.Module)
    
    def test_forward(self):
        embed_dim = 64
        batch_size = 8
        seq_length = 10

        attention = VanillaSelfAttention(embed_dim=embed_dim)

        x = torch.randn(batch_size, seq_length, embed_dim)

        out = attention(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (batch_size, seq_length, embed_dim))


class TestTransformerBlock(unittest.TestCase):
    def test_init_is_nn_Module(self):
        embed_dim = 64
        dropout = 0.2
        max_seq_length = 10

        block = TransformerBlock(
            embed_dim=embed_dim,
            dropout=dropout,
            seq_length=max_seq_length,
        )

        self.assertIsInstance(block, nn.Module)
    
    def test_forward_output_dims(self):
        embed_dim = 64
        dropout = 0.2
        max_seq_length = 10

        block = TransformerBlock(
            embed_dim=embed_dim,
            dropout=0.2,
            seq_length=max_seq_length,
        )

        batch_size = 8
        seq_length = 10

        x = torch.randn(batch_size, seq_length, embed_dim)

        out = block(x)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (batch_size, seq_length, embed_dim))



if __name__ == "__main__":
    unittest.main()