import sys, os
sys.path.append(os.getcwd())

import unittest

import torch
import torch.nn as nn

from source.model.basic_blocks import (
    TimeFrequencyConvolutionBlock,
    TimeDistributedSelfAttentionBlock,
    DownSample2DBlock,
    UpSample2DBlock,
)
from source.model.TFC_TDSA_UNet.building_blocks import (
    TFC_TDSA, TFC_TDSA_DownSample, TFC_TDSA_UpSample
)


class TestIntermediateBlock(unittest.TestCase):
    def test_init_is_nn_Module(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdf_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(tfc_tdf_block, nn.Module)
    
    def test_init_multihead_self_attention_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        num_attention_heads = 4
        activation = "ReLU"
        bias = False

        tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        )

        expected_tfc_block = TimeFrequencyConvolutionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias
        )

        expected_tdsa_block = TimeDistributedSelfAttentionBlock(
            embed_dim=frequency_bins,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdsa_block.tfc_block.__str__(), 
                         expected_tfc_block.__str__())
        self.assertEqual(tfc_tdsa_block.tdsa_block.__str__(), 
                         expected_tdsa_block.__str__())
    
    def test_init_vanilla_self_attention_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        num_attention_heads = 1
        use_vanilla_self_attention = True
        activation = "ReLU"
        bias = False

        tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        )

        expected_tfc_block = TimeFrequencyConvolutionBlock(
            in_channels=in_channels,
            growth_rate=growth_rate,
            num_layers=num_layers,
            kernel_size=kernel_size,
            activation=activation,
            bias=bias
        )

        expected_tdsa_block = TimeDistributedSelfAttentionBlock(
            embed_dim=frequency_bins,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdsa_block.tfc_block.__str__(), 
                         expected_tfc_block.__str__())
        self.assertEqual(tfc_tdsa_block.tdsa_block.__str__(), 
                         expected_tdsa_block.__str__())

    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        num_attention_heads = 4
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        ).to(device)

        x = torch.rand([1, in_channels, 44, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdsa_block(x)
        expected_shape = [1, growth_rate, 44, frequency_bins]

        self.assertEqual(list(output.shape), expected_shape)

    def test_vanilla_self_attention_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        num_attention_heads = 1
        use_vanilla_self_attention = True
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        ).to(device)

        x = torch.rand([1, in_channels, 44, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdsa_block(x)
        expected_shape = [1, growth_rate, 44, frequency_bins]

        self.assertEqual(list(output.shape), expected_shape)


class TestDownSampleBlock(unittest.TestCase):
    def test_init_is_nn_Module(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdsa_downsample_block = TFC_TDSA_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(tfc_tdsa_downsample_block, nn.Module)
    
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdsa_downsample_block = TFC_TDSA_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
        )

        expected_tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        )

        expected_downsample_block = DownSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdsa_downsample_block.tfc_tdsa_block.__str__(), 
                         expected_tfc_tdsa_block.__str__())
        
        self.assertEqual(tfc_tdsa_downsample_block.downsample_block.__str__(), 
                         expected_downsample_block.__str__())


    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdsa_downsample_block = TFC_TDSA_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        ).to(device)
        
        x = torch.rand([1, in_channels, 44, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdsa_downsample_block(x)
        expected_shape = [1, growth_rate, 22, frequency_bins // 2]

        self.assertEqual(list(output.shape), expected_shape)

    def test_vanilla_self_attention_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 1
        use_vanilla_self_attention = True
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdsa_downsample_block = TFC_TDSA_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        ).to(device)
        
        x = torch.rand([1, in_channels, 44, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdsa_downsample_block(x)
        expected_shape = [1, growth_rate, 22, frequency_bins // 2]

        self.assertEqual(list(output.shape), expected_shape)


class TestUpSampleBlock(unittest.TestCase):
    def test_init_is_nn_Module(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdsa_upsample_block = TFC_TDSA_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(tfc_tdsa_upsample_block, nn.Module)
    
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdsa_upsample_block = TFC_TDSA_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
        )

        expected_tfc_tdsa_block = TFC_TDSA(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        )

        expected_upsample_block = UpSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdsa_upsample_block.tfc_tdsa_block.__str__(), 
                         expected_tfc_tdsa_block.__str__())
        
        self.assertEqual(tfc_tdsa_upsample_block.upsample_block.__str__(), 
                         expected_upsample_block.__str__())


    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 4
        frequency_bins = 512
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdsa_downsample_block = TFC_TDSA_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            activation=activation,
            bias=bias
        ).to(device)
        
        x = torch.rand([1, in_channels, 22, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdsa_downsample_block(x)
        expected_shape = [1, growth_rate, 44, frequency_bins * 2]

        self.assertEqual(list(output.shape), expected_shape)

    def test_vanilla_attention_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        num_attention_heads = 1
        frequency_bins = 512
        activation = "ReLU"
        use_vanilla_self_attention = True
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdsa_downsample_block = TFC_TDSA_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            num_attention_heads=num_attention_heads,
            use_vanilla_self_attention=use_vanilla_self_attention,
            activation=activation,
            bias=bias
        ).to(device)
        
        x = torch.rand([1, in_channels, 22, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdsa_downsample_block(x)
        expected_shape = [1, growth_rate, 44, frequency_bins * 2]

        self.assertEqual(list(output.shape), expected_shape)

if __name__ == "__main__":
    # suite = unittest.TestSuite()

    # suite.addTest(TestDownSampleBlock("test_forward_output_dims"))

    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    unittest.main()