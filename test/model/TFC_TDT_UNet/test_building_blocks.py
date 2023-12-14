import sys, os
sys.path.append(os.getcwd())

import unittest

import torch
import torch.nn as nn

from source.model.basic_blocks import (
    TimeFrequencyConvolutionBlock,
    TimeDistributedTransformerBlock,
    DownSample2DBlock,
    UpSample2DBlock,
)
from source.model.unet_blocks import UNetBlock, UNetDownSampleBlock, UNetUpSampleBlock
from source.model.TFC_TDT_UNet.building_blocks import (
    TFC_TDT, TFC_TDT_DownSample, TFC_TDT_UpSample
)


class TestIntermediateBlock(unittest.TestCase):
    def test_init_is_nn_Module(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdt_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(tfc_tdt_block, nn.Module)
    
    def test_init_is_UNetBlock(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        activation = "ReLU"
        bias = False

        tfc_tdt_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            activation=activation,
            bias=bias
        )

        self.assertIsInstance(tfc_tdt_block, UNetBlock)
    
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        tfc_tdt_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
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

        expected_tdt_block = TimeDistributedTransformerBlock(
            embed_dim=frequency_bins,
            num_layers=num_layers,
            dropout=dropout,
            seq_length=frequency_bins,
        )

        self.assertEqual(tfc_tdt_block.tfc_block.__str__(), 
                         expected_tfc_block.__str__())
        self.assertEqual(tfc_tdt_block.tdt_block.__str__(), 
                         expected_tdt_block.__str__())

    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdt_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        ).to(device)

        x = torch.rand([1, in_channels, 44, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdt_block(x)
        expected_shape = [1, growth_rate, 44, frequency_bins]

        self.assertEqual(list(output.shape), expected_shape)



class TestDownSampleBlock(unittest.TestCase):
    def test_init_is_UNetDownSampleBlock(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        tfc_tdt_downsample_block = TFC_TDT_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )


        self.assertIsInstance(tfc_tdt_downsample_block, UNetDownSampleBlock)
    
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        tfc_tdt_downsample_block = TFC_TDT_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )

        expected_tfc_tdt_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )

        expected_downsample_block = DownSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdt_downsample_block.unet_block.__str__(), 
                         expected_tfc_tdt_block.__str__())
        
        self.assertEqual(tfc_tdt_downsample_block.down_block.__str__(), 
                         expected_downsample_block.__str__())


    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")
        
        tfc_tdt_downsample_block = TFC_TDT_DownSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        ).to(device)


        x = torch.rand([1, in_channels, 44, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdt_downsample_block(x)
        expected_shape = [1, growth_rate, 22, frequency_bins // 2]

        self.assertEqual(list(output.shape), expected_shape)


class TestUpSampleBlock(unittest.TestCase):
    def test_init_is_UNetUpSampleBlock(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        tfc_tdt_upsample_block = TFC_TDT_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )


        self.assertIsInstance(tfc_tdt_upsample_block, UNetUpSampleBlock)
    
    def test_init_architecture(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        tfc_tdt_upsample_block = TFC_TDT_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )

        expected_tfc_tdt_block = TFC_TDT(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        )

        expected_downsample_block = UpSample2DBlock(
            in_channels=growth_rate,
            out_channels=growth_rate,
            activation=activation,
            bias=bias
        )

        self.assertEqual(tfc_tdt_upsample_block.unet_block.__str__(), 
                         expected_tfc_tdt_block.__str__())
        
        self.assertEqual(tfc_tdt_upsample_block.up_block.__str__(), 
                         expected_downsample_block.__str__())


    def test_forward_output_dims(self):
        in_channels = 4
        num_layers = 2
        growth_rate = 8
        kernel_size = (3, 3)
        frequency_bins = 1024
        dropout = 0.2
        activation = "ReLU"
        bias = False

        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu")

        tfc_tdt_upsample_block = TFC_TDT_UpSample(
            in_channels=in_channels,
            num_layers=num_layers,
            growth_rate=growth_rate,
            kernel_size=kernel_size,
            frequency_bins=frequency_bins,
            dropout=dropout,
            activation=activation,
            bias=bias
        ).to(device)

        x = torch.rand([1, in_channels, 22, frequency_bins]).to(device) # [batch, channel, n_frames, frequency_bins]
        output = tfc_tdt_upsample_block(x)
        expected_shape = [1, growth_rate, 44, frequency_bins * 2]

        self.assertEqual(list(output.shape), expected_shape)
if __name__ == "__main__":
    # suite = unittest.TestSuite()

    # suite.addTest(TestDownSampleBlock("test_forward_output_dims"))

    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    unittest.main()