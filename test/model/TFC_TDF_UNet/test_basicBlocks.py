import sys, os
sys.path.append(os.getcwd())

from source.model.TFC_TDF_UNet.basicBlocks import (
    TimeDistributedFullyConnectedBlock,
    TimeDistributedConvolutionBlock,
    TimeDistributedSelfAttentionBlock,
    TimeFrequencyConvolutionBlock,
    DownSample2DBlock,
    UpSample2DBlock,
)
import torch.nn as nn
import unittest
import torch

class TestTDFBlock(unittest.TestCase):
    def test_init_other_activation(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=1, activation='LeakyReLU')
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(4096, 4096, bias=False),
                nn.LeakyReLU(),
            )
        ])
        self.assertEqual(
            tdf_block.norm_fc_act_stack.__str__(),
            expected_architecture.__str__())

    def test_init_hiddenlayer_equals_1(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=1)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(4096, 4096, bias=False),
                nn.ReLU(),
            )
        ])
        self.assertEqual(
            tdf_block.norm_fc_act_stack.__str__(),
            expected_architecture.__str__())

    def test_init_hiddenlayer_equals_2(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=2)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(4096, 128, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(128, 4096, bias=False),
                nn.ReLU(),
            ),
        ])
        self.assertEqual(
            tdf_block.norm_fc_act_stack.__str__(),
            expected_architecture.__str__())

    def test_init_hiddenlayer_larger_than_2(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=4)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(4096, 128, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(128, 128, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(128, 128, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Linear(128, 4096, bias=False),
                nn.ReLU(),
            )
        ])
        self.assertEqual(
            tdf_block.norm_fc_act_stack.__str__(),
            expected_architecture.__str__())

    def test_forward_hiddenlayer_equals_1(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=1)
        input = torch.rand([1, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [1, 4, 22050, 128])

    def test_forward_hiddenlayer_equals_2(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=2)
        input = torch.rand([1, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [1, 4, 22050, 128])

    def test_forward_hiddenlayer_greater_than_2(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=5)
        input = torch.rand([1, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [1, 4, 22050, 128])
        
    def test_forward_batch_greater_than_1(self):
        tdf_block = TimeDistributedFullyConnectedBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=5)
        input = torch.rand([3, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [3, 4, 22050, 128])


class TestTDCBlock(unittest.TestCase):
    def test_init_other_activation(self):
        tdc_block = TimeDistributedConvolutionBlock(in_channels=4, growth_rate=24, num_layers=1, kernel_size=3, activation='LeakyReLU')
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(4),
                nn.Conv1d(4, 24, kernel_size=3, padding=1, bias=False),
                nn.LeakyReLU(),
            )
        ])
        self.assertEqual(
            tdc_block.norm_conv_act_stack.__str__(),
            expected_architecture.__str__())

    def test_init_num_layers_equals_1(self):
        tdc_block = TimeDistributedConvolutionBlock(in_channels=4, growth_rate=24, num_layers=1, kernel_size=3)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(4),
                nn.Conv1d(4, 24, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            )
        ])
        self.assertEqual(
            tdc_block.norm_conv_act_stack.__str__(),
            expected_architecture.__str__())
        
    def test_init_num_layers_greater_than_1(self):
        tdc_block = TimeDistributedConvolutionBlock(in_channels=4, growth_rate=24, num_layers=4, kernel_size=3)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(4),
                nn.Conv1d(4, 24, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm1d(4+24),
                nn.Conv1d(4+24, 24, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm1d(4+(24*2)),
                nn.Conv1d(4+(24*2), 24, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm1d(4+(24*3)),
                nn.Conv1d(4+(24*3), 24, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            ),
        ])
        self.assertEqual(
            tdc_block.norm_conv_act_stack.__str__(),
            expected_architecture.__str__())

    def test_forward_num_layers_equals_1(self):
        tdc_block = TimeDistributedConvolutionBlock(in_channels=4, growth_rate=24, num_layers=1, kernel_size=3)
        input = torch.rand([1, 4, 1378, 1024]) # [B, C, T, F]
        output = tdc_block(input)
        self.assertEqual(list(output.shape), [1, 24, 1378, 1024])

    def test_forward_num_layers_greater_than_1(self):
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.cuda.is_available() else
            'cpu'
        )
        tdc_block = TimeDistributedConvolutionBlock(in_channels=4, growth_rate=24, num_layers=3, kernel_size=3).to(device)
        input = torch.rand([1, 4, 1378, 1024]).to(device) # [B, C, T, F]
        output = tdc_block(input)
        self.assertEqual(list(output.shape), [1, 24, 1378, 1024])

    def test_forward_batch_larger_than_1(self):
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.cuda.is_available() else
            'cpu'
        )
        tdc_block = TimeDistributedConvolutionBlock(in_channels=4, growth_rate=24, num_layers=3, kernel_size=3).to(device)
        input = torch.rand([2, 4, 1378, 1024]).to(device) # [B, C, T, F]
        output = tdc_block(input)
        self.assertEqual(list(output.shape), [2, 24, 1378, 1024])


class TestTDSABlock(unittest.TestCase):
    def test_init_num_layers_equals_1(self):
        tdsa_block = TimeDistributedSelfAttentionBlock(embed_dim=1024, num_heads=8, num_layers=1)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.MultiheadAttention(1024, 8, dropout=0., bias=False),
                nn.ReLU(),
            )
        ])
        self.assertEqual(
            tdsa_block.norm_attn_act_stack.__str__(),
            expected_architecture.__str__())

    def test_init_num_layers_greater_than_1(self):
        tdsa_block = TimeDistributedSelfAttentionBlock(embed_dim=1024, num_heads=8, num_layers=2)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.MultiheadAttention(1024, 8, dropout=0., bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.MultiheadAttention(1024, 8, dropout=0., bias=False),
                nn.ReLU(),
            )
        ])
        self.assertEqual(
            tdsa_block.norm_attn_act_stack.__str__(),
            expected_architecture.__str__())
    
    def test_forward_num_layers_equals_1(self):
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.cuda.is_available() else
            'cpu'
        ) 
        tdsa_block = TimeDistributedSelfAttentionBlock(embed_dim=1024, num_heads=8, num_layers=1).to(device)
        input = torch.rand([1, 4, 1378, 1024]).to(device)
        output = tdsa_block(input)

        self.assertEqual(list(output.shape), [1, 4, 1378, 1024])

    def test_forward_num_layers_greater_than_1(self):
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.cuda.is_available() else
            'cpu'
        ) 
        tdsa_block = TimeDistributedSelfAttentionBlock(embed_dim=1024, num_heads=8, num_layers=3).to(device)
        input = torch.rand([1, 4, 1378, 1024]).to(device) # [B, C, T, F]
        output = tdsa_block(input)

        self.assertEqual(list(output.shape), [1, 4, 1378, 1024])
    
    def test_forward_batch_larger_than_1(self):
        device = (
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.cuda.is_available() else
            'cpu'
        ) 
        tdsa_block = TimeDistributedSelfAttentionBlock(embed_dim=1024, num_heads=8, num_layers=3).to(device)
        input = torch.rand([3, 4, 100, 1024]).to(device) # [B, C, T, F]
        output = tdsa_block(input)

        self.assertEqual(list(output.shape), [3, 4, 100, 1024])


class TestDownSampleBlock(unittest.TestCase):
    def test_init_same_in_out_channels(self):
        downsample_block = DownSample2DBlock(in_channels=4, out_channels=4)
        expected_architecture = nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Conv2d(4, 4, kernel_size=[2, 2], stride=2, padding=0, bias=False),
                nn.ReLU(),
            )
        self.assertEqual(
            downsample_block.norm_conv_act.__str__(),
            expected_architecture.__str__())
    
    def test_init_different_in_out_channels(self):
        downsample_block = DownSample2DBlock(in_channels=2, out_channels=4)
        expected_architecture = nn.Sequential(
                nn.BatchNorm2d(2),
                nn.Conv2d(2, 4, kernel_size=[2, 2], stride=2, padding=0, bias=False),
                nn.ReLU(),
            )
        self.assertEqual(
            downsample_block.norm_conv_act.__str__(),
            expected_architecture.__str__())

    def test_forward_output_dims_same_in_out_channels(self):
        downsample_block = DownSample2DBlock(in_channels=4, out_channels=4)
        input = torch.rand([1, 4, 2048, 1024]) # [B, C, T, F]
        output = downsample_block(input)
        self.assertEqual(list(output.shape), [1, 4, 1024, 512])
    
    def test_forward_output_dims_different_in_out_channels(self):
        downsample_block = DownSample2DBlock(in_channels=4, out_channels=2)
        input = torch.rand([1, 4, 2048, 1024])
        output = downsample_block(input)
        self.assertEqual(list(output.shape), [1, 2, 1024, 512])

    def test_forward_output_dims_batach_greater_than_1(self):
        downsample_block = DownSample2DBlock(in_channels=4, out_channels=4)
        input = torch.rand([3, 4, 1024, 512]) # [B, C, T, F]
        output = downsample_block(input)
        self.assertEqual(list(output.shape), [3, 4, 512, 256])
        

class TestUpSampleBlock(unittest.TestCase):
    def test_init_same_in_out_channels(self):
        downsample_block = UpSample2DBlock(in_channels=4, out_channels=4)
        expected_architecture = nn.Sequential(
                nn.BatchNorm2d(4),
                nn.ConvTranspose2d(4, 4, kernel_size=[2, 2], stride=2, padding=0, bias=False),
                nn.ReLU(),
            )
        self.assertEqual(
            downsample_block.norm_conv_act.__str__(),
            expected_architecture.__str__())
    
    def test_init_different_in_out_channels(self):
        downsample_block = UpSample2DBlock(in_channels=2, out_channels=4)
        expected_architecture = nn.Sequential(
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(2, 4, kernel_size=[2, 2], stride=2, padding=0, bias=False),
                nn.ReLU(),
            )
        self.assertEqual(
            downsample_block.norm_conv_act.__str__(),
            expected_architecture.__str__())

    def test_forward_output_dims_same_in_out_channels(self):
        downsample_block = UpSample2DBlock(in_channels=4, out_channels=4)
        input = torch.rand([1, 4, 1024, 512]) # [B, C, T, F]
        output = downsample_block(input)
        self.assertEqual(list(output.shape), [1, 4, 2048, 1024])
    
    def test_forward_output_dims_different_in_out_channels(self):
        downsample_block = UpSample2DBlock(in_channels=4, out_channels=2)
        input = torch.rand([1, 4, 1024, 512])
        output = downsample_block(input)
        self.assertEqual(list(output.shape), [1, 2, 2048, 1024])

    def test_forward_output_dims_batach_greater_than_1(self):
        downsample_block = UpSample2DBlock(in_channels=4, out_channels=4)
        input = torch.rand([3, 4, 512, 256]) # [B, C, T, F]
        output = downsample_block(input)
        self.assertEqual(list(output.shape), [3, 4, 1024, 512])


class TestTFCBlock(unittest.TestCase):
    def test_init_num_layer_equals_1(self):
        tfc_block = TimeFrequencyConvolutionBlock(in_channels=4, growth_rate=8, num_layers=1)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Conv2d(4, 8, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.ReLU(),
            )
        ])
        self.assertEqual(
            tfc_block.norm_conv_act_stack.__str__(),
            expected_architecture.__str__())

    def test_init_kernel_size_not_odd(self):
        with self.assertRaises(ValueError):
            tfc_block = TimeFrequencyConvolutionBlock(in_channels=4, growth_rate=8, num_layers=1, kernel_size=(2, 2))

    def test_init_num_layer_greater_than_1(self):
        tfc_block = TimeFrequencyConvolutionBlock(in_channels=4, growth_rate=8, num_layers=3)
        expected_architecture = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(4),
                nn.Conv2d(4, 8, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4+8),
                nn.Conv2d(4+8, 8, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.BatchNorm2d(4+(8*2)),
                nn.Conv2d(4+(8*2), 8, kernel_size=(3, 3), stride=1, padding=1, bias=False),
                nn.ReLU(),
            ),
        ])
        self.assertEqual(
            tfc_block.norm_conv_act_stack.__str__(),
            expected_architecture.__str__())

    def test_forward_num_layer_equals_1(self):
        tfc_block = TimeFrequencyConvolutionBlock(in_channels=2, growth_rate=8, num_layers=1)
        input = torch.rand([1, 2, 1024, 512])
        output = tfc_block(input)
        self.assertEqual(list(output.shape), [1, 8, 1024, 512])

    def test_forward_num_layer_greater_than_1(self):
        tfc_block = TimeFrequencyConvolutionBlock(in_channels=2, growth_rate=8, num_layers=3)
        input = torch.rand([1, 2, 1024, 512])
        output = tfc_block(input)
        self.assertEqual(list(output.shape), [1, 8, 1024, 512])


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromModule(TestTFCBlock())
    unittest.TextTestRunner().run(suite)

    # unittest.main()
