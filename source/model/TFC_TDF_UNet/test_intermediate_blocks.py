import sys, os
sys.path.append(os.getcwd())

from model.TFC_TDF_UNet.intermediate_blocks import TDFBlock, TDCBlock
import torch.nn as nn
import unittest
import torch

class TestTDFBlock(unittest.TestCase):
    def test_init_hiddenlayer_equals_1(self):
        tdf_block = TDFBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=1)
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
        tdf_block = TDFBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=2)
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
        tdf_block = TDFBlock(channels=4, frequency_bins=4096, bottleneck=128, num_layers=4)
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
        tdf_block = TDFBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=1)
        input = torch.rand([1, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [1, 4, 22050, 128])

    def test_forward_hiddenlayer_equals_2(self):
        tdf_block = TDFBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=2)
        input = torch.rand([1, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [1, 4, 22050, 128])

    def test_forward_hiddenlayer_greater_than_2(self):
        tdf_block = TDFBlock(channels=4, frequency_bins=128, bottleneck=32, num_layers=5)
        input = torch.rand([1, 4, 22050, 128])
        output = tdf_block(input)
        self.assertEqual(list(output.shape), [1, 4, 22050, 128])
        

class TestTDCBlock(unittest.TestCase):
    def test_init_num_layers_equals_1(self):
        tdc_block = TDCBlock(in_channels=4, growth_rate=24, num_layers=1, kernel_size=3)
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
        tdc_block = TDCBlock(in_channels=4, growth_rate=24, num_layers=4, kernel_size=3)
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
        tdc_block = TDCBlock(in_channels=4, growth_rate=24, num_layers=1, kernel_size=3)
        input = torch.rand([1, 4, 11025, 1024]) # [B, C, T, F]
        output = tdc_block(input)
        self.assertEqual(list(output.shape), [1, 24, 11025, 1024])

    def test_forward_num_layers_greater_than_1(self):
        tdc_block = TDCBlock(in_channels=4, growth_rate=24, num_layers=3, kernel_size=3).to('mps')
        input = torch.rand([1, 4, 11025, 1024]).to('mps') # [B, C, T, F]
        output = tdc_block(input)
        self.assertEqual(list(output.shape), [1, 24, 11025, 1024])

if __name__ == "__main__":
    unittest.main()
