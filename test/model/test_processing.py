import sys, os
sys.path.append(os.getcwd())

from source.model.processing import STFTProcessing
import unittest
import torch


class TestSTFTProcessing(unittest.TestCase):
    def test_preprocess_output_dims(self):
        stft_processing = STFTProcessing(n_fft=2048, hop_length=1024, device='cpu')
        waveform = torch.rand([1, 2, 44100]) # [batch, channel, n_samples]
        spec = stft_processing.preprocess(waveform)
        expected_shape = [1, 4, 44, 1024] # [batch, 2 * channel, 1+ n_samples // hop_length, n_fft // 2]
        self.assertEqual(list(spec.shape), expected_shape)
    
    def test_preprocess_output_dims_batch(self):
        stft_processing = STFTProcessing(n_fft=2048, hop_length=1024, device='cpu')
        waveform = torch.rand([3, 2, 44100])
        spec = stft_processing.preprocess(waveform)
        expected_shape = [3, 4, 44, 1024]
        self.assertEqual(list(spec.shape), expected_shape)

    def test_postprocess_output_dims(self):
        stft_processing = STFTProcessing(n_fft=2048, hop_length=1024, device='cpu')
        spec = torch.rand([1, 4, 44, 1024])
        waveform = stft_processing.postprocess(spec)
        expected_shape = [1, 2, 44100]

        self.assertAlmostEqual(list(waveform.shape)[-1], expected_shape[-1], delta=70)
        self.assertEqual(list(waveform.shape)[:-1], expected_shape[:-1])

if __name__ == "__main__":
    unittest.main()
    