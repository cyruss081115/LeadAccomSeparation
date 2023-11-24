import sys, os
sys.path.append(os.getcwd())

from source.model.TFC_TDF_UNet.processing import STFTProcessing
import unittest
import torch


class TestSTFTProcessing(unittest.TestCase):
    def test_preprocess_output_dims(self):
        stft_processing = STFTProcessing(n_fft=2048, hop_length=1024)
        waveform = torch.rand([1, 2, 44100]) # [batch, channel, n_samples]
        spec = stft_processing.preprocess(waveform)
        expected_shape = [1, 4, 1024, 44] # [batch, 2 * channel, n_fft // 2, 1+ n_samples // hop_length]
        self.assertEqual(list(spec.shape), expected_shape)
    
    def test_preprocess_output_dims_batch(self):
        stft_processing = STFTProcessing(n_fft=2048, hop_length=1024)
        waveform = torch.rand([3, 2, 44100])
        spec = stft_processing.preprocess(waveform)
        expected_shape = [3, 4, 1024, 44]
        self.assertEqual(list(spec.shape), expected_shape)

    def test_postprocess_output_dims(self):
        stft_processing = STFTProcessing(n_fft=2048, hop_length=1024)
        spec = torch.rand([1, 4, 1024, 44])
        waveform = stft_processing.postprocess(spec)
        expected_shape = [1, 2, 44100]

        self.assertAlmostEqual(list(waveform.shape)[-1], expected_shape[-1], delta=70)
        self.assertEqual(list(waveform.shape)[:-1], expected_shape[:-1])

if __name__ == "__main__":
    unittest.main()
    