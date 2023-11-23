import torch
import einops
from abc import ABC

class Processing(ABC):
    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Preprocess the waveform to a feature tensor. Called before the model.
        """
        pass
    def postprocess(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Postprocess the feature tensor to a waveform. Called after the model.
        """
        pass

class STFTProcessing(Processing):
    def __init__(self, n_fft: int, hop_length: int):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Apply STFT to a c-channeled waveform, concatenate the real and imaginary parts, and return the result.
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (b, c, n_samples)
        Returns:
            torch.Tensor: Tensor of dimension (b, 2 * c, n_fft, n_frames)
        """
        assert waveform.dim() == 3, "The input waveform must be of dimension (b, c, n_samples)."

        batch, channel, _ = waveform.shape
        waveform = einops.rearrange(waveform, 'b c n -> (b c) n')
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=True)
        spec = torch.view_as_real(spec) # [batch * channel, n_fft, n_frames, 2]
        spec = einops.rearrange(
            spec, "(b c) f t r-> b (r c) f t", b=batch, c=channel) # split b,c and concat real and imaginary parts
        return spec

    def postprocess(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Apply inverse STFT to a c-channeled feature, and return the result.
        Args:
            feature (torch.Tensor): Tensor of feature of dimension (b, 2 * c, n_fft, n_frames)
        Returns:
            torch.Tensor: Tensor of dimension (b, c, n_samples)
        """
        assert feature.dim() == 4, "The input feature must be of dimension (b, 2 * c, n_fft, n_frames)."

        batch, _, _, _ = feature.shape
        feature = einops.rearrange(feature, 'b (r c) f t -> (b c) f t r', r=2)
        feature = torch.view_as_complex(feature) # [batch * channel, n_fft, n_frames]
        waveform = torch.istft(feature, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=False)
        waveform = einops.rearrange(waveform, "(b c) n -> b c n", b=batch)

        return waveform

class VAEProcessing(Processing):
    def __init__(self):
        raise NotImplementedError

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def postprocess(self, feature: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
