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
    def __init__(self, n_fft: int, hop_length: int, device: torch.device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True).to(device)
        self.device = device

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Apply STFT to a c-channeled waveform, concatenate the real and imaginary parts, and return the result.

        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (b, c, n_samples)

        Returns:
            torch.Tensor: Tensor of dimension (b, 2 * c, n_frames, n_fft // 2)
        """
        assert waveform.dim() == 3, "The input waveform must be of dimension (b, c, n_samples)."

        batch, channel, n_samples = waveform.shape
        waveform = einops.rearrange(waveform, 'b c n -> (b c) n')
        spec = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, center=True, return_complex=True)
        spec = torch.view_as_real(spec) # [batch * channel, n_fft, n_frames, 2]
        spec = einops.rearrange(
            spec, "(b c) f t r-> b (r c) t f", b=batch, c=channel) # split b,c and concat real and imaginary parts
        return spec[..., :self.n_fft // 2]

    def postprocess(self, feature: torch.Tensor) -> torch.Tensor:
        r"""
        Apply inverse STFT to a c-channeled feature, and return the result.

        Note:
            The feature must be of dimension (b, 2 * c, n_frames, n_fft // 2). It will padded with zeros to match the
            frequency dimension requirements by `torch.istft`.
            The output number of samples may return a shorter signal than the original waveform due to `torch.istft`

        Args:
            feature (torch.Tensor): Tensor of feature of dimension (b, 2 * c, n_frames, n_fft // 2)
            
        Returns:
            torch.Tensor: Tensor of dimension (b, c, n_samples).
        """
        assert feature.dim() == 4, "The input feature must be of dimension (b, 2 * c, n_frames, n_fft // 2)."
        assert feature.shape[-1] == self.n_fft // 2, "The last dimension of the input feature must be n_fft // 2."

        b, _, _, _ = feature.shape
        feature = einops.rearrange(feature, 'b (r c) t f -> (b c) f t r', r=2)
        feature = feature.contiguous()

        # pad freq dimension to match dim required by one sided torch.istft
        bc, f, t, r = feature.shape
        n = (self.n_fft // 2) + 1
        padding = torch.zeros([bc, n - f, t, r])
        feature = torch.cat([feature, padding], dim=1)

        feature = torch.view_as_complex(feature) # [batch * channel, n_fft, n_frames]
        waveform = torch.istft(feature, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, return_complex=False)
        waveform = einops.rearrange(waveform, "(b c) n -> b c n", b=b)

        return waveform

class VAEProcessing(Processing):
    def __init__(self):
        raise NotImplementedError

    def preprocess(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def postprocess(self, feature: torch.Tensor) -> torch.Tensor:
        return NotImplementedError

__all__ = ['Processing', 'STFTProcessing']