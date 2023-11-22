import torch
import torchaudio
import matplotlib.pyplot as plt

def plot_spectrogram(waveform: torch.Tensor, n_fft: int = 4096, hop_length: int = 4):
    r"""
    Plots the spectrogram of a waveform.
    Args:
        waveform (torch.Tensor): Tensor of audio of dimension (n_channels, n_samples)
        n_fft (int, optional): Size of FFT. Defaults to 4096.
        hop_length (int, optional): Length of hop between STFT windows. Defaults to 4.
    """
    stft = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length)(waveform)
    magnitude = stft[0].abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    _, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(spectrogram, cmap='viridis', vmin=-60, vmax=0, origin='lower', aspect='auto')
    ax.set(title='Spectrogram', xlabel='Time', ylabel='Frequency')
    plt.tight_layout()
