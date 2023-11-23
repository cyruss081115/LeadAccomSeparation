import os
import torch
import torchaudio
from .log_utils import gettime, info
from typing import overload
import matplotlib.pyplot as plt

@overload
def plot_spectrogram(waveform: torch.Tensor, n_fft: int, hop_length: int, title: str, figsize: tuple, save_dir: str, verbose: bool):
    ...

@overload
def plot_spectrogram(spectrogram: torch.Tensor, title: str, figsize: tuple, save_dir: str, verbose: bool):
    ...

def plot_spectrogram(
        input: torch.Tensor,
        n_fft: int = None,
        hop_length: int = None,
        title: str = "Spectrogram",
        figsize: tuple = (8, 6),
        save_dir: str = None,
        verbose: bool = True):
    r"""
    Plots the spectrogram from the first channel of input. The input can be either a waveform or a spectrogram.
    By providing the n_fft and hop_length, the input will be converted to a spectrogram,
    else, the input will be treated as a spectrogram.

    Args:
        input (torch.Tensor): Tensor of either a spectrogram (n_channels, n_fft, n_frames) or a waveform (n_channels, n_samples).
        n_fft (int, optional): Size of FFT.
        hop_length (int, optional): Length of hop between STFT windows.
        title (str, optional): Title of the figure. Defaults to "Spectrogram".
        figsize (tuple, optional): Size of figure. Defaults to (8, 6).
        save_dir (str, optional): Path to save the figure. No saving when value is None. Defaults to None.
        verbose (bool, optional): Whether to print the information of saving the figure. Defaults to True.
    """
    is_waveform = n_fft is not None and hop_length is not None
    if is_waveform:
        info("Converting waveform to spectrogram...") if verbose else ...
        input = torch.stft(
            input, n_fft=n_fft, hop_length=hop_length, window=torch.hann_window(n_fft, periodic=True), return_complex=True)
    else:
        info("Using the input as a spectrogram...") if verbose else ...

    magnitude = input[0].abs() # keep only the first channel
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()

    _, ax = plt.subplots(figsize=figsize)
    ax.imshow(spectrogram, cmap='viridis', vmin=-60, vmax=0, origin='lower', aspect='auto')
    ax.set(title=title, xlabel='Frame', ylabel='Frequency')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_filename = f"{gettime(daysep='-', sep='-', timesep='-')}_{title.replace(' ', '_').lower()}.png"
        save_path = os.path.join(save_dir, save_filename)
        plt.savefig(save_path)
        info('Saved spectrogram to ', save_path) if verbose else ...
        plt.close()