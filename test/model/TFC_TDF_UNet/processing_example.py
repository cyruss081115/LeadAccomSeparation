import sys, os
sys.path.append(os.getcwd())

import torch
import torchaudio
from source.model.processing import STFTProcessing
from source.utils.path_utils import DATASETS_ROOT_DIR, OUTPUT_DIR
from source.utils.audio_utils import plot_spectrogram


if __name__ == "__main__":
    frame_start, frame_end = 0, 100

    musdb18hq_train_dir = os.path.join(DATASETS_ROOT_DIR, 'musdb18hq', 'train')

    file1 = os.path.join(musdb18hq_train_dir, 'A Classic Education - NightOwl', 'mixture.wav')
    waveform1, sample_rate = torchaudio.load(file1)
    waveform1 = waveform1[:, frame_start*sample_rate:frame_end*sample_rate]
    plot_spectrogram(
        waveform1, n_fft=2048, hop_length=1024, title='Raw waveform 1', figsize=(8, 6), save_dir=OUTPUT_DIR, verbose=True)

    file2 = os.path.join(musdb18hq_train_dir, 'Actions - One Minute Smile', 'mixture.wav')
    waveform2, sample_rate = torchaudio.load(file2)
    waveform2 = waveform2[:, frame_start*sample_rate:frame_end*sample_rate]

    waveform = torch.stack([waveform1, waveform2], dim=0)
    print("Waveform shape: ", waveform.shape)

    processing = STFTProcessing(n_fft=2048, hop_length=1024)
    spec = processing.preprocess(waveform)
    print("Spectrogram shape: ", spec.shape)
    plot_spectrogram(
        spec[0, [0,1], :, :], title='Preprocessed spectrogram 1', figsize=(8, 6), save_dir=OUTPUT_DIR, verbose=True)

    reconstructed_waveform = processing.postprocess(spec)
    print("Reconstructed waveform shape: ", reconstructed_waveform.shape)
    torchaudio.save(
        os.path.join(OUTPUT_DIR, 'reconstructed.wav'), reconstructed_waveform[0], sample_rate)
    torchaudio.save(
        os.path.join(OUTPUT_DIR, 'waveform.wav'), waveform[0], sample_rate)
