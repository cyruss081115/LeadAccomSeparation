import sys, os
sys.path.append(os.getcwd())

import argparse

import torch
import torchaudio

from source.utils.path_utils import DATASETS_ROOT_DIR, OUTPUT_DIR
import source.utils.log_utils as logu
from source.model import TFC_TDT_UNet, STFTProcessing


def extract_vocal(
        processor,
        model,
        mixture,
        segment_size=3.0,
        sample_rate=44100,
        device=None,
):
    if device is None:
        device = mixture.device

    _, _, duration = mixture.shape

    chunk_len = int(sample_rate * segment_size)
    start, end = 0, chunk_len

    out_fragments = []

    while start < duration:
        chunk = mixture[:, :, start:end].to('cpu')
        model.eval()
        with torch.no_grad():
            spec = processor.preprocess(chunk).to(device)
            out_spec = model.forward(spec)
            out = processor.postprocess(out_spec.to('cpu'))

        out_fragments.append(out)
        start += chunk_len
        end += chunk_len
    
    return torch.cat(out_fragments, dim=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lead and Acompaniment Separation with TFC-TDF-UNet')

    parser.add_argument('--weights_path', type=str, default="./source/model/TFC_TDT_UNet/weights/vocals_transformer_0511.pth")
    parser.add_argument('--mixture_path', type=str, default=os.path.join(DATASETS_ROOT_DIR, "musdb18hq/test/Mu - Too Bright/mixture.wav"))
    parser.add_argument('--segment_size', type=float, default=10.0)

    args, _ = parser.parse_known_args()

    device = (
        'cuda' if torch.cuda.is_available() else 
        'mps' if torch.backends.mps.is_available() else 
        'cpu'
    )

    # Load model
    model = TFC_TDT_UNet(
            num_channels=4,
            unet_depth=3,
            tfc_tdt_internal_layers=1,
            growth_rate=24,
            kernel_size=(3, 3),
            frequency_bins=1024,
            dropout=0.2,
            activation="ReLU",
            bias=False
        )
    
    state_dict = torch.load(args.weights_path, map_location='cpu')
    for key in list(state_dict.keys()):
        state_dict[key.replace("module.", "")] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.to(device)

    stft_processor = STFTProcessing(2048, 1024, 'cpu')

    original_track_output_path = os.path.join(OUTPUT_DIR + '/original.wav')
    separated_track_output_path = os.path.join(OUTPUT_DIR + '/separated.wav')

    waveform, sample_rate = torchaudio.load(args.mixture_path)
    logu.info(f"Loaded {args.mixture_path} with sample rate {sample_rate} and shape {waveform.shape}")

    torchaudio.save(original_track_output_path, waveform, sample_rate)
    logu.info(f"Saved original track to {original_track_output_path}")

    waveform = waveform.unsqueeze(0).to(device)
    logu.info('Separating tracks...')
    separated = extract_vocal(stft_processor, model, mixture=waveform, segment_size=args.segment_size, device=device)

    torchaudio.save(separated_track_output_path, separated.squeeze(), sample_rate)
    logu.success(f"Saved separated track to {separated_track_output_path}")

