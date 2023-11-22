import sys, os
sys.path.append(os.getcwd())

import torch
import torchaudio
from source.utils.path_utils import DATASETS_ROOT_DIR
import source.utils.log_utils as logu

logu.info(f"Downloading MUSDB18HQ dataset to {DATASETS_ROOT_DIR}...")
if not os.path.exists(DATASETS_ROOT_DIR):
    os.makedirs(DATASETS_ROOT_DIR)
    logu.info(f"Created directory at {DATASETS_ROOT_DIR}...")

musdb_hq_train = torchaudio.datasets.MUSDB_HQ(DATASETS_ROOT_DIR, subset='train', sources=['mixture', 'vocals'], download=True)
musdb_hq_test = torchaudio.datasets.MUSDB_HQ(DATASETS_ROOT_DIR, subset='test', sources=['mixture', 'vocals'], download=True)

print(f"Done!")
