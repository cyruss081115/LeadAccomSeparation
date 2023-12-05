# Lead and Accompaniment Separation
## TODO items
- [X] STFT preprocessing and postprocessing
- [X] Build TFC-TDF-UNet
- [X] Build TFC-TDSA-UNet
- [X] Data augmentation
- [X] Prepare presentation slides
- [ ] Prepare report

## Current directory structure
```
..
├── datasets
│   └── musdb18hq
│       ├── test
│       └── train
├── outputs
├── scripts
│   ├── download_musdb_hq.py
│   ├── train.py
│   ├── train_accelerate.py
│   └── train_openunmix.py
├── source
│   ├── model
│   │   ├── TFC_TDF_UNet
│   │   │   ├── __init__.py
│   │   │   ├── buildingBlocks.py
│   │   │   └── unet.py
│   │   ├── TFC_TDSA_UNet
│   │   │   ├── __init__.py
│   │   │   ├── buildingBlocks.py
│   │   │   └── unet.py
│   │   ├── __init__.py
│   │   ├── basicBlocks.py
│   │   ├── processing.py
│   │   └── unetBlock.py
│   └── utils
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── log_utils.py
│       └── path_utils.py
├── src
│   └── openunmix
└── test
    ├── model
    │   ├── TFC_TDF_UNet
    │   │   ├── __init__.py
    │   │   ├── processing_example.py
    │   │   ├── test_buildingBlocks.py
    │   │   └── test_unet.py
    │   ├── TFC_TDSA_UNet
    │   │   ├── __init__.py
    │   │   ├── test_buildingBlocks.py
    │   │   └── test_unet.py
    │   ├── __init__.py
    │   ├── test_basicBlocks.py
    │   └── test_processing.py
    └── run_test.py


```
