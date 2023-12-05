# Lead and Accompaniment Separation
## TODO items
- [X] STFT preprocessing and postprocessing
- [X] Build UNet
- [X] Data augmentation
- [ ] Prepare report
- [X] Prepare presentation slides

## Current directory structure
```
.
├── datasets
│   └── musdb18hq
│       ├── test
│       └── train
├── outputs
├── scripts
│   ├── download_musdb_hq.py
│   ├── evaluate.py
│   ├── train.py
│   ├── train_accelerate.py
│   └── train_openunmix.py
├── source
│   ├── model
│   │   ├── TFC_TDF_UNet
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
└── test
    ├── model
    │   ├── TFC_TDF_UNet
    │   │   ├── __init__.py
    │   │   ├── processing_example.py
    │   │   ├── test_buildingBlocks.py
    │   │   └── test_unet.py
    │   ├── __init__.py
    │   ├── test_basicBlocks.py
    │   └── test_processing.py
    └── run_test.py



```
