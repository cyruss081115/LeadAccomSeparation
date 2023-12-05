import sys, os

sys.path.append(os.getcwd())

from source.utils.path_utils import DATASETS_ROOT_DIR, OUTPUT_DIR
import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
import os
import copy
import torchaudio

from openunmix import data
from openunmix import model
from openunmix import utils
from openunmix import transforms

from source.model.TFC_TDF_UNet import TFC_TDF_UNet_v1
from source.model.processing import STFTProcessing
from source.model.TFC_TDSA_UNet import TFC_TDSA_UNet

tqdm.monitor_interval = 0


def train(args, model, encoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    model.train()
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        X = encoder(x)
        Y_hat = model(X)
        Y = encoder(y)
        Y = Y[:, :, :Y_hat.shape[2], :] # crop to predicted size
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    return losses.avg


def valid(args, model, encoder, device, valid_sampler):
    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            X = encoder(x)
            Y_hat = model(X)
            Y = encoder(y)
            Y = Y[:, :, :Y_hat.shape[2], :] # crop to predicted size
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


def get_statistics(args, encoder, dataset):
    # encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.SourceFolderDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def main():
    parser = argparse.ArgumentParser(description="Open Unmix Trainer")

    # which target do we want to train?
    parser.add_argument(
        "--target",
        type=str,
        default="vocals",
        help="target source (will be passed to the dataset)",
    )

    # Dataset paramaters
    parser.add_argument(
        "--dataset",
        type=str,
        default="musdb",
        choices=[
            "musdb",
            "aligned",
            "sourcefolder",
            "trackfolder_var",
            "trackfolder_fix",
        ],
        help="Name of the dataset.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help="provide output path base folder name",
    )
    parser.add_argument(
        "--model", type=str, help="Name or path of pretrained model to fine-tune"
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path of checkpoint to resume training"
    )

    # Training Parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=140,
        help="maximum number of train epochs (default: 140)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.00001, help="weight decay"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    # Model Parameters
    parser.add_argument(
        "--seq-dur",
        type=float,
        default=3,
        help="Sequence duration in seconds"
        "value of <=0.0 will use full/variable length",
    )
    parser.add_argument(
        "--unidirectional",
        action="store_true",
        default=False,
        help="Use unidirectional LSTM",
    )
    parser.add_argument(
        "--nfft", type=int, default=2048, help="STFT fft size and window size"
    )
    parser.add_argument("--nhop", type=int, default=1024, help="STFT hop size")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=512,
        help="hidden size parameter of bottleneck layers",
    )
    parser.add_argument(
        "--bandwidth", type=int, default=16000, help="maximum model bandwidth in herz"
    )
    parser.add_argument(
        "--nb-channels",
        type=int,
        default=4,
        help="set number of channels for model (1, 2)",
    )
    parser.add_argument(
        "--nb-workers", type=int, default=0, help="Number of workers for dataloader."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Speed up training init for dev purposes",
    )
    parser.add_argument(
        "--use-unet-stft", action="store_true", default=True, help="Use UNet STFT processing"
    )

    # Misc Parameters
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument("--samples-per-track", type=int, default=64)
    parser.add_argument(
        "--source-augmentations", type=str, default=["gain", "channelswap"], nargs="+"
    )

    args, _ = parser.parse_known_args()

    args.root = os.path.join(DATASETS_ROOT_DIR, 'musdb18hq')
    
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else
              "cpu")
    print("Using device:", device)
    dataloader_kwargs = (
        {"num_workers": args.nb_workers, "pin_memory": True} if device == 'cuda' else {}
    )

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_dataset, valid_dataset, args = data.load_datasets(args)

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, **dataloader_kwargs
    )

    if args.use_unet_stft:
        stft_processing = STFTProcessing(n_fft=args.nfft, hop_length=args.nhop, device=device)
        encoder = stft_processing.preprocess
    else:
        stft, _ = transforms.make_filterbanks(
            n_fft=args.nfft, n_hop=args.nhop, sample_rate=train_dataset.sample_rate
        )
        encoder = torch.nn.Sequential(
            stft, model.ComplexNorm(mono=args.nb_channels == 1)
        ).to(device)

    separator_conf = {
        "nfft": args.nfft,
        "nhop": args.nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": args.nb_channels,
    }

    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    # if args.checkpoint or args.model or args.debug:
    #     scaler_mean = None
    #     scaler_std = None
    # else:
    #     scaler_mean, scaler_std = get_statistics(args, encoder, train_dataset)

    # max_bin = utils.bandwidth_to_max_bin(
    #     train_dataset.sample_rate, args.nfft, args.bandwidth
    # )

    if args.model:
        # fine tune model
        print(f"Fine-tuning model from {args.model}")
        model = utils.load_target_models(
            args.target, model_str_or_path=args.model, device=device, pretrained=True
        )[args.target]
        model = model.to(device)
    else:
        # model = model.OpenUnmix(
        #     input_mean=scaler_mean,
        #     input_scale=scaler_std,
        #     nb_bins=args.nfft // 2 + 1,
        #     nb_channels=args.nb_channels,
        #     hidden_size=args.hidden_size,
        #     max_bin=max_bin,
        #     unidirectional=args.unidirectional,
        # ).to(device)
        # model = TFC_TDF_UNet_v1(
        #     num_channels=args.nb_channels,
        #     unet_depth=3,
        #     tfc_tdf_interal_layers=1,
        #     growth_rate=24,
        #     kernel_size=(3, 3),
        #     frequency_bins=args.nfft // 2,
        #     bottleneck=args.nfft // 16,
        #     activation="ReLU",
        #     bias=False
        # ).to(device)
        model = TFC_TDSA_UNet(
            num_channels=args.nb_channels,
            unet_depth=3,
            tfc_tdsa_internal_layers=1,
            growth_rate=24,
            kernel_size=(3, 3),
            frequency_bins=args.nfft // 2,
            num_attention_heads=1,
            activation="ReLU",
            bias=False
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)


    # if a checkpoint is specified: resume training
    if args.checkpoint:
        model_path = Path(args.checkpoint).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss = train(args, model, encoder, device, train_sampler, optimizer)
        valid_loss = valid(args, model, encoder, device, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=args.target,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "args": vars(args),
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
        }

        with open(Path(target_path, args.target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()
