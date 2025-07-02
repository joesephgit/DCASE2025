
import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import transformers
import wandb

from dataset.dcase25 import get_training_set, get_test_set
from helpers.init import worker_init_fn
from helpers.utils import mixstyle
from helpers import complexity
from models.ntu_baseline import get_ntu_model
from pathlib import Path

os.environ["WANDB_SILENT"] = "true"


class PLModule(pl.LightningModule):
    """
    PyTorch Lightning Module for training the DCASE'25 baseline model (shared across all devices).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # -------- Preprocessing Pipeline --------
        self.mel = torch.nn.Sequential(
            torchaudio.transforms.Resample(
                orig_freq=config.orig_sample_rate,
                new_freq=config.sample_rate
            ),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.window_length,
                hop_length=config.hop_length,
                n_mels=config.n_mels,
                f_min=config.f_min,
                f_max=config.f_max
            )
        )
        self.mel_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True),
            #torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
        )

        # -------- Baseline Model --------
        self.model = get_ntu_model(n_classes=config.n_classes,
                                   in_channels=config.in_channels,
                                   base_channels=config.base_channels,
                                   channels_multiplier=config.channels_multiplier,
                                   expansion_rate=config.expansion_rate,
                                   divisor = config.divisor
                                   )

        # -------- Device/Label Definitions --------
        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = [
            'airport', 'bus', 'metro', 'metro_station', 'park',
            'public_square', 'shopping_mall', 'street_pedestrian',
            'street_traffic', 'tram'
        ]
        # Grouping devices into real/seen/unseen categories
        self.device_groups = {
            'a': "real", 'b': "real", 'c': "real",
            's1': "seen", 's2': "seen", 's3': "seen",
            's4': "unseen", 's5': "unseen", 's6': "unseen"
        }

        # Containers to store step outputs (PyTorch Lightning 2.x pattern)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        '''optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )'''

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.4,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, _, labels, _, _ = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, labels)

        # Log learning rate and epoch
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("epoch", self.current_epoch)

        # Log training loss
        self.log("train/loss", loss.detach().cpu())

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, _ = val_batch
        y_hat = self.forward(x)

        # Compute loss per sample
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # Initialize per-device stats
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)

        # Accumulate device-wise stats
        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devcnt.{d}"] += 1
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]

        # Initialize per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)

        # Accumulate label-wise stats
        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch. Aggregates per-device/per-label stats and logs them.
        """
        # Flatten the outputs into a dict of lists
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        # Compute overall metrics
        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()

        logs = {"acc": acc, "loss": avg_loss}

        # Per-device stats
        for d in self.device_ids:
            dev_loss = outputs[f"devloss.{d}"].sum()
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            dev_correct = outputs[f"devn_correct.{d}"].sum()
            logs[f"loss.{d}"] = dev_loss / dev_cnt
            logs[f"acc.{d}"] = dev_correct / dev_cnt
            logs[f"cnt.{d}"] = dev_cnt

            # Group stats
            group_name = self.device_groups[d]
            logs[f"acc.{group_name}"] = logs.get(f"acc.{group_name}", 0.) + dev_correct
            logs[f"count.{group_name}"] = logs.get(f"count.{group_name}", 0.) + dev_cnt
            logs[f"lloss.{group_name}"] = logs.get(f"lloss.{group_name}", 0.) + dev_loss

        # Reduce group stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Per-label stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()

            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt.float()

        # Compute macro-average accuracy over all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        # Log everything with 'val/' prefix
        self.log_dict({f"val/{k}": v for k, v in logs.items()})
        self.validation_step_outputs.clear()
        print('\n')

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, _ = test_batch

        # For memory constraints, switch model to half-precision
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()

        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")  

        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        results = {
            "loss": samples_loss.mean(),
            "n_correct": n_correct,
            "n_pred": torch.as_tensor(len(labels), device=self.device)
        }

        # Per-device stats
        for d in self.device_ids:
            results[f"devloss.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devcnt.{d}"] = torch.as_tensor(0., device=self.device)
            results[f"devn_correct.{d}"] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results[f"devloss.{d}"] += samples_loss[i]
            results[f"devn_correct.{d}"] += n_correct_per_sample[i]
            results[f"devcnt.{d}"] += 1

        # Per-label stats
        for lbl in self.label_ids:
            results[f"lblloss.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lblcnt.{lbl}"] = torch.as_tensor(0., device=self.device)
            results[f"lbln_correct.{lbl}"] = torch.as_tensor(0., device=self.device)

        for i, lbl_index in enumerate(labels):
            lbl_name = self.label_ids[lbl_index]
            results[f"lblloss.{lbl_name}"] += samples_loss[i]
            results[f"lbln_correct.{lbl_name}"] += n_correct_per_sample[i]
            results[f"lblcnt.{lbl_name}"] += 1

        self.test_step_outputs.append({k: v.cpu() for k, v in results.items()})

    def on_test_epoch_end(self):
        # Flatten the outputs
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k, v in step_output.items():
                outputs[k].append(v)

        # Stack each list of tensors
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs["loss"].mean()
        acc = outputs["n_correct"].sum() / outputs["n_pred"].sum()
        logs = {"acc": acc, "loss": avg_loss}

        # Device-level stats
        for d in self.device_ids:
            dev_loss = outputs[f"devloss.{d}"].sum()
            dev_cnt = outputs[f"devcnt.{d}"].sum()
            dev_correct = outputs[f"devn_correct.{d}"].sum()
            logs[f"loss.{d}"] = dev_loss / dev_cnt
            logs[f"acc.{d}"] = dev_correct / dev_cnt
            logs[f"cnt.{d}"] = dev_cnt

            # Device groups
            grp = self.device_groups[d]
            logs[f"acc.{grp}"] = logs.get(f"acc.{grp}", 0.) + dev_correct
            logs[f"count.{grp}"] = logs.get(f"count.{grp}", 0.) + dev_cnt
            logs[f"lloss.{grp}"] = logs.get(f"lloss.{grp}", 0.) + dev_loss

        # Group-level stats
        for grp in set(self.device_groups.values()):
            logs[f"acc.{grp}"] = logs[f"acc.{grp}"] / logs[f"count.{grp}"]
            logs[f"lloss.{grp}"] = logs[f"lloss.{grp}"] / logs[f"count.{grp}"]

        # Label-level stats
        for lbl in self.label_ids:
            lbl_loss = outputs[f"lblloss.{lbl}"].sum()
            lbl_cnt = outputs[f"lblcnt.{lbl}"].sum()
            lbl_correct = outputs[f"lbln_correct.{lbl}"].sum()
            logs[f"loss.{lbl}"] = lbl_loss / lbl_cnt
            logs[f"acc.{lbl}"] = lbl_correct / lbl_cnt
            logs[f"cnt.{lbl}"] = lbl_cnt

        # Macro-average accuracy over all labels
        logs["macro_avg_acc"] = torch.mean(torch.stack([logs[f"acc.{l}"] for l in self.label_ids]))

        self.log_dict({f"test/{k}": v for k, v in logs.items()})
        self.test_step_outputs.clear()


def train(config):
    """
        Main training loop using PyTorch Lightning.

        Args:
            config: Namespace or dictionary containing all experiment hyperparameters
                    (subset size, number of epochs, LR, etc.).
        """
    # Configure W&B logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Baseline System for DCASE'25 Task 1.",
        tags=["DCASE25"],
        config=config,  # logs all hyperparameters
        name=config.experiment_name
    )

    # Dataloader for training
    assert config.subset == 25, (
        "In DCASE`25 subset must be fixed to 25%."
    )

    roll_samples = config.orig_sample_rate * config.roll_sec
    train_ds = get_training_set(config.subset, device=None, roll=roll_samples)
    train_dl = DataLoader(
        dataset=train_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory = True,
        shuffle=True,
    )

    # Dataloader for testing
    test_ds = get_test_set(device=None)
    test_dl = DataLoader(
        dataset=test_ds,
        worker_init_fn=worker_init_fn,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory = True
    )

    # Create PyTorch Lightning module
    pl_module = PLModule(config)

    # Compute model complexity (MACs, parameters) and log to W&B
    sample = next(iter(test_dl))[0][0].unsqueeze(0)  # Single sample
    shape = pl_module.mel_forward(sample).size()
    macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=shape)
    wandb_logger.experiment.config["MACs"] = macs
    wandb_logger.experiment.config["Parameters_Bytes"] = params_bytes

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=1, monitor="val/acc", mode="max")

    # create the pytorch lightening trainer
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         gradient_clip_val = 0.5,
                         gradient_clip_algorithm="norm",                                                  
                         num_sanity_val_steps=0,
                         logger=wandb_logger,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1,
                         precision=config.precision,
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback])
                         #callbacks=[pl.callbacks.ModelCheckpoint(save_last=True)])

    # Fit (train + validation)
    trainer.fit(pl_module, train_dl, test_dl)

    # Final test (using the same test_dl here)
    ckp_path = Path(checkpoint_callback.best_model_path)
    trainer.test(ckpt_path=ckp_path, dataloaders=test_dl)

    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCASE 25 argument parser')

    # General arguments
    parser.add_argument("--project_name", type=str, default="DCASE25_Task1")
    parser.add_argument("--experiment_name", type=str, default="General")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--orig_sample_rate", type=int, default=44100)

    # in DCASE`25 Task 1 only the 25% subset can be used
    parser.add_argument("--subset", type=int, default=25)

    # Model hyperparameters
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--channels_multiplier", type=float, default=1.5)
    parser.add_argument("--expansion_rate", type=float, default=2)
    parser.add_argument('--divisor', type=float, default=5)

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mixstyle_p", type=float, default=0.4)
    parser.add_argument("--mixstyle_alpha", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--roll_sec", type=float, default=0)
    parser.add_argument('--dir_p', type=float, default=0.4)  #  mic impulse response

    # Learning rate schedule
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--warmup_steps", type=int, default=20)

    # Spectrogram parameters
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--window_length", type=int, default=8192)
    parser.add_argument("--hop_length", type=int, default=1364)
    parser.add_argument("--n_fft", type=int, default=8912)
    parser.add_argument("--n_mels", type=int, default=256)
    parser.add_argument("--freqm", type=int, default=48)
    parser.add_argument("--timem", type=int, default=0)
    parser.add_argument("--f_min", type=int, default=1)
    parser.add_argument("--f_max", type=int, default=None)
    args = parser.parse_args()
    train(args)
