import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import wandb
from json import dump

from dataset.dcase24 import get_test_set, get_eval_set, get_training_set 
from helpers.init import worker_init_fn
#from models.baseline import get_model
from models.ntu_baseline import get_ntu_model
from helpers.utils import mixstyle, set_seed #, ntu_mixstyle, mixstyle_1
from helpers import nessi
from helpers import complexity
#from torch.optim.lr_scheduler import ReduceLROnPlateau

from shutil import copy2
from pathlib import Path
from glob import glob
import stat
from numpy import random
import copy
import scipy.signal as sci
#from torch.autograd import Variable

os.environ["WANDB_SILENT"] = "true"
#wandb.login()

#from pytorch_lightning.strategies import DeepSpeedStrategy


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.sample_rate
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.window_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max
        )

        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )    

        self.mel_augment = torch.nn.Sequential(
            freqm,
            #specag
            timem
        )
        

        # the baseline model
        self.model = get_ntu_model(n_classes=config.n_classes,
                                   in_channels=config.in_channels,
                                   base_channels=config.base_channels,
                                   channels_multiplier=config.channels_multiplier,
                                   expansion_rate=config.expansion_rate,
                                   divisor = config.divisor
                                   )

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        #xs = self.mels(x)
        x = self.mel(x)
        #print(x.shape)
        #xs = sci.resample(xs.cpu(), xs.shape[3] // 2, axis=3)
        #xs = torch.from_numpy(xs).to(self.device)
        
        
        if self.training:    
            x = self.mel_augment(x)
            #xs = self.mel_augment(xs)      
        x = (x + 1e-5).log()
        #xs = (xs + 1e-5).log() 
        #x = torch.cat((x,xs), dim=1)

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

        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.config.lr)
        #optimizer = torch.optim.RAdam(self.parameters(), lr=self.config.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=0.4,
            #num_cycles=0.2,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }


        return [optimizer], [lr_scheduler_config]
    
    #def mixup_criterion(self,criterion, pred, y_a, y_b, lam):
    #        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)    

    def training_step_old(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """

        #criterion = torch.nn.CrossEntropyLoss()

        x, files, labels, devices, cities = train_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self.device)
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config.mixstyle_p > 0:
            # frequency mixstyle
            #x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha, 1e-6)
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()
        #samples_loss = criterion(y_hat, labels.to(torch.float))#, reduction="none")
        #

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)

        return loss

    
    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: one batch from the train dataloader
        :param batch_idx: index of the batch
        :return: loss to update model parameters
        """

        x, files, labels, devices, cities = train_batch
        labels = labels.to(torch.long).to(self.device)

        # Convert raw audio to log mel spectrograms
        x = self.mel_forward(x)

        # Apply mixstyle augmentation if enabled
        if self.config.mixstyle_p > 0:
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha, eps=1e-6)

        # Forward pass through student model
        pred = self.model(x)
        ce_loss = F.cross_entropy(pred, labels, reduction="none")
        loss = ce_loss.mean()

        # Logging
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu(), prog_bar=True)

        return loss


    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):

        x, files, labels, devices, cities = val_batch

        y_hat = self.forward(x)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self.device)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        #dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        self.log_dict({"val/" + k: logs[k] for k in logs}, prog_bar=True)
        self.validation_step_outputs.clear()
        print('\n')

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self.device)

        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)
        #y_hat = model_int8(x.cpu())
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # assure fp16
        #self.model.half()

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)
        #y_hat = model_int8(x.cpu())       

        return files, y_hat
    
    def on_after_backward(self):
        # Log total gradient norm
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, prog_bar=False)

        # Log trainable parameter values
        self.log("param1", self.model.param1.item(), prog_bar=False)
        self.log("param2", self.model.param2.item(), prog_bar=False)
        self.log("param3", self.model.param3.item(), prog_bar=False)
        self.log("param4", self.model.param4.item(), prog_bar=False)



def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from dataset.ntu_dcase24_v1 import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # get pointer to h5 file containing audio samples
    #hf_in = open_h5('h5py_audio_wav')
 
    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)

    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True)
    #close_h5(hf_in)  

    # get model complexity from nessi
    sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
    shape = pl_module.mel_forward(sample).size()
    macs, params_bytes = complexity.get_torch_macs_memory(pl_module.model, input_size=shape)
    print(f"Model Complexity: MACs: {macs}, Params: {params_bytes}")

    # obtain and store details on model for reporting in the technical report
    info = {}
    info['MACs'] = macs
    info['Params'] = params_bytes
    res = trainer.test(pl_module, test_dl)
    info['test'] = res

    # generate predictions on evaluation set
    eval_dl = DataLoader(dataset=get_eval_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    predictions = trainer.predict(pl_module, dataloaders=eval_dl)
    # all filenames
    all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
    # all predictions
    all_predictions = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
    all_predictions = F.softmax(all_predictions, dim=1)

    # write eval set predictions to csv file
    df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
    le = preprocessing.LabelEncoder()
    le.fit_transform(df[['scene_label']].values.reshape(-1))
    class_names = le.classes_
    df = {'filename': all_files}
    scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
    df['scene_label'] = scene_labels
    for i, label in enumerate(class_names):
        df[label] = all_predictions[:, i]
    df = pd.DataFrame(df)

    # save eval set predictions, model state_dict and info to output folder
    df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
    torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
    with open(os.path.join(out_dir, "info.json"), "w") as json_file:
        dump(info, json_file)


if __name__ == '__main__':
    
    parser = ArgumentParser(description='DCASE 24 argument parser')

    # general
    parser.add_argument('--project_name', type=str, default="DCASE24_Task1")
    #parser.add_argument('--experiment_name', type=str, default="Baseline")
    parser.add_argument('--experiment_name', type=str, default="from crkatmmn")
    parser.add_argument('--num_workers', type=int, default=0)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")

    # evaluation
    parser.add_argument('--ckpt_id', type=str, default= None)  # for loading trained model, corresponds to wandb id

    # dataset
    # subset in {100, 50, 25, 10, 5}
    parser.add_argument('--orig_sample_rate', type=int, default=44100)
    parser.add_argument('--subset', type=int, default=25)

    # model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--in_channels', type=int, default=1)
    # adapt the complexity of the neural network (3 main dimensions to scale the baseline)
    parser.add_argument('--base_channels', type=int, default=24)
    # making chan multiplier less than 1 degrades perf
    parser.add_argument('--channels_multiplier', type=float, default=1.5)
    parser.add_argument('--expansion_rate', type=float, default=2)
    parser.add_argument('--divisor', type=float, default=5)

    # training
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.3)
    #parser.add_argument('--mixup_alpha', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    # disable as smaller splits will not have enough audio samples to take advantage of this augmentation
    parser.add_argument('--roll_sec', type=float, default=0)  # roll waveform over time,
    parser.add_argument('--dir_p', type=float, default=0.4)  #  mic impulse response

    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--warmup_steps', type=int, default=20)

    # preprocessing
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--window_length', type=int, default=8192)  # in samples (corresponds to 96 ms)
    parser.add_argument('--hop_length', type=int, default=1364)  # in samples (corresponds to ~16 ms)
    parser.add_argument('--n_fft', type=int, default=8192)  # length (points) of fft, e.g. 4096 point FFT
    parser.add_argument('--n_mels', type=int, default=256)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=0)  # mask up to 'timem' spectrogram frames
    parser.add_argument('--f_min', type=int, default=1)  # mel bins are created for freqs. between 'f_min' and 'f_max'
    parser.add_argument('--f_max', type=int, default=None) 

    args = parser.parse_args()

    evaluate(args)

