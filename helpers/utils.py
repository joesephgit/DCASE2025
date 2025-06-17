import numpy as np
import torch
from pytorch_lightning import seed_everything
from torch.distributions.beta import Beta
import random
from pytorch_lightning.callbacks import Callback
from os import environ


def mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
    return x

def mixstyle_1(x1, x2, p=0.4, alpha=0.3, eps=1e-6):
    if np.random.rand() > p:
        return x1, x2
    batch_size1 = x1.size(0)
    batch_size2 = x2.size(0)
    
    x = torch.cat((x1, x2), dim = 0)
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
    x1 = x[0:batch_size1, :, :, :]
    x2 = x[batch_size1:, :, :, :]
    return x1, x2

def ntu_mixstyle(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        beta_dist = torch.distributions.Beta(alpha, alpha)
        lam = beta_dist.sample()
    else:
        lam = 1.0

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def set_seed(seed=42):
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_everything(seed, workers=True)
    
    '''random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Ensures consistent results'''

class QuantizationCallback(Callback):
    def __init__(self):
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.model.cpu()
        pl_module.model_int8 = torch.ao.quantization.convert(pl_module.model, inplace=False)
        pl_module.model.cuda()


class QuantParamFreezeCallback(Callback):
    def __init__(self, freeze_params_epochs=4):
        self.freeze_params_epochs = freeze_params_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.max_epochs - trainer.current_epoch <= self.freeze_params_epochs:
            # for the last epochs, do the following:
            # Freeze quantizer parameters
            pl_module.model.apply(torch.ao.quantization.disable_observer)
            # Freeze batch norm mean and variance estimates
            pl_module.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)