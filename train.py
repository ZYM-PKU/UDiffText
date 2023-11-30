import os, sys
import torch
import random
import pytorch_lightning as pl

from omegaconf import OmegaConf
from dataset.dataloader import get_dataloader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import save_image

from util import *


def train():

    sys.path.append(os.getcwd())

    # torch settings
    torch.multiprocessing.set_start_method('spawn') # multiprocess mode
    torch.set_float32_matmul_precision('medium') # matrix multiply precision

    config_path = 'configs/train.yaml'
    cfgs = OmegaConf.load(config_path)

    seed = random.randint(0, 2147483647)
    seed_everything(seed, workers=True)

    dataloader = get_dataloader(cfgs)
    model = init_model(cfgs)
    model.learning_rate = cfgs.base_learning_rate

    checkpoint_callback = ModelCheckpoint(dirpath = cfgs.save_ckpt_dir, every_n_epochs = cfgs.save_ckpt_freq)

    trainer = pl.Trainer(callbacks = [checkpoint_callback], **cfgs.lightning)
    trainer.fit(model = model, train_dataloaders = dataloader)


if __name__=='__main__':

    train()


