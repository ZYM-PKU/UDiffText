import torch
import torch.utils.data as data
import pytorch_lightning as pl
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from pytorch_lightning.callbacks import ModelCheckpoint


def get_dataloader(cfgs):

    dataset = instantiate_from_config(cfgs.dataset)
    dataloader = data.DataLoader(dataset=dataset, batch_size=cfgs.batch_size, shuffle=False, num_workers=cfgs.num_workers)

    return dataloader

def get_model(cfgs):

    model = instantiate_from_config(cfgs.model)
    if "load_ckpt_path" in cfgs:
        model.load_state_dict(torch.load(cfgs.load_ckpt_path, map_location="cpu")["state_dict"], strict=False)

    return model

def train(cfgs):

    dataloader = get_dataloader(cfgs)
    model = get_model(cfgs)

    checkpoint_callback = ModelCheckpoint(dirpath = cfgs.ckpt_dir, every_n_epochs = cfgs.check_freq)

    trainer = pl.Trainer(callbacks = [checkpoint_callback], **cfgs.lightning)
    trainer.fit(model = model, train_dataloaders = dataloader)

    
if __name__ == "__main__":

    config_path = 'configs/pretrain.yaml'
    cfgs = OmegaConf.load(config_path)
    train(cfgs)