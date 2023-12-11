import torch
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from sgm.modules.diffusionmodules.sampling import *


def init_model(cfgs):

    model_cfg = OmegaConf.load(cfgs.model_cfg_path)
    ckpt = cfgs.load_ckpt_path

    model = instantiate_from_config(model_cfg.model)
    model.init_from_ckpt(ckpt)

    if cfgs.type == "train":
        model.train()
    else:
        model.to(torch.device("cuda", index=cfgs.gpu))
        model.eval()
        model.freeze()

    return model

def init_sampling(cfgs):

    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    }

    guider_config = {
        "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
        "params": {"scale": cfgs.scale[0]},
    }

    sampler = EulerEDMSampler(
        num_steps=cfgs.steps,
        discretization_config=discretization_config,
        guider_config=guider_config,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=999.0,
        s_noise=1.0,
        verbose=True,
        device=torch.device("cuda", index=cfgs.gpu)
    )

    return sampler

def deep_copy(batch):

    c_batch = {}
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            c_batch[key] = torch.clone(batch[key])
        elif isinstance(batch[key], (tuple, list)): 
            c_batch[key] = batch[key].copy()
        else:
            c_batch[key] = batch[key]
    
    return c_batch

def prepare_batch(cfgs, batch):

    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(torch.device("cuda", index=cfgs.gpu))

    batch_uc = deep_copy(batch)

    if "ntxt" in batch:
        batch_uc["txt"] = batch["ntxt"]
    else:
        batch_uc["txt"] = ["" for _ in range(len(batch["txt"]))]

    if "label" in batch:
        batch_uc["label"] = ["" for _ in range(len(batch["label"]))]

    return batch, batch_uc