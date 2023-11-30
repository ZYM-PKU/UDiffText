import torch
from omegaconf import OmegaConf
from sgm.util import instantiate_from_config
from sgm.modules.diffusionmodules.sampling import *

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

def init_model(cfg):

    model_cfg = OmegaConf.load(cfg.model_cfg_path)
    ckpt = cfg.load_ckpt_path

    model = instantiate_from_config(model_cfg.model)
    model.init_from_ckpt(ckpt)

    if cfg.type == "train":
        model.train()
    else:
        model.to(torch.device("cuda", index=cfg.gpu))
        model.eval()
        model.freeze()

    return model

def init_sampling(cfgs):

    discretization_config = {
        "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
    }

    if cfgs.dual_conditioner:
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.DualCFG",
            "params": {"scale": cfgs.scale},
        }

        sampler = EulerEDMDualSampler(
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
    else:
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

    if not cfgs.dual_conditioner:
        batch_uc = deep_copy(batch)

        if "ntxt" in batch:
            batch_uc["txt"] = batch["ntxt"]
        else:
            batch_uc["txt"] = ["" for _ in range(len(batch["txt"]))]

        if "label" in batch:
            batch_uc["label"] = ["" for _ in range(len(batch["label"]))]

        return batch, batch_uc, None
    
    else:
        batch_uc_1 = deep_copy(batch)
        batch_uc_2 = deep_copy(batch)

        batch_uc_1["ref"] = torch.zeros_like(batch["ref"])
        batch_uc_2["ref"] = torch.zeros_like(batch["ref"])

        batch_uc_1["label"] = ["" for _ in range(len(batch["label"]))]

        return batch, batch_uc_1, batch_uc_2