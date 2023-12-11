import torch
import random
import numpy as np
import os

from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext
from os.path import join as ospj
from torchvision.utils import save_image
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from dataset.dataloader import get_dataloader

from util import *
from metrics import calc_fid, calc_lpips


def predict(cfgs, model, sampler, batch):

    context = nullcontext if cfgs.aae_enabled else torch.no_grad
    
    with context():
        
        batch, batch_uc_1 = prepare_batch(cfgs, batch)

        c, uc_1 = model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc_1,
            force_uc_zero_embeddings=cfgs.force_uc_zero_embeddings,
        )
        
        x = sampler.get_init_noise(cfgs, model, cond=c, batch=batch, uc=uc_1)
        samples_z = sampler(model, x, cond=c, batch=batch, uc=uc_1, init_step=0,
                            aae_enabled = cfgs.aae_enabled, detailed = cfgs.detailed)

        samples_x = model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        return samples, samples_z


def test(model, sampler, dataloader, cfgs):
    
    output_dir = cfgs.output_dir
    os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    real_dir = ospj(output_dir, "real")
    fake_dir = ospj(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    temp_dir = cfgs.temp_dir
    os.system(f"rm -rf {temp_dir}")
    os.makedirs(ospj(temp_dir, "attn_map"), exist_ok=True)
    os.makedirs(ospj(temp_dir, "seg_map"), exist_ok=True)
    os.makedirs(ospj(temp_dir, "inters"), exist_ok=True)

    if cfgs.ocr_enabled:
        predictor = instantiate_from_config(cfgs.predictor_config)
        predictor.parseq = predictor.parseq.to(sampler.device)

        correct_num = 0
        total_num = 0

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        if idx >= cfgs.max_iter: break

        name = batch["name"][0]
        results, results_z = predict(cfgs, model, sampler, batch)

        # run ocr
        if cfgs.ocr_enabled:
            
            r_bbox = batch["r_bbox"]
            gt_txt = batch["label"]
            results_crop = []
            for i, bbox in enumerate(r_bbox):
                r_top, r_bottom, r_left, r_right = bbox
                results_crop.append(results[i, :, r_top:r_bottom, r_left:r_right])
            pred_txt = predictor.img2txt(results_crop)

            correct_count = sum([int(pred_txt[i].lower()==gt_txt[i].lower()) for i in range(len(gt_txt))])
            print(f"Expected text: {batch['label']}")
            if correct_count < len(gt_txt):
                print(f"\033[1;31m OCR Result: {pred_txt} \033[0m")
            else:
                print(f"\033[1;32m OCR Result: {pred_txt} \033[0m")
            correct_num += correct_count
            total_num += len(gt_txt)
        
        # save results
        result = results.cpu().numpy().transpose(0, 2, 3, 1) * 255
        result = np.concatenate(result, axis = -2)

        outputs = []
        for key in ("image", "masked", "mask"):
            if key in batch:
                output = batch[key]
                if key != "mask":
                    output = (output + 1.0) / 2.0
                output = output.cpu().numpy().transpose(0, 2, 3, 1) * 255
                output = np.concatenate(output, axis = -2)
                if key == "mask":
                    output = np.tile(output, (1,1,3))
                outputs.append(output)

        outputs.append(result)
        real = Image.fromarray(outputs[0].astype(np.uint8))
        fake = Image.fromarray(outputs[-1].astype(np.uint8))
        real.save(ospj(output_dir, "real", f"{name}.png"))
        fake.save(ospj(output_dir, "fake", f"{name}.png"))

        output = np.concatenate(outputs, axis = 0)
        output = Image.fromarray(output.astype(np.uint8))
        output.save(ospj(output_dir, f"{name}.png"))

    if cfgs.ocr_enabled:
        print(f"OCR test completed. Mean accuracy: {correct_num/total_num}")
    
    if cfgs.quan_test:
        calc_fid(fake_dir, real_dir)
        calc_lpips(fake_dir, real_dir)


if __name__ == "__main__":

    cfgs = OmegaConf.load("./configs/test.yaml")

    seed = random.randint(0, 2147483647)
    seed_everything(seed)

    model = init_model(cfgs)
    sampler = init_sampling(cfgs)
    dataloader = get_dataloader(cfgs, "val")

    test(model, sampler, dataloader, cfgs)

