import cv2
import torch
import os, glob
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from os.path import join as ospj

from util import *


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


def demo_predict(input_blk, text, num_samples, steps, scale, seed, show_detail):

    global cfgs, global_index

    global_index += 1

    if num_samples > 1: cfgs.noise_iters = 0

    cfgs.batch_size = num_samples
    cfgs.steps = steps
    cfgs.scale[0] = scale
    cfgs.detailed = show_detail
    seed_everything(seed)

    sampler = init_sampling(cfgs)

    image = input_blk["image"]
    mask = input_blk["mask"]
    image = cv2.resize(image, (cfgs.W, cfgs.H))
    mask = cv2.resize(mask, (cfgs.W, cfgs.H))

    mask = (mask == 0).astype(np.int32)

    image = torch.from_numpy(image.transpose(2,0,1)).to(dtype=torch.float32) / 127.5 - 1.0
    mask = torch.from_numpy(mask.transpose(2,0,1)).to(dtype=torch.float32).mean(dim=0, keepdim=True)
    masked = image * mask
    mask = 1 - mask

    seg_mask = torch.cat((torch.ones(len(text)), torch.zeros(cfgs.seq_len-len(text))))

    # additional cond
    txt = f"\"{text}\""
    original_size_as_tuple = torch.tensor((cfgs.H, cfgs.W))
    crop_coords_top_left = torch.tensor((0, 0))
    target_size_as_tuple = torch.tensor((cfgs.H, cfgs.W))

    image = torch.tile(image[None], (num_samples, 1, 1, 1))
    mask = torch.tile(mask[None], (num_samples, 1, 1, 1))
    masked = torch.tile(masked[None], (num_samples, 1, 1, 1))
    seg_mask = torch.tile(seg_mask[None], (num_samples, 1))
    original_size_as_tuple = torch.tile(original_size_as_tuple[None], (num_samples, 1))
    crop_coords_top_left = torch.tile(crop_coords_top_left[None], (num_samples, 1))
    target_size_as_tuple = torch.tile(target_size_as_tuple[None], (num_samples, 1))

    text = [text for i in range(num_samples)]
    txt = [txt for i in range(num_samples)]
    name = [str(global_index) for i in range(num_samples)]

    batch = {
        "image": image,
        "mask": mask,
        "masked": masked,
        "seg_mask": seg_mask,
        "label": text,
        "txt": txt,
        "original_size_as_tuple": original_size_as_tuple,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size_as_tuple": target_size_as_tuple,
        "name": name
    }

    samples, samples_z = predict(cfgs, model, sampler, batch)
    samples = samples.cpu().numpy().transpose(0, 2, 3, 1) * 255
    results = [Image.fromarray(sample.astype(np.uint8)) for sample in samples]

    if cfgs.detailed:
        sections = []
        attn_map = Image.open(f"./temp/attn_map/attn_map_{global_index}.png")
        seg_maps = np.load(f"./temp/seg_map/seg_{global_index}.npy")
        for i, seg_map in enumerate(seg_maps):
            seg_map = cv2.resize(seg_map, (cfgs.W, cfgs.H))
            sections.append((seg_map, text[0][i]))
        seg = (results[0], sections)
    else:
        attn_map = None
        seg = None

    return results, attn_map, seg


if __name__ == "__main__":

    os.makedirs("./temp", exist_ok=True)
    os.makedirs("./temp/attn_map", exist_ok=True)
    os.makedirs("./temp/seg_map", exist_ok=True)

    cfgs = OmegaConf.load("./configs/demo.yaml")

    model = init_model(cfgs)
    global_index = 0

    block = gr.Blocks().queue()
    with block:

        with gr.Row():

            gr.HTML(
                """
                <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
                <h1 style="font-weight: 600; font-size: 2rem; margin: 0.5rem;">
                    UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models
                </h1>        
                <ul style="text-align: center; margin: 0.5rem;"> 
                    <li style="display: inline-block; margin:auto;"><a href='https://arxiv.org/abs/2312.04884'><img src='https://img.shields.io/badge/Arxiv-2312.04884-DF826C'></a></li>
                    <li style="display: inline-block; margin:auto;"><a href='https://github.com/ZYM-PKU/UDiffText'><img src='https://img.shields.io/badge/Code-UDiffText-D0F288'></a></li>
                    <li style="display: inline-block; margin:auto;"><a href='https://udifftext.github.io'><img src='https://img.shields.io/badge/Project-UDiffText-8ADAB2'></a></li>
                </ul> 
                <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin: 0.5rem;">
                    Our proposed UDiffText is capable of synthesizing accurate and harmonious text in either synthetic or real-word images, thus can be applied to tasks like scene text editing (a), arbitrary text generation (b) and accurate T2I generation (c)
                </h2>
                <div align=center><img src="file/demo/teaser.png" alt="UDiffText" width="80%"></div> 
                </div>
                """
            )

        with gr.Row():

            with gr.Column():

                input_blk = gr.Image(source='upload', tool='sketch', type="numpy", label="Input", height=512)
                text = gr.Textbox(label="Text to render:", info="the text you want to render at the masked region")
                run_button = gr.Button(variant="primary")

                with gr.Accordion("Advanced options", open=False):

                    num_samples = gr.Slider(label="Images", info="number of generated images, locked as 1", minimum=1, maximum=1, value=1, step=1)
                    steps = gr.Slider(label="Steps", info ="denoising sampling steps", minimum=1, maximum=200, value=50, step=1)
                    scale = gr.Slider(label="Guidance Scale", info="the scale of classifier-free guidance (CFG)", minimum=0.0, maximum=10.0, value=4.0, step=0.1)
                    seed = gr.Slider(label="Seed", info="random seed for noise initialization", minimum=0, maximum=2147483647, step=1, randomize=True)
                    show_detail = gr.Checkbox(label="Show Detail", info="show the additional visualization results", value=False)

            with gr.Column():

                gallery = gr.Gallery(label="Output", height=512, preview=True)

                with gr.Accordion("Visualization results", open=True):

                    with gr.Tab(label="Attention Maps"):
                        gr.Markdown("### Attention maps for each character (extracted from middle blocks at intermediate sampling step):")
                        attn_map = gr.Image(show_label=False, show_download_button=False)
                    with gr.Tab(label="Segmentation Maps"):
                        gr.Markdown("### Character-level segmentation maps (using upscaled attention maps):")
                        seg_map = gr.AnnotatedImage(height=384, show_label=False)

        # examples
        examples = []
        example_paths = sorted(glob.glob(ospj("./demo/examples", "*")))
        for example_path in example_paths:
            label = example_path.split(os.sep)[-1].split(".")[0].split("_")[0]
            examples.append([example_path, label])

        gr.Markdown("## Examples:")
        gr.Examples(
            examples=examples,
            inputs=[input_blk, text]
        )

        run_button.click(fn=demo_predict, inputs=[input_blk, text, num_samples, steps, scale, seed, show_detail], outputs=[gallery, attn_map, seg_map])

    block.launch()