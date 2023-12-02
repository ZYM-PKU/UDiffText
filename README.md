## UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models

<a href='https://arxiv.org/pdf/******'><img src='https://img.shields.io/badge/Arxiv-******-DF826C'></a> 
<a href='https://github.com/ZYM-PKU/UDiffText'><img src='https://img.shields.io/badge/Code-UDiffText-D0F288'></a> 
<a href='https://huggingface.co/spaces/******'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-UDiffText-8ADAB2'></a> 

#### Our proposed UDiffText is capable of synthesizing accurate and harmonious text in either synthetic or real-word images, thus can be applied to tasks like scene text editing (a), arbitrary text generation (b) and accurate T2I generation (c)

![UDiffText Teaser](demo/teaser.png)

### 📬 News

**2023.11.30** Upload version 1.0

### 🔨 Installation

1. Clone this repo: 
```
git clone https://github.com/ZYM-PKU/UDiffText.git
cd UDiffText
```

2. Install required Python packages

```
conda create -n udiff python=3.11
conda activate udiff
pip install -r requirements.txt
```

3. Make the checkpoint directory and build the tree structure

```
mkdir ./checkpoints

checkpoints
├── AEs                    // AutoEncoder
├── encoders             
    ├── LabelEncoder       // Character-level encoder
    └── ViTSTR             // STR encoder
├── predictors             // STR model
├── pretrained             // Pretrained SD
└── ***.ckpt               // UDiffText checkpoint
```

### 💻 Training

1. Prepare your data

- Create a data directory **{your data root}** in your disk and put your data in it.
- For the downloading and preprocessing of Laion-OCR dataset, please refer to [TextDiffuser](https://github.com/microsoft/unilm/tree/master/textdiffuser) and our **./scripts/preprocess/laion_ocr_pre.ipynb**.

2. Train the character-level encoder

Set the parameters in **./configs/pretrain.yaml** and run:

```
python pretrain.py
```

3. Train the UDiffText model

Set the parameters in **./configs/train.yaml**, especially the paths:

```
load_ckpt_path: ./checkpoints/pretrained/512-inpainting-ema.ckpt // Checkpoint of the pretrained SD
model_cfg_path: ./configs/train/textdesign_sd_2.yaml // UDiffText model config
dataset_cfg_path: ./configs/dataset/locr.yaml // Use the Laion-OCR dataset
```

and run:

```
python train.py
```

### 📏 Evaluation

1. Download our available [checkpoints]() and put them in the corresponding directories in **./checkpoints**.

2. Set the parameters in **./configs/test.yaml**, especially the paths:

```
load_ckpt_path: "./checkpoints/***.ckpt"  // Downloaded UDiffText checkpoint
model_cfg_path: "./configs/test/textdesign_sd_2.yaml"  // UDiffText model config
dataset_cfg_path: "./configs/dataset/locr.yaml"  // Use the Laion-OCR dataset
```

and run:

```
python test.py
```

### 🖼️ Demo


### 🎉 Acknowledgement

### 🪬 Citation