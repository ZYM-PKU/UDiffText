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

1. Make the checkpoint directory and build the tree structure

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

2. Set training parameters in configs/train.yaml, especially the paths:

```
load_ckpt_path: ./checkpoints/pretrained/512-inpainting-ema.ckpt // Pretrained SD
model_cfg_path: ./configs/train/textdesign_sd_2.yaml // Model configuration
dataset_cfg_path: ./configs/dataset/locr.yaml // Use Laion-OCR dataset
```

### 📏 Evaluation

### 🖼️ Demo

### 🎉 Acknowledgement

### 🪬 Citation