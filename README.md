<div align="center">

# DiffIML: Diffusion-based Image Manipulation Localization

<p align="center">
  <a href="#-introduction">Introduction</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-training">Training</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-robustness-test">Robustness</a> •
  <a href="#-citation">Citation</a>
</p>

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/%F0%9F%A4%97-Diffusers-yellow)](https://github.com/huggingface/diffusers)
[![Built on IMDLBenCo](https://img.shields.io/badge/built%20on-IMDLBenCo-2ea44f)](https://github.com/scu-zjz/IMDLBenCo)

</div>

---

## 📖 Introduction

**DiffIML** is a novel framework for **Image Manipulation Localization (IML)** that re-formulates pixel-level forgery localization as a **conditional latent diffusion** task. Instead of directly regressing the binary tampering mask in the pixel space, DiffIML:

1. Compresses the binary **mask** *and* the **edge mask** into a compact latent space using a custom **LightVAE** (a slimmed, distilled variant of the Stable-Diffusion VAE that operates on 1-channel masks).
2. Uses a **Segformer-based Condition Encoder** to extract multi-scale RGB features as the diffusion condition.
3. Trains a **conditional UNet** with a **DDIM** scheduler in the joint *(mask, edge)* latent space.
4. At inference time, runs a small number of DDIM steps with **Test-Time Augmentation (horizontal flip)** to produce both the localization mask and an auxiliary edge prediction.

The whole codebase is built on top of the [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) benchmark, which provides a unified training / evaluation / robustness-test interface for image-forensics models.

<div align="center">

| Component | Role |
|:---:|:---|
| `LightVAE` | Lightweight 1-channel VAE for mask & edge compression |
| `ConditionEncoder` | Multi-scale Segformer (mit-b2/b3/b4/b5) for image features |
| `UNet (diffusers)` | Conditional denoising network in latent space |
| `DDIMScheduler` | Fast deterministic sampling at inference |

</div>

---

## 📁 Project Structure

```
DiffIML/
├── IMDLBenCo/                       # core codebase (forked from IMDLBenCo)
│   ├── model_zoo/
│   │   ├── diffiml/                 # ⭐ DiffIML model
│   │   │   ├── diffiml.py           # DiffIML / LightVAE / SlimVAE / VAE
│   │   │   ├── segformer.py         # Segformer (mit-b2 ~ mit-b5) backbone
│   │   │   └── resnet.py
│   │   ├── iml_vit/  cat_net/  mvss_net/  trufor/ ...   # baseline models
│   ├── datasets/                    # ManiDataset / JsonDataset / BalancedDataset
│   ├── evaluation/                  # PixelF1 / ImageF1 / VAE upper bound
│   ├── training_scripts/
│   │   ├── train.py                 # main training entry
│   │   ├── test.py                  # main testing entry
│   │   ├── train_light_vae.py       # ⭐ LightVAE distillation
│   │   ├── test_robust.py           # robustness evaluation
│   │   └── gen_robustness_image.py
│   ├── transforms/   modules/   utils/
│   └── registry.py
├── runs/                            # training / testing shell scripts
│   ├── demo_train_diffiml.sh        # ⭐ train DiffIML
│   ├── demo_train_light_vae.sh      # ⭐ train LightVAE
│   ├── demo_test_diffiml.sh         # ⭐ test DiffIML
│   ├── demo_test_robustness_diffiml.sh
│   ├── balanced_dataset.json        # training dataset config (template)
│   └── test_datasets.json           # testing dataset config (template)
├── tests/                           # unit tests
├── configs/                         # extra configs
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-name>/DiffIML.git
cd DiffIML
```

### 2. Create a conda environment

```bash
conda create -n diffiml python=3.10 -y
conda activate diffiml
```

### 3. Install PyTorch (choose your CUDA version)

```bash
# example: CUDA 11.8
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install other dependencies

```bash
pip install -r requirements.txt
pip install -e .                       # install IMDLBenCo in editable mode
pip install "numpy<2"
```

### 5. (Optional) HuggingFace mirror (China)

If you cannot reach HuggingFace, set the mirror endpoint:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 📦 Datasets & Pretrained Weights

### Datasets

DiffIML follows the standard IMDLBenCo / CAT-Net protocol. The training set is a **balanced** mixture of:

| Dataset | Type | Source |
|---|---|---|
| CASIAv2 | splicing / copy-move | [CASIA](https://github.com/namtpham/casia2groundtruth) |
| FantasticReality | splicing | [FantasticReality](http://zoi.utia.cas.cz/files/Fantastic_Reality.tar.gz) |
| IMD2020 | real-world tampering | [IMD2020](http://staff.utia.cas.cz/novozada/db/) |
| tampCOCO (sp / cm / bcm / bcmc) | synthetic | [CAT-Net](https://github.com/mjkwon2021/CAT-Net) |
| compRAISE | compressed RAISE | [CAT-Net](https://github.com/mjkwon2021/CAT-Net) |

Test sets used in our paper:

| Dataset | Used for |
|---|---|
| CASIAv1 | in-domain F1 |
| Columbia | cross-domain |
| Coverage | copy-move |
| NIST16 | post-processing robust |
| Autosplice / CocoGlide | AI-generated tampering |

Configure dataset paths in:
- `runs/balanced_dataset.json` &nbsp;— training mixture
- `runs/test_datasets.json` &nbsp;— testing benchmarks

> ⚠️ **Replace** the example absolute paths (e.g. `/mnt/data0/public_datasets/IML/...`) with **your local paths**.

### Pretrained Weights

| Weight | Description | How to obtain |
|---|---|---|
| `mit_b3.pth` | Segformer ImageNet pretrain | Download from [Segformer official repo](https://github.com/NVlabs/SegFormer) and put it under `IMDLBenCo/model_zoo/diffiml/` |
| `light_vae_weights.pth` | LightVAE (mask + edge compressor) | Train it via `runs/demo_train_light_vae.sh` (see below) **or** download from the [Releases](#) page |
| `diffiml_best.pth` | Final DiffIML checkpoint | Released after training, see [Releases](#) |

---

## 🚂 Training

DiffIML training is a **two-stage** pipeline.

### Stage 1 — Train LightVAE (mask compressor)

```bash
bash runs/demo_train_light_vae.sh
```

Key arguments (edit inside the `.sh`):
```bash
DATA_PATH="runs/balanced_dataset.json"      # training data
LATENT_DIM=4
BASE_CHANNELS=32
NORM_LAYER="BatchNorm"
LATENT_WEIGHT=0.01                          # latent-alignment loss weight
EPOCHS=40
BATCH_SIZE=8
LR=1e-4
```

The trained LightVAE will be saved to:
```
log/train_light_vae/checkpoints/light_vae_weights_new.pth
```

### Stage 2 — Train DiffIML

```bash
bash runs/demo_train_diffiml.sh
```

Key arguments:
```bash
--model DiffIML
--backbone segformer_b3        # segformer_b2 / b3 / b4 / b5
--prior_rate 0.1               # mask-prior noise injection
--seg_weight 0.2               # mask vs edge loss balance
--num_inference_steps 8        # DDIM steps
--infer_time 5                 # number of stochastic samples (averaged)
--image_size 512
--batch_size 16
--epochs 60
--data_path runs/balanced_dataset.json
--test_data_path /your/path/to/CASIA1.0
```

The model is trained with **DDP** (default 8 GPUs). Adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` as needed.

---

## 🧪 Evaluation

### Pixel-level F1 on multiple datasets

```bash
bash runs/demo_test_diffiml.sh
```

Edit `runs/test_datasets.json` to point to your local benchmark folders, then:

```bash
--checkpoint_path /path/to/your/diffiml_checkpoint_dir
--test_data_json  ./runs/test_datasets.json
--test_batch_size 16
--image_size 512
--infer_time 5
--num_inference_steps 8
```

The script will iterate over all datasets defined in the JSON and report:
- **PixelF1** (forgery localization)
- **ImageF1** (image-level detection, optional)

### LightVAE upper-bound evaluation

To check how much the LightVAE bottleneck constrains the mask quality:

```bash
python evaluate_light_vae.py
python evaluate_light_vae_latent.py
# or, framework-style:
python -m IMDLBenCo.evaluation.eval_vae_upper_bound
```

---

## 🛡 Robustness Test

We follow the standard IMDLBenCo robustness protocol (Gaussian blur, JPEG compression, noise, resize…).

```bash
bash runs/demo_test_robustness_diffiml.sh
```

Generated noisy images and metrics are stored under `log/robust_diffiml/`.

---

## 📊 Results (reference)

> Numbers below are reproduced under the *CAT-Net protocol* (image_size = 512, batch = 16, 8×A100).

| Method | CASIAv1 | Columbia | Coverage | NIST16 | Autosplice | CocoGlide |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MVSS-Net | 0.452 | 0.638 | 0.453 | 0.292 | 0.553 | 0.409 |
| CAT-Net | 0.711 | 0.804 | 0.345 | 0.302 | 0.694 | 0.511 |
| TruFor | 0.737 | 0.847 | 0.521 | 0.394 | 0.799 | 0.523 |
| **DiffIML (ours)** | **—** | **—** | **—** | **—** | **—** | **—** |

> The exact numbers will be filled in upon paper release.

---

## 🏗 Built on IMDLBenCo

This project is built on top of the wonderful **[IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo)** benchmark by SCU-ZJZ Lab. All baseline models (`IML-ViT`, `CAT-Net`, `MVSS-Net`, `TruFor`, `MantraNet`, `ObjectFormer`, `PSCC-Net`, `SPAN`, `OpenSDI`, `LatentIML`, `SDIML`, `NoiseDet` …) are kept inside `IMDLBenCo/model_zoo/` for fair comparison and easy ablation.

We thank the authors for releasing such a clean and extensible codebase.

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@misc{diffiml2025,
  title  = {DiffIML: Diffusion-based Image Manipulation Localization with Mask Latent Modeling},
  author = {Your Name and Collaborators},
  year   = {2025},
  note   = {Code: https://github.com/<your-name>/DiffIML}
}
```

And please also cite the underlying benchmark:

```bibtex
@inproceedings{ma2024imdlbenco,
  title     = {IMDL-BenCo: A Comprehensive Benchmark and Codebase for Image Manipulation Detection \& Localization},
  author    = {Ma, Xiaochen and Zhu, Xuekang and Su, Lei and Jiang, Zhuohang and Du, Bo and Wang, Xiwen and Lei, Zeyu and Feng, Wentao and Pun, Chi-Man and Zhou, Ji-Zhe},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2024}
}
```

---

## 🤝 Acknowledgements

- [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) — benchmark backbone we build on
- [Mesorch](https://github.com/scu-zjz/Mesorch) — inspiration for the README layout
- [diffusers](https://github.com/huggingface/diffusers) — UNet / DDIM implementation
- [Segformer](https://github.com/NVlabs/SegFormer) — condition encoder
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) — teacher VAE for distillation

---

## 📝 License

This project is released under the **CC-BY-4.0** license, the same as [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo). See [LICENSE](LICENSE) for details.
