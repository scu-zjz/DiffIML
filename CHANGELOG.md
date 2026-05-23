# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - Initial public release

### Added
- DiffIML model (`IMDLBenCo/model_zoo/diffiml/`) — diffusion-based image
  manipulation localization with mask + edge latent denoising.
- LightVAE — lightweight 1-channel VAE for mask compression, with a dedicated
  distillation training script (`train_light_vae.py`).
- Segformer-based ConditionEncoder (`mit-b2 / b3 / b4 / b5`).
- Test-Time Augmentation (horizontal flip) at inference.
- Robustness evaluation pipeline (Gaussian blur / noise / JPEG / resize).
- LightVAE upper-bound evaluation (`evaluate_light_vae.py`,
  `evaluate_light_vae_latent.py`,
  `IMDLBenCo/evaluation/eval_vae_upper_bound.py`).
- Demo training / testing shell scripts in `runs/`.
- Standard project files: `README.md`, `LICENSE`, `NOTICE`,
  `CONTRIBUTING.md`, `requirements.txt`, `.gitignore`.

### Built upon
- [IMDLBenCo](https://github.com/scu-zjz/IMDLBenCo) (CC-BY-4.0)
- All baseline models retained for fair comparison: IML-ViT, CAT-Net,
  MVSS-Net, TruFor, MantraNet, ObjectFormer, PSCC-Net, SPAN, OpenSDI,
  LatentIML, SDIML, NoiseDet.
