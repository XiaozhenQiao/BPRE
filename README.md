# BPRE

Official implementation of **Bidirectional Prototype-Reward co-Evolution for Test-Time Adaptation of Vision-Language Models**.

BPRE is a test-time adaptation framework for vision-language models. It evaluates test samples with a multi-dimensional reward module and updates class prototypes through a bidirectional prototype-reward co-evolution process, without requiring source data or target labels.

## News

- Code for BPRE is released.
- The paper has been accepted by **IEEE Transactions on Multimedia, 2026**.

## Method Overview

BPRE contains two core components:

- **Multi-dimensional Quality-aware Reward Module (MQRM)**: estimates sample quality using semantic similarity, prediction confidence, and feature diversity.
- **Prototype-Reward Interactive Evolution (PRIE)**: uses reward scores to update visual prototypes and uses refined prototypes to improve reward estimation and final prediction.

During testing, BPRE dynamically maintains class-specific visual prototypes and combines CLIP logits, cache logits, and prototype residual refinement for prediction.

## Installation

```bash
git clone https://github.com/XiaozhenQiao/BPRE.git
cd BPRE

conda create -n bpre python=3.10 -y
conda activate bpre
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version from the official PyTorch website.

## Data Preparation

Please organize datasets following [docs/DATASETS.md](docs/DATASETS.md). The expected structure is:

```text
$DATA/
|-- imagenet/
|-- caltech-101/
|-- oxford_pets/
|-- stanford_cars/
|-- ...
```

The provided scripts use:

```text
/data/zhaozy/qiaoxiaozhen/data/TTA-PT
```

Please modify `DATA_ROOT` in `scripts_bpre/*.sh` or pass `--data-root` manually for your environment.

## Checkpoints

CLIP weights are loaded automatically by `clip.load`. In this codebase, the default checkpoint directory is configured in `clip/clip.py`. Please make sure the corresponding CLIP weights are available or adjust the path for your system.

## Running

Run BPRE on Caltech101 with RN50:

```bash
bash scripts_bpre/run_cd_RN50.sh
```

Run BPRE on Caltech101 with ViT-B/16:

```bash
bash scripts_bpre/run_cd_vit.sh
```

You can also run directly:

```bash
python main_bpre.py \
  --config configs \
  --data-root /path/to/datasets \
  --datasets caltech101 \
  --backbone RN50
```

Supported CLIP backbones include:

```text
RN50, ViT-B/16, SigLIP, OpenCLIP
```

## Configuration

Each dataset should have its own configuration file under `configs/`, for example:

```text
configs/caltech101.yaml
```

Important BPRE parameters include:

- `shot_capacity`: number of cached samples per class.
- `alpha`, `beta`: cache-logit scaling parameters.
- `temperature`: reward/prototype weighting temperature.
- `gamma`: prototype residual refinement weight.
- `momentum`: prototype update momentum.
- `update_interval`: prototype update interval.
- `threshold`: normalized entropy threshold for cache admission.
- `warmup_steps`, `min_reward`: reward warmup control.
- `lambda_sim`, `lambda_conf`, `lambda_div`: reward component weights.

## Citation

If you find this project useful, please cite:

```bibtex
@article{qiao2026bidirectional,
  title={Bidirectional prototype-reward co-evolution for test-time adaptation of vision-language models},
  author={Qiao, Xiaozhen and Huang, Peng and Yuan, Jiakang and Guo, Xianda and Ye, Bowen and Xue, Chaocan and Zheng, Ye and Sun, Zhe and Li, Xuelong},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```

## Contact

For questions, please contact Xiaozhen Qiao at:

```text
xiaozhennnqiao@mail.ustc.edu.cn
```

## Acknowledgements

This project builds on CLIP and follows common dataset organization protocols from prior CLIP adaptation works. We thank the authors of CLIP, CoOp/CoCoOp, TPT, TDA, and DPE for their public resources and inspiring work.
