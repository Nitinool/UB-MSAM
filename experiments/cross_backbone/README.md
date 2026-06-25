# BIBM E3: Cross-backbone 实验

> 回应 R2.2 / R3.1：证明 U-BLoss 是 backbone-agnostic 的训练目标，不依赖 SAM2 特定架构。

## 📁 文件结构

```
experiments/cross_backbone/
├── common.py              # 共享: 数据集/模型/loss/指标/训练循环
├── train_scratch.py       # MONAI from scratch (无预训练)
├── train_pretrained.py    # smp + ImageNet 预训练 encoder
├── eval.py                # 统一评测 (任意 ckpt × 任意 dataset)
└── README.md              # 本文件
```

## 📦 依赖

```bash
# 两个版本都要:
pip install einops medpy pandas tqdm

# train_scratch.py:
pip install monai

# train_pretrained.py:
pip install segmentation_models_pytorch timm
```

## 🎯 模型对比

| 脚本 | U-Net | Swin-UNet | lr | epochs |
|---|---|---|---|---|
| `train_scratch.py` | MONAI UNet (from scratch) | MONAI SwinUNETR (from scratch) | 1e-3 | 150 |
| `train_pretrained.py` | smp.Unet + ResNet34 (ImageNet) | smp.Unet + SwinV2 (ImageNet) | 1e-4 | 100 |

## 🚀 训练命令

### Pretrained 版本（推荐，baseline 接近论文 77.83% / 81.18%）

```bash
cd ~/jupyterworkspace/UB-MSAM

# BUSI 4 个实验 (4 卡并行)
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train_pretrained.py --backbone unet       --dataset BUSI --exp-name bibm_e3_busi_unet_pre_baseline" > logs/e3_busi_unet_pre_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python experiments/cross_backbone/train_pretrained.py --backbone unet --use-ubl --dataset BUSI --exp-name bibm_e3_busi_unet_pre_ubl"     > logs/e3_busi_unet_pre_ubl.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python experiments/cross_backbone/train_pretrained.py --backbone swin_unetr --dataset BUSI --exp-name bibm_e3_busi_swin_pre_baseline"     > logs/e3_busi_swin_pre_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python experiments/cross_backbone/train_pretrained.py --backbone swin_unetr --use-ubl --dataset BUSI --exp-name bibm_e3_busi_swin_pre_ubl" > logs/e3_busi_swin_pre_ubl.log 2>&1 &

# ISIC 4 个实验 (BUSI 跑完后)
# 把 --dataset BUSI 换成 --dataset ISIC, exp-name 里的 busi 换成 isic

# CVC 同理
```

### Scratch 版本（对照实验，证明"预训练不是 U-BLoss 起作用的原因"）

```bash
# 同样命令, 换 train_pretrained.py → train_scratch.py, exp-name 加 _scratch
CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train_scratch.py \
    --backbone unet --use-ubl --dataset BUSI --exp-name bibm_e3_busi_unet_scratch_ubl
```

### DDP 多卡（单实验加速）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    experiments/cross_backbone/train_pretrained.py \
    --backbone unet --use-ubl --dataset BUSI --exp-name bibm_e3_busi_unet_pre_ubl
```

## 📊 评测

```bash
# 评测任意 ckpt 在任意 dataset
python experiments/cross_backbone/eval.py \
    --ckpt runs/bibm_e3_busi_unet_pre_ubl/checkpoints/checkpoint.pt \
    --dataset BUSI \
    --output results/bibm_e3_busi_unet_pre_ubl.json

# 评测 ISIC 上的 ckpt
python experiments/cross_backbone/eval.py \
    --ckpt runs/bibm_e3_isic_swin_pre_ubl/checkpoints/checkpoint.pt \
    --dataset ISIC
```

**eval.py 会自动从 ckpt 里读 backbone 和 model_type**，不用手动指定。
- `--split val`（默认）= 用 val.txt 当 held-out test
- `--output xxx.json` = 同时存 json 方便填论文

## 📋 完整实验矩阵（如果全跑）

```
2 模型版本 (scratch, pretrained)
× 2 backbone (unet, swin_unetr)
× 2 设置 (baseline, +UBL)
× 3 数据集 (BUSI, ISIC, CVC)
= 24 个实验
```

**建议优先级**：
1. 🔴 pretrained × BUSI 4 个（最关键，回应 R2.2）
2. 🟡 pretrained × ISIC 4 个 + pretrained × CVC 4 个（多数据集验证）
3. 🟢 scratch × BUSI 4 个（对照，证明预训练不是关键）

## 📁 数据集路径

默认在 `common.py` 的 `DATASET_ROOTS`：
```python
BUSI:   /home/zhengsongming/jupyterworkspace/datasets/BUSI_for_SAM2
ISIC:   /home/zhengsongming/jupyterworkspace/datasets/ISIC_for_SAM2
CVC:    /home/zhengsongming/jupyterworkspace/datasets/CVC_for_SAM2
Kvasir: /home/zhengsongming/jupyterworkspace/datasets/Kvasir_for_SAM2
```

不一致时用 `--dataset-root /your/path` override。

## ⚙️ 训练超参

| 超参 | scratch | pretrained |
|---|---|---|
| batch_size | 8 | 8 |
| lr | 1e-3 | 1e-4 |
| epochs | 150 | 100 |
| weight_decay | 0.01 | 0.01 |
| size | 1024 | 1024 |
| ubl_weight | 2.0 | 2.0 |
| pos_weight (BCE) | 7.0 | 7.0 |
| scheduler | warmup(10%) + cosine | warmup(10%) + cosine |

## ⚠️ 常见坑

1. **SwinV2 OOM**: 1024×1024 batch=8 显存吃紧 → `--batch-size 4`
2. **scratch 训练慢**: 150 epoch from-scratch，BUSI 单卡 ~5h
3. **pretrained 第一次跑要下载 ImageNet 权重**: ResNet34 ~85MB, SwinV2 ~87MB
4. **eval.py 读 ckpt 报错**: 可能 model_type 字段没有，老 ckpt 兼容性 — 告诉 Claude 加 fallback
