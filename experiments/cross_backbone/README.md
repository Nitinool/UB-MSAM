# BIBM E3: Cross-backbone 实验

> 回应 R2.2 / R3.1：证明 U-BLoss 是 backbone-agnostic 的训练目标，不依赖 SAM2 特定架构。
> Table 4b 下半 4 行：U-Net baseline / U-Net + UBL / Swin-UNETR baseline / Swin-UNETR + UBL

---

## 📦 依赖

只需要装一个：
```bash
pip install monai
```

MONAI 内置 UNet 和 SwinUNETR 实现。我们用 MONAI 是为了：
- ✅ 一行代码就能拿到 SOTA-quality 的实现
- ✅ 同一个接口跑两种 backbone
- ✅ 论文里写法："U-Net (MONAI implementation)"

---

## 🚀 4 个实验命令

```bash
cd ~/jupyterworkspace/UB-MSAM
conda activate sam22

# 1. U-Net baseline (无 U-BLoss)
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/cross_backbone/train.py \
    --backbone unet \
    --exp-name bibm_e3_unet_baseline

# 2. U-Net + U-BLoss
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/cross_backbone/train.py \
    --backbone unet --use-ubl \
    --exp-name bibm_e3_unet_ubl

# 3. Swin-UNETR baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/cross_backbone/train.py \
    --backbone swin_unetr \
    --exp-name bibm_e3_swin_baseline

# 4. Swin-UNETR + U-BLoss
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/cross_backbone/train.py \
    --backbone swin_unetr --use-ubl \
    --exp-name bibm_e3_swin_ubl
```

> 注：当前 `train.py` 只用单卡（CUDA_VISIBLE_DEVICES=0,1,2,3 只是告诉 PyTorch "可见这 4 张"，实际只用 0 号）。如果需要 DDP 多卡训练，告诉 Claude 扩。但 CNN 训练单卡 V100 已经够快了（BUSI 50 epoch ~2h）。

---

## 📊 评测

跑完每个实验后，用 `eval_template_cnn.ipynb` 评测：

```bash
cd ~/jupyterworkspace/UB-MSAM/baseline/dataset_BUSI/
cp eval_template_cnn.ipynb eval_bibm_e3_unet_baseline.ipynb
```

打开 `eval_bibm_e3_unet_baseline.ipynb`，改两行：
```python
BACKBONE = 'unet'         # 或 'swin_unetr'
CKPT_PATH = '/home/zhengsongming/jupyterworkspace/UB-MSAM/runs/bibm_e3_unet_baseline/checkpoints/checkpoint.pt'
```

跑 → 拿到 final_summary 表。

---

## 📋 期望的实验结果（救火 narrative）

如果 narrative 立得住，应该看到：

| Backbone | Setting | Dice (期望) | Δ |
|---|---|---|---|
| U-Net | baseline | ~77.83% | -- |
| U-Net | **+ U-BLoss** | ~80-82% | **+2-4%** ✅ |
| Swin-UNETR | baseline | ~81.18% | -- |
| Swin-UNETR | **+ U-BLoss** | ~83-85% | **+2-4%** ✅ |

> 论文 Table 1 里 U-Net baseline=77.83%, Swin-Unet=81.18%，这里的 baseline 应该差不多（不会完全一样，因为框架和超参不同）。**关键是 +U-BLoss 后有 gain**，无论绝对值如何。

如果 gain 不明显甚至变差：
- 立刻告诉 Claude，可能需要调 `--ubl-weight`
- 或者降低 lr (1e-4 → 5e-5)
- 不排除 U-BLoss 真的对 CNN 不适用（这种情况下我们要调整 narrative 描述）

---

## ⚙️ 训练超参（已写在 train.py 默认值）

| 超参 | 值 |
|---|---|
| batch_size | 4 |
| lr | 1e-4 (cosine decay) |
| epochs | 50 |
| weight_decay | 0.01 |
| size | 1024 |
| ubl_weight | 2.0 |
| seed | 42 |

如需调，直接命令行 override，例如：
```bash
python experiments/cross_backbone/train.py --backbone unet --use-ubl \
    --ubl-weight 1.0 --lr 5e-5 --exp-name bibm_e3_unet_ubl_v2
```

---

## ⚠️ 可能的坑

1. **MONAI 没装**: `pip install monai` 解决
2. **OOM**: SwinUNETR 在 1024×1024 时显存吃紧。如果 V100 32GB 不够，降 `--batch-size 2`
3. **训练慢**: 50 epoch BUSI 单卡 V100 大概 2 小时。如果你想加速，可以降到 30 epoch 试试，看 loss 是否还在下降
4. **loss=NaN**: 可能 lr 太大，降到 5e-5

---

## 🎯 输出对应 Table 4b 下半

```
Table 4b 下半 (Cross-backbone transfer):
┌─────────────────────────────────────────────────┐
│ 5  U-Net (baseline)        77.83  32.51    --   │ ← 论文已有
│ 6  U-Net + U-BLoss         TBD    TBD     TBD   │ ← 待填 (来自 bibm_e3_unet_ubl)
│ 7  Swin-Unet (baseline)    81.18  27.75    --   │ ← 论文已有
│ 8  Swin-Unet + U-BLoss     TBD    TBD     TBD   │ ← 待填 (来自 bibm_e3_swin_ubl)
└─────────────────────────────────────────────────┘
```

> 注：论文 baseline 数字 77.83/81.18 是用 SAM2 训练框架跑的 U-Net/Swin-Unet（详见 Table 1）。新的 `bibm_e3_unet_baseline` 和 `bibm_e3_swin_baseline` 是用本脚本 (MONAI) 跑的，**数字可能略有差异**。
>
> 我们论文里只需要 Table 4b 下半的 4 行；baseline 行可以用本脚本跑的（推荐，保证公平：同一脚本下加 / 不加 U-BLoss 对比），也可以引用 Table 1 的数字。**建议都跑新的 4 行，保证 Δ 是同一脚本下的纯净对比**。
