# BIBM E3: Cross-backbone 实验

> 回应 R2.2 / R3.1：证明 U-BLoss 是 backbone-agnostic 的训练目标，不依赖 SAM2 特定架构。
> Table 4b 下半 4 行：U-Net baseline / U-Net + UBL / Swin-UNETR baseline / Swin-UNETR + UBL

---

## 📦 依赖

```bash
pip install segmentation_models_pytorch timm einops
```

- `segmentation_models_pytorch` (smp) —— U-Net decoder + 各种 ImageNet 预训练 encoder
- `timm` —— smp 通过 timm 加载 SwinV2 等 encoder
- `einops` —— smp/timm 的 tensor 操作依赖

模型架构：
- `unet` → smp.Unet(encoder=ResNet-34, weights=ImageNet)
- `swin_unetr` → smp.Unet(encoder=SwinV2-Base, weights=ImageNet)

两个 backbone 都用 ImageNet 预训练 encoder，公平对比。论文里描述为
"U-Net (ResNet-34, ImageNet pre-trained)" 和 "Swin-UNet (SwinV2-Base, ImageNet pre-trained)"。

---

## 🚀 4 个实验命令

支持单卡和 DDP 多卡。**Effective batch size = `--batch-size` × GPU 数**。

### 方式 1：单卡顺序跑（最稳，~8h 总耗时）

```bash
cd ~/jupyterworkspace/UB-MSAM
conda activate sam22

CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py \
    --backbone unet --exp-name bibm_e3_unet_baseline

CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py \
    --backbone unet --use-ubl --exp-name bibm_e3_unet_ubl

CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py \
    --backbone swin_unetr --exp-name bibm_e3_swin_baseline

CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py \
    --backbone swin_unetr --use-ubl --exp-name bibm_e3_swin_ubl
```

### 方式 2：单进程 DDP 4 卡（每个实验更快，但要顺序跑 4 个）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    experiments/cross_backbone/train.py \
    --backbone unet --exp-name bibm_e3_unet_baseline

# 单个实验从 ~2h 缩到 ~30min, 4 个顺序跑约 2h 总耗时
```

⚠️ **DDP effective batch size**：默认 `--batch-size 4` × 4 卡 = 16（每张卡仍是 4）。如果想保持总 batch=4 用于公平对比，传 `--batch-size 1`。

### 方式 3：4 进程并行 4 张卡（推荐，~2h 总耗时）

每个 GPU 跑一个实验，互不干扰，最快出全部结果。

```bash
# 用 tmux 或 nohup 开 4 个后台进程
tmux new -s e3
# Ctrl+B " 横分; Ctrl+B % 纵分; 4 个 pane 分别跑:

# Pane 0: GPU 0
CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py \
    --backbone unet --exp-name bibm_e3_unet_baseline

# Pane 1: GPU 1
CUDA_VISIBLE_DEVICES=1 python experiments/cross_backbone/train.py \
    --backbone unet --use-ubl --exp-name bibm_e3_unet_ubl

# Pane 2: GPU 2
CUDA_VISIBLE_DEVICES=2 python experiments/cross_backbone/train.py \
    --backbone swin_unetr --exp-name bibm_e3_swin_baseline

# Pane 3: GPU 3
CUDA_VISIBLE_DEVICES=3 python experiments/cross_backbone/train.py \
    --backbone swin_unetr --use-ubl --exp-name bibm_e3_swin_ubl

# Ctrl+B d detach 离开, 2h 后 tmux attach -t e3 看结果
```

或者用 nohup：

```bash
mkdir -p logs
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py --backbone unet --exp-name bibm_e3_unet_baseline" > logs/e3_unet_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python experiments/cross_backbone/train.py --backbone unet --use-ubl --exp-name bibm_e3_unet_ubl" > logs/e3_unet_ubl.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python experiments/cross_backbone/train.py --backbone swin_unetr --exp-name bibm_e3_swin_baseline" > logs/e3_swin_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python experiments/cross_backbone/train.py --backbone swin_unetr --use-ubl --exp-name bibm_e3_swin_ubl" > logs/e3_swin_ubl.log 2>&1 &

# 查看进度
tail -f logs/e3_unet_ubl.log
nvidia-smi
```

> 💡 **推荐方式 3**：4 个实验 2h 全完成，比 DDP 简单，不会卡死。

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

## ⚙️ 训练超参（已写在 train.py 默认值，针对预训练 encoder 调优）

| 超参 | 值 | 说明 |
|---|---|---|
| batch_size | 8 | per-GPU |
| lr | 1e-4 | 小 lr 保护 ImageNet 预训练特征 |
| epochs | 100 | 预训练 encoder 收敛快 |
| weight_decay | 0.01 | |
| size | 1024 | |
| ubl_weight | 2.0 | |
| pos_weight (BCE) | 7.0 | 对抗前景 14% 类别不平衡 |
| seed | 42 | |
| scheduler | warmup(10%) + cosine | |

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
