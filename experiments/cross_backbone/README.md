# BIBM E3: Cross-backbone 实验

> 回应 R2.2 / R3.1：证明 U-BLoss 是 backbone-agnostic 的训练目标，不依赖 SAM2 特定架构。

## 📁 文件结构

```
experiments/cross_backbone/
├── common.py              # 共享: 数据集/模型/loss/指标/训练循环/early stopping
├── train_scratch.py       # MONAI from scratch (用于 Swin)
├── train_pretrained.py    # smp + ImageNet 预训练 (用于 U-Net)
├── eval.py                # 单次评测 (任意 ckpt × 任意 dataset)
├── eval_all.py            # 批量评测 (扫描整个 runs 目录)
└── README.md              # 本文件
```

## 📦 依赖

```bash
pip install segmentation_models_pytorch timm monai einops medpy pandas tqdm
```

## 🎯 模型方案（已定稿）

| Backbone | 脚本 | Encoder | 预训练 | lr | epochs |
|---|---|---|---|---|---|
| **U-Net** | `train_pretrained.py` | smp.Unet + ResNet-34 | ✅ ImageNet | 1e-4 | 100 |
| **Swin-UNETR** | `train_scratch.py` | MONAI SwinUNETR | ❌ from scratch | 1e-3 | 100 |

> **为什么不一致**：SwinV2 的 ImageNet 预训练权重在 1024 输入上有窗口尺寸兼容性问题（window8_256 锁死 256，12to16 版本下载慢/不稳定），改用 from scratch + early stopping 已能达到合理性能。论文 narrative 不依赖两个 backbone 绝对值一致，只看 **+U-BLoss 的 Δ gain**。

## 🚀 训练命令（BUSI，4 卡并行）

```bash
cd ~/jupyterworkspace/UB-MSAM

# 清掉旧 ckpt
rm -rf runs/bibm_e3_*

# 启动 4 个实验 (每个占 1 张卡, 互不干扰)
# U-Net 预训练 (2 个)
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train_pretrained.py --backbone unet       --dataset BUSI --exp-name bibm_e3_busi_unet_pre_baseline" > logs/e3_busi_unet_pre_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python experiments/cross_backbone/train_pretrained.py --backbone unet --use-ubl --dataset BUSI --exp-name bibm_e3_busi_unet_pre_ubl"     > logs/e3_busi_unet_pre_ubl.log 2>&1 &

# Swin from scratch (2 个, 100 epoch)
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python experiments/cross_backbone/train_scratch.py --backbone swin_unetr       --dataset BUSI --epochs 100 --exp-name bibm_e3_busi_swin_scratch_baseline" > logs/e3_busi_swin_scratch_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python experiments/cross_backbone/train_scratch.py --backbone swin_unetr --use-ubl --dataset BUSI --epochs 100 --exp-name bibm_e3_busi_swin_scratch_ubl"     > logs/e3_busi_swin_scratch_ubl.log 2>&1 &

# 监控 (Ctrl+C 退出 tail, 不会停训练)
tail -f logs/e3_busi_unet_pre_ubl.log

# 一眼看所有 log 进度
for f in logs/e3_busi_*.log; do echo "=== $(basename $f) ==="; tail -1 "$f"; done
```

### 其他数据集（ISIC / CVC）

把 `--dataset BUSI` 换成 `--dataset ISIC` 或 `--dataset CVC`，exp-name 里的 `busi` 也换掉。输出目录自动变：
- BUSI → `runs/`
- ISIC → `ISIC_runs/`
- CVC → `CVC_runs/`

## 📊 评测

```bash
# 批量评测 BUSI 所有实验 (自动优先 best_checkpoint.pt, 没有才用 checkpoint.pt)
python experiments/cross_backbone/eval_all.py --dataset BUSI

# 批量评测 ISIC
python experiments/cross_backbone/eval_all.py --dataset ISIC

# 只评测某个子集 (比如只看 +UBL 的)
python experiments/cross_backbone/eval_all.py --dataset BUSI --filter ubl

# 单次评测某个 ckpt (调试用, 输出详细 benign/malignant 分类)
python experiments/cross_backbone/eval.py \
    --ckpt runs/bibm_e3_busi_unet_pre_ubl/checkpoints/best_checkpoint.pt \
    --dataset BUSI
```

**eval 会自动从 ckpt 读 backbone 和 model_type**（scratch/pretrained），不用手动指定。

输出：
- 终端打印汇总表（每个实验一行 + ckpt_kind 标注 best/last）
- 存 CSV 到 `runs/eval_summary.csv`（或 `ISIC_runs/eval_summary.csv`）

## 🔧 Early Stopping 机制

每个实验训练时会做：
1. **从 train.txt 切 90/10**：90% 真训练 + 10% 内部 val（固定 seed=42，可复现）
2. **每个 epoch 在内部 val 上算 dice**
3. **保存两个 ckpt**：
   - `best_checkpoint.pt` — val dice 最高的 epoch（**评测优先用这个**）
   - `checkpoint.pt` — 最后一个 epoch（向后兼容）

> **val.txt 全程不碰**，作为 held-out test 集。这样 ckpt 选择没有数据泄漏。

训练 log 里会看到：
```
Epoch 50/100 | loss=0.58 | bce=0.43 | dice=0.62 | val_dice=0.68 | lr=...
  >> New best val_dice=0.6823 @ epoch 50, saved best_checkpoint.pt
```

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

| 超参 | U-Net (pretrained) | Swin (scratch) |
|---|---|---|
| batch_size | 8 | 8 |
| lr | 1e-4 | 1e-3 |
| epochs | 100 | 100 |
| weight_decay | 0.01 | 0.01 |
| size | 1024 | 1024 |
| ubl_weight | 2.0 | 2.0 |
| pos_weight (BCE) | 7.0 | 7.0 |
| scheduler | warmup(10%) + cosine | warmup(10%) + cosine |
| early stopping | 按 internal val dice | 按 internal val dice |

## 📋 完整实验矩阵

```
2 backbone (unet_pre, swin_scratch)
× 2 设置 (baseline, +UBL)
× 3 数据集 (BUSI, ISIC, CVC)
= 12 个实验
```

**优先级**：
1. 🔴 BUSI 4 个（最关键，回应 R2.2）
2. 🟡 ISIC 4 个 + CVC 4 个（多数据集验证）

## ⚠️ 常见坑

1. **SwinV2 预训练 OOM/报错** → 已改 from scratch，不会有这个问题
2. **U-Net 预训练第一次要下载 ResNet34 权重** ~85MB，从 hf-mirror.com（需 `export HF_ENDPOINT=https://hf-mirror.com`）
3. **Swin from scratch 过拟合** → early stopping 会救，看 best_checkpoint.pt 的 epoch 是否在 30-60
4. **eval_all.py 找不到 ckpt** → 确认 `runs/bibm_e3_xxx/checkpoints/` 下有 `best_checkpoint.pt` 或 `checkpoint.pt`
5. **显存不够** → Swin 1024 batch=8 可能 OOM，降 `--batch-size 4`

## 📝 论文描述（对应 Table 4b 下半）

```
Cross-backbone transfer (drop-in U-BLoss replacement of base loss):
  U-Net (ResNet-34, ImageNet pre-trained)     baseline  → +U-BLoss
  Swin-UNETR (from scratch)                   baseline  → +U-BLoss
```

> 论文里诚实写两个 backbone 训练设置不同（U-Net 预训练 / Swin from scratch），理由见上面"为什么不一致"。
