# BIBM 2026 实验清单 & 操作手册

> 截稿 2026/07/05 · 当前 2026/06/24 · 剩约 11 天

---

## 📋 实验全清单（按优先级）

| 优先级 | 类别 | 实验 | 训练? | 评测 ipynb | 服务器上耗时 |
|---|---|---|---|---|---|
| 🔴 P0 | 主实验 | Full U-BLoss baseline | ✅ 复用旧 ckpt | eval_bibm_full_ubloss.ipynb | 30 min 评测 |
| 🔴 P0 | E-design v1 | w/o boundary restriction | 训 (`busi_ablation_no_boundary.yaml`) | eval_bibm_design_no_boundary.ipynb | ~3h 训 + 30 min 评 |
| 🔴 P0 | E-design v2 | w/o stop-gradient | 训 (`busi_ablation_no_stopgrad.yaml`) | eval_bibm_design_no_stopgrad.ipynb | ~3h 训 + 30 min 评 |
| 🔴 P0 | E-design v3 | multiplicative `H·BCE` | 训 (`busi_ablation_multiplicative.yaml`) | eval_bibm_design_multiplicative.ipynb | ~3h 训 + 30 min 评 |
| 🔴 P0 | E4 zero-shot | SAM2 on Kvasir-SEG | ❌ 纯推理 | (新建) eval_bibm_e4_sam2_on_kvasir.ipynb | 30 min |
| 🔴 P0 | E4 zero-shot | MedSAM2 on Kvasir-SEG | ❌ 纯推理 | (新建) eval_bibm_e4_medsam2_on_kvasir.ipynb | 30 min |
| 🟡 P1 | E3 跨 backbone | U-Net + U-BLoss | ✅ `experiments/cross_backbone/train.py` | (新建) eval_bibm_e3_unet_ubl.ipynb | ~2h 训 + 30 min 评 |
| 🟡 P1 | E3 跨 backbone | Swin-Unet + U-BLoss | ✅ `experiments/cross_backbone/train.py` | (新建) eval_bibm_e3_swin_ubl.ipynb | ~2h 训 + 30 min 评 |
| 🟡 P1 | E3 跨 backbone | U-Net baseline | ✅ `experiments/cross_backbone/train.py` | (新建) eval_bibm_e3_unet_baseline.ipynb | ~2h 训 + 30 min 评 |
| 🟡 P1 | E3 跨 backbone | Swin-Unet baseline | ✅ `experiments/cross_backbone/train.py` | (新建) eval_bibm_e3_swin_baseline.ipynb | ~2h 训 + 30 min 评 |
| 🟢 P2 | E-baseline | SAM-Adapter | ✅ 装外部仓库 | 后定 | 1-2 天 |
| 🟢 P2 | E-baseline | Medical SAM Adapter | ✅ 装外部仓库 | 后定 | 1-2 天 |

---

## 🎯 数据对应论文哪里

```
main_bibm.tex
├── Table 1 (SOTA 对比) ────────── Full U-BLoss + SAM-Adapter + Med-SAM-Adapter
├── Table 3 (Zero-shot) ───────── SAM2 + MedSAM2 在 Kvasir 上的数字 (E4)
└── Table 4b (Ablation)
    ├── 上半 Design ablation ──── E-design 3 个 variant + Full (4 行)
    └── 下半 Cross-backbone ──── E3 (4 行: U-Net + UBL / Swin + UBL + 两个 baseline)
```

---

## 🛠️ 评测 ipynb 工作流

两个评测模板：
- `baseline/dataset_BUSI/eval_template.ipynb` — **SAM2** 评测模板（需要 box prompt）
- `baseline/dataset_BUSI/eval_template_cnn.ipynb` — **CNN** 评测模板（无 prompt，用于 E3 U-Net / Swin-UNETR）

**每跑完一个实验，做这三步**：

```bash
# 1. 复制对应模板（不进 git）
cd ~/jupyterworkspace/UB-MSAM/baseline/dataset_BUSI/

# SAM2 实验：
cp eval_template.ipynb eval_bibm_design_no_boundary.ipynb

# E3 CNN 实验：
cp eval_template_cnn.ipynb eval_bibm_e3_unet_ubl.ipynb

# 2. 改 ckpt 路径（SAM2 模板改一行，CNN 模板改两行: BACKBONE 和 CKPT_PATH）

# 3. 跑这个 cell → 得到 final_summary 表
```

**eval_bibm_*.ipynb 都在 .gitignore 里**，不进 git，不会冲突。

---

## 🚀 训练命令模板

### E-design (SAM2 + Adapter)

```bash
cd ~/jupyterworkspace/UB-MSAM
conda activate sam22

# E-design v1
EXP_NAME=bibm_design_no_boundary CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
    -c configs/sam2.1_training/busi_ablation_no_boundary.yaml \
    --use-cluster 0 --num-gpus 4

# E-design v2
EXP_NAME=bibm_design_no_stopgrad CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
    -c configs/sam2.1_training/busi_ablation_no_stopgrad.yaml \
    --use-cluster 0 --num-gpus 4

# E-design v3
EXP_NAME=bibm_design_multiplicative CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
    -c configs/sam2.1_training/busi_ablation_multiplicative.yaml \
    --use-cluster 0 --num-gpus 4
```

### E3 (Cross-backbone, 用 MONAI U-Net / Swin-UNETR)

```bash
# 装依赖
pip install monai

# 推荐：4 张卡并行 4 个实验 (~2h 全完成)
# 用 tmux 或 nohup 后台跑, 不会因 ssh 断开而中断:
nohup bash -c "CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py --backbone unet --exp-name bibm_e3_unet_baseline" > logs/e3_unet_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=1 python experiments/cross_backbone/train.py --backbone unet --use-ubl --exp-name bibm_e3_unet_ubl" > logs/e3_unet_ubl.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=2 python experiments/cross_backbone/train.py --backbone swin_unetr --exp-name bibm_e3_swin_baseline" > logs/e3_swin_baseline.log 2>&1 &
nohup bash -c "CUDA_VISIBLE_DEVICES=3 python experiments/cross_backbone/train.py --backbone swin_unetr --use-ubl --exp-name bibm_e3_swin_ubl" > logs/e3_swin_ubl.log 2>&1 &

# 单实验 DDP 4 卡 (顺序跑 4 个, 总耗时 ~2h):
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    experiments/cross_backbone/train.py --backbone unet --exp-name bibm_e3_unet_baseline
```

详见 `experiments/cross_backbone/README.md`。

---

## 📌 现有 checkpoint 复用清单

| 实验 | ckpt 路径 | 备注 |
|---|---|---|
| Full U-BLoss（论文 93.46% 那个） | `/home/zhengsongming/jupyterworkspace/03医学图像分割/sam2_finetune_logs/busi_bLoss_adapter_run18/checkpoints/checkpoint.pt` | 旧仓库还在，可直接复用 |

---

## ⚠️ 一些坑

1. **`no_stopgrad` 可能 NaN**：训练时盯一下 loss，第一个 epoch loss 变 NaN 立刻 Ctrl+C 告诉 Claude
2. **`eval_template.ipynb` 不要直接跑**：跑就会产生 outputs 进 git，造成冲突。一定先 `cp` 出副本
3. **runs/ 目录已 gitignore**：ckpt + tensorboard log 不会进 git
4. **服务器路径写死的注意**：所有路径都是 `/home/zhengsongming/jupyterworkspace/...`，换机器要改
