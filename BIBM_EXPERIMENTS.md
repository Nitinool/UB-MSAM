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
| 🟡 P1 | E3 跨 backbone | U-Net + U-BLoss | ✅ 训独立脚本 | (新建) eval_bibm_e3_unet_ubl.ipynb | ~半天 |
| 🟡 P1 | E3 跨 backbone | Swin-Unet + U-BLoss | ✅ 训独立脚本 | (新建) eval_bibm_e3_swin_ubl.ipynb | ~半天 |
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

模板：`baseline/dataset_BUSI/eval_template.ipynb`

**每跑完一个实验，做这三步**：

```bash
# 1. 复制模板（不进 git）
cd ~/jupyterworkspace/UB-MSAM/baseline/dataset_BUSI/
cp eval_template.ipynb eval_bibm_design_no_boundary.ipynb

# 2. 改第一处 ckpt 路径（其他都不动）
# SAM2_CHECKPOINT_PATH = "/home/.../UB-MSAM/runs/bibm_design_no_boundary/checkpoints/checkpoint.pt"

# 3. 跑这个 cell → 得到 final_summary 表
jupyter notebook eval_bibm_design_no_boundary.ipynb
```

**eval_*.ipynb 都在 .gitignore 里**，不进 git，不会冲突。

---

## 🚀 训练命令模板

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
