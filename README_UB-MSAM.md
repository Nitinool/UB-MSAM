# UB-MSAM —— 代码工程速览（整理后）

> 整理时间：2026/06/24
> 工程大小：27 MB
> 用途：BIBM 2026 改稿对应的实验代码（仅 2D 图像分割，不含视频/3D）

---

## 📂 目录结构

```
D:\Programs\UB-MSAM\
├── README.md / README_UB-MSAM.md   ← 你在看的就是后者
├── INSTALL.md                      ← SAM2 官方安装指南
├── setup.py / pyproject.toml       ← pip install -e . 用
│
├── adapter/                        ← 用户手记
│   └── 操作流程.txt                  操作流程：adapter 怎么插的 / 训练命令
│
├── boundary_loss/                  ⚠️ 早期实验残留，未被引用
│   ├── loss_fns.py                   早期版 U-BLoss 实现，仅作历史参考
│   └── fine-tune/                    （空）
│
├── checkpoints/
│   └── download_ckpts.sh           ← 下载 SAM2.1 预训练权重（4 个 size）
│
├── baseline/                       🎯 运行入口！ipynb 训练脚本
│   ├── 训练脚本                       速记的命令行示例（cat 即可看）
│   ├── dataset_BUSI/                 BUSI 实验 ipynb（含 fewshot / adapter / bloss）
│   ├── dataset_CVC/                  CVC 实验
│   ├── dataset_ISIC/                 ISIC 实验
│   ├── dataset_kavsir/               Kvasir-SEG 零样本
│   └── visualizations/               敏感性 / few-shot 可视化
│
├── sam2/                           SAM2 主代码（已删 video predictor）
│   ├── modeling/
│   │   ├── backbones/
│   │   │   ├── hieradet.py              SAM2 原版 backbone
│   │   │   └── hieradet_adapterv1.py 🔑 含 Adapter class + use_adapter 钩子
│   │   ├── adapter/
│   │   │   └── adapter.py            ⚠️ 重复实现，与 hieradet_adapterv1 里的 Adapter 一样，未被引用
│   │   ├── memory_attention.py / memory_encoder.py / sam2_base.py / ...
│   ├── configs/
│   │   ├── sam2/ + sam2.1/           原版 SAM2 / SAM2.1 模型配置
│   │   └── sam2.1_training/        🎯 训练 yaml 都在这（详见下面）
│   ├── build_sam.py                 build_sam2() 入口
│   ├── sam2_image_predictor.py       图像推理（推理用）
│   ├── automatic_mask_generator.py
│   └── csrc/                        CUDA 算子源码
│
├── training/                       训练框架
│   ├── train.py                    🎯 训练入口
│   ├── trainer.py                  🔑 含 adapter 冻结逻辑 (use_adapter_finetuning)
│   ├── loss_fns.py                 🔑 U-BLoss 真正实现：
│   │                                 - uncertainty_guided_boundary_loss()
│   │                                 - class MultiStepMultiMasksAndIous_UBL (yaml 里就是引用它)
│   │                                 - class MultiStepMultiMasksAndIous (原版)
│   ├── optimizer.py
│   ├── model/sam2.py               SAM2Train 训练 wrapper
│   ├── dataset/
│   │   ├── vos_dataset.py          ⚠️ 名字里有 vos 但 BUSI/ISIC/CVC 也用它读单张图
│   │   ├── vos_raw_dataset.py      PNGRawDataset 是关键
│   │   └── transforms.py
│   └── utils/                       checkpoint_utils / logger / distributed
│
├── assets/model_diagram.png         SAM2 官方架构图（可删，但 README.md 引用了）
└── .github / .clang-format / 各种 license / pyproject ...
```

---

## 🎯 三个关键改动点（论文核心代码）

| # | 位置 | 作用 |
|---|---|---|
| 1 | `sam2/modeling/backbones/hieradet_adapterv1.py` | 在 Hiera encoder 每个 MultiScaleBlock 末尾插入 `Adapter`（bottleneck MLP，输出 zero init）|
| 2 | `training/loss_fns.py:20-69` `uncertainty_guided_boundary_loss()` | U-BLoss 公式：Sobel 提取 GT 边界 → `(1+H)·BCE` → 仅在边界上聚合 |
| 3 | `training/trainer.py:1005-1029` | 根据 `model_conf.use_adapter` 冻结所有参数、只解冻名字含 `adapter` 的参数 |

---

## 📋 训练 yaml 清单

`sam2/configs/sam2.1_training/` 下所有 yaml：

| yaml | 用途 | 模式 |
|---|---|---|
| `busi_fulltune_direct.yaml` | BUSI 全量微调（不冻结）| baseline |
| `busi_adapter_finetune.yaml` | BUSI Adapter PEFT，无 U-BLoss | ablation |
| `busi_adapter_bLoss_finetune.yaml` | **BUSI Adapter + U-BLoss** | 🎯 主实验 |
| `busi_fewshot_10/25/50.yaml` | BUSI 10%/25%/50% 数据 few-shot | 数据效率分析 |
| `medsam2_baseline_10/25/50.yaml` | MedSAM2 baseline 在 10/25/50% 数据上 | 数据效率对比 |
| `CVC_adapter_bLoss_finetune.yaml` | **CVC + Adapter + U-BLoss** | 🎯 主实验 |
| `ISIC_adapter_bLoss_finetune.yaml` | **ISIC + Adapter + U-BLoss** | 🎯 主实验 |
| `kavsir_adapter_bLoss_finetune.yaml` | Kvasir-SEG + Adapter + U-BLoss | 零样本目标域用 |
| `Sessile_adapter_bLoss_finetune.yaml` | Sessile（CVC 的子集）+ U-BLoss | 可选 |

> **数据路径**：所有 yaml 都写死了 `/home/zhengsongming/jupyterworkspace/datasets/<DATASET>_for_SAM2/`。服务器上传后**必须改 path**，详见 `RUN_ON_SERVER.md`。

---

## 🚀 标准训练命令

单卡：
```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    -c configs/sam2.1_training/busi_adapter_bLoss_finetune.yaml \
    --use-cluster 0 --num-gpus 1
```

多卡：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/train.py \
    -c configs/sam2.1_training/busi_adapter_bLoss_finetune.yaml \
    --use-cluster 0 --num-gpus 4
```

> 注意：`-c` 后面**不要加 `sam2/configs/`**，hydra 会自动从 sam2 包里找 configs。

---

## ⚠️ 已删除的内容（与 BIBM 论文无关）

整理 2026/06/24 删除：
- `demo/` — Web demo 后端 + 前端
- `sav_dataset/` — SAV 视频数据集工具
- `tools/vos_inference.py` — 视频对象分割推理
- `sam2/sam2_video_predictor.py` / `_legacy.py` — 视频推理预测器
- `sam2/benchmark.py` — 视频 benchmark
- `training/scripts/sav_frame_extraction_submitit.py` — SAV 帧提取
- `sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml` — MOSE 视频微调
- `assets/sa_v_dataset.jpg`
- `backend.Dockerfile` / `docker-compose.yaml`
- `build_sam.py` 里的 `build_sam2_video_predictor` 已移除并加注释

恢复方法：从 SAM2 上游 fork 拷回，或 git revert 整理 commit。

---

## ⚠️ 有意保留但未引用的"残留"

按用户要求保留：
- `boundary_loss/loss_fns.py` — 早期版 U-BLoss 实现（**真正用的是 `training/loss_fns.py`**）
- `sam2/modeling/adapter/adapter.py` — 独立的 Adapter 实现（**真正用的是 `hieradet_adapterv1.py` 内嵌的 Adapter**）

如果以后清理：上面两个删了不影响任何功能。
