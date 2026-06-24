# 服务器跑实验指南

> 服务器：4× Tesla V100-SXM2 32GB / CUDA 11.8 / Ubuntu 18.04 (kernel 4.15) / shell sh-4.4
> 截稿前要补的 4 组实验：E-design / E3 / E4 / E-baseline
> 工作流：本地推 → 服务器 git pull → conda env + 跑 → 结果回传

---

## 1️⃣ 服务器首次部署

```bash
# 1. 拉代码
git clone git@github.com:Nitinool/UB-MSAM.git
cd UB-MSAM

# 2. 创建 conda env（python 3.10+）
conda create -n ubmsam python=3.10 -y
conda activate ubmsam

# 3. 装依赖
# CUDA 11.8 → PyTorch 2.1.x（SAM2 官方建议 ≥2.5，但 cu118 适配最稳的是 2.1）
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 4. 编译安装 SAM2（含 csrc CUDA 算子）
pip install -e . --no-build-isolation

# 5. 下载 SAM2.1 预训练权重
cd checkpoints && bash download_ckpts.sh && cd ..
```

依赖问题排查见文末「常见坑」。

---

## 2️⃣ 数据集准备

所有 yaml 当前写死的路径是：
```
/home/zhengsongming/jupyterworkspace/datasets/<NAME>_for_SAM2/
    ├── JPEGImages/      所有 RGB 图
    ├── Annotations/     所有 mask (PNG, 0/255)
    └── ImageSets/
        ├── train.txt    一行一个文件名（不带扩展名）
        ├── val.txt
        └── test.txt
```

需要在服务器准备的目录：
- `BUSI_for_SAM2/`（含 benign + malignant，剔除 normal）
- `ISIC_for_SAM2/`（官方 split：2594 / 100 / 1000）
- `CVC_for_SAM2/`（612 帧，8:1:1 seed 42）
- `Kvasir_for_SAM2/`（1000 张，仅作为 zero-shot 目标域，不需要 train.txt）

如果服务器上数据集放在别处，**用 hydra override 改 path 比改 yaml 干净**，例如：
```bash
python training/train.py \
    -c configs/sam2.1_training/busi_adapter_bLoss_finetune.yaml \
    +dataset.img_folder=/data/BUSI/JPEGImages \
    +dataset.gt_folder=/data/BUSI/Annotations \
    +dataset.file_list_txt=/data/BUSI/ImageSets/train.txt \
    --use-cluster 0 --num-gpus 1
```

---

## 3️⃣ 四组实验 → 命令清单

V100 单卡 32GB，batch_size=3 在 1024×1024 跑 BUSI 大概 3-6h 一次。可用 4 卡并行跑不同实验。

### E-design：U-BLoss 设计消融（最关键，回应 R3.4）

需新写 3 个 yaml（基于 `busi_adapter_bLoss_finetune.yaml` 改）：

```bash
# variant 1: w/o boundary restriction（全图加权）
# 需要改 training/loss_fns.py:308 把 gt_boundary 全设 1
# 建议：在 yaml 里加开关 loss.use_boundary_mask: false，源码加 if 判断

# variant 2: w/o stop-gradient on H
# 删除 training/loss_fns.py:61 的 .detach()

# variant 3: multiplicative H·BCE
# 改成 weighted_loss_map = uncertainty_map.detach() * boundary_loss_map
```

推荐：把这三个 variant 做成 `loss_fns.py` 里的开关参数，新建 3 个 yaml：
- `busi_ablation_no_boundary.yaml`
- `busi_ablation_no_stopgrad.yaml`
- `busi_ablation_multiplicative.yaml`

每个跑一次 BUSI（~5 hours），4 卡可同时进行。

### E3：U-BLoss 跨 backbone（回应 R2.2，narrative pivot 的灵魂）

不能直接复用现有训练框架（SAM2 训练逻辑跟 U-Net 完全不同）。**推荐路线**：

**Option A**（简单）：用 `monai.networks.nets.UNet` + 独立 PyTorch 训练循环
- 新写 `experiments/cross_backbone/train_unet.py`
- 用同一个 BUSI split 跑两次：`U-Net (baseline)` + `U-Net + U-BLoss`
- 数据加载用 `MONAI` 自带的 `Dataset` + `Compose`
- Swin-Unet 同理（`monai.networks.nets.SwinUNETR`）

**Option B**（较快但少灵活）：用 `nnUNetv2`，在 nnU-Net 训练循环里挂 U-BLoss

时间预估：写代码 0.5 天 + 跑 4 个组合 (UNet baseline / UNet+UBL / Swin baseline / Swin+UBL) × 各 3-5h = 1 天总计。

### E4：SAM2 / MedSAM2 在 Kvasir 零样本（回应 R2.3）

不需要训练，**只做推理**。脚本骨架：

```python
# 加载预训练 SAM2 / MedSAM2
predictor = sam2_image_predictor.SAM2ImagePredictor(model)

# 遍历 Kvasir 测试集
for img, gt_mask in kvasir_loader:
    # 用 GT 派生 box prompt
    box = mask_to_bbox(gt_mask, jitter=10)
    predictor.set_image(img)
    masks, _, _ = predictor.predict(box=box, multimask_output=False)
    # 算 Dice, IoU, 95HD
```

时间：~30 分钟跑完。

### E-baseline：SAM-Adapter / Medical SAM Adapter（回应 R1.6 / R2.1）

两个独立仓库：
- SAM-Adapter: https://github.com/tianrun-chen/SAM-Adapter-PyTorch
- Medical SAM Adapter: https://github.com/SuperMedIntel/Medical-SAM-Adapter

**简化方案**（节省时间）：直接引用它们论文里报告的 BUSI/ISIC 数值，标脚注说明来源。

**严谨方案**（如果有时间）：服务器另开 2 个 conda env 装两个仓库，在你的同一份 BUSI/ISIC/CVC split 上重跑，约 2 天。

---

## 4️⃣ 实验结果回传到论文 TBD 的流程

实验跑完后：

```bash
# 服务器上
# 1. 整理结果到一个 results.json，包含每个 cell 的 mean ± std
# 2. push 到 GitHub（可建 results 分支）
git add experiments/results/
git commit -m "E3 results"
git push

# 本地
git pull
# 把数字填到 D:\Programs\paper-ubmsam\main_bibm.tex 的 TBD 位置
# Claude 可以帮你做这一步
```

把数字告诉 Claude，会自动替换 main_bibm.tex 中所有 TBD 并同步更新叙述段。

---

## 5️⃣ 常见坑

### CUDA 11.8 + PyTorch 版本搭配
- SAM2 官方推荐 torch>=2.5，但 cu118 上 2.5 偶有 csrc 编译问题
- 实测稳定：`torch==2.1.2 + torchvision==0.16.2 + cu118`

### `pip install -e .` 报错
- 多半是 csrc CUDA 算子在编译，错误信息看 `gcc` / `nvcc`
- 临时绕过：在 setup.py 里把 `BuildExtension` 注释掉，跑训练时 fallback 到纯 PyTorch impl（性能损失 < 5%）

### Hydra 找不到配置
- `-c` 后面**不要带 `sam2/configs/`**，hydra 是从 `sam2.configs.sam2.1_training.*` 找的
- 正确：`-c configs/sam2.1_training/busi_adapter_bLoss_finetune.yaml`

### 显存爆
- V100 32GB，batch_size=3 + 1024×1024 + 4 stages adapter 应该够用
- 不够时降到 batch_size=2 + grad accumulation

### bf16 / fp16
- V100 不支持 bf16，只能用 fp32 或 fp16
- yaml 里搜 `precision` / `amp`，确认没设成 bf16

---

## 6️⃣ 后续 TODO（在服务器上）

1. [ ] conda env 配好，能跑通 `busi_adapter_bLoss_finetune.yaml` 一个 epoch
2. [ ] 数据路径要么放在 `/home/zhengsongming/jupyterworkspace/datasets/`，要么用 hydra override
3. [ ] 写 3 个 design ablation 的开关 + yaml（E-design）
4. [ ] 写 U-Net / Swin-Unet 训练脚本（E3）
5. [ ] 写 SAM2 / MedSAM2 在 Kvasir 上的推理脚本（E4）
6. [ ] 跑完所有实验，结果整理回传

完成后告诉 Claude，让它把 TBD 填到论文 `main_bibm.tex`。
