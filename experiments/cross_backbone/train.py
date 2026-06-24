"""
BIBM E3: Cross-backbone training script.

跑 U-Net 或 Swin-UNETR，可选叠加 U-BLoss，用于证明 U-BLoss 是 backbone-agnostic 的训练目标 (回应 R2.2)。

用法 (在 ~/jupyterworkspace/UB-MSAM/ 下):
    # U-Net baseline (无 U-BLoss)
    python experiments/cross_backbone/train.py --backbone unet --exp-name bibm_e3_unet_baseline

    # U-Net + U-BLoss
    python experiments/cross_backbone/train.py --backbone unet --use-ubl --exp-name bibm_e3_unet_ubl

    # Swin-UNETR baseline
    python experiments/cross_backbone/train.py --backbone swin_unetr --exp-name bibm_e3_swin_baseline

    # Swin-UNETR + U-BLoss
    python experiments/cross_backbone/train.py --backbone swin_unetr --use-ubl --exp-name bibm_e3_swin_ubl

输出: ./runs/<exp_name>/{checkpoint.pt, train.log, config.json}
依赖: monai (pip install monai)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# 数据集 (BUSI, 复用 SAM2 的目录结构)
# =============================================================================
class BUSIDataset(Dataset):
    """BUSI 数据集. 目录结构:
        root/
            JPEGImages/<name>/00000.jpg
            Annotations/<name>/00000.png
            ImageSets/train.txt (一行一个 name)
    """
    def __init__(self, root, split_txt, size=1024, augment=True):
        self.root = root
        self.size = size
        self.augment = augment
        with open(split_txt) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.img_dir = os.path.join(root, "JPEGImages")
        self.gt_dir = os.path.join(root, "Annotations")
        # 训练时过滤掉空 mask 的样本 (BUSI normal 已被剔除, 但保险起见)
        self.names = [n for n in self.names if self._has_lesion(n)]
        print(f"Dataset: {len(self.names)} samples loaded from {split_txt}")

    def _has_lesion(self, name):
        gt_path = os.path.join(self.gt_dir, name, "00000.png")
        if not os.path.exists(gt_path):
            return False
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        return gt is not None and (gt > 127).sum() > 0

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.img_dir, name, "00000.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(self.gt_dir, name, "00000.png"), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # ImageNet 标准化
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        gt = (gt > 127).astype(np.float32)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)  # [1, H, W]

        if self.augment:
            # 简单的水平翻转
            if np.random.rand() < 0.5:
                img_t = torch.flip(img_t, dims=[2])
                gt_t = torch.flip(gt_t, dims=[2])

        return img_t, gt_t


# =============================================================================
# 模型构建
# =============================================================================
def build_model(backbone: str, img_size: int):
    if backbone == 'unet':
        from monai.networks.nets import UNet
        return UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif backbone == 'swin_unetr':
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(
            img_size=(img_size, img_size),
            in_channels=3,
            out_channels=1,
            spatial_dims=2,
            feature_size=24,
            use_v2=True,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# =============================================================================
# U-BLoss (inline 实现, 与 training/loss_fns.py 中的逻辑完全一致)
# 行为对应论文 main 设置: boundary_restriction=True, stop_gradient=True, additive=True
# =============================================================================
def uncertainty_guided_boundary_loss_v2(pred_logits, gt_mask, eps=1e-6):
    """
    Args:
        pred_logits: [N, 1, H, W] 模型 logits 输出
        gt_mask: [N, 1, H, W] GT (0/1)
    Returns:
        scalar loss
    """
    # 1. 计算熵 (predictive uncertainty)
    prob = torch.sigmoid(pred_logits)
    prob_c = torch.clamp(prob, eps, 1 - eps)
    entropy = -prob_c * torch.log2(prob_c) - (1 - prob_c) * torch.log2(1 - prob_c)
    # stop-gradient: 熵不参与 backprop
    entropy = entropy.detach()

    # 2. 用 Sobel 提取边界 mask
    sobel_kernel = torch.tensor(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        dtype=torch.float32, device=gt_mask.device
    ).view(1, 1, 3, 3)
    boundary = F.conv2d(gt_mask, sobel_kernel, padding=1)
    boundary = (boundary.abs() > 0.1).float()  # [N, 1, H, W]

    # 3. 加权 BCE: (1 + H) · BCE, 仅在边界上
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction='none')
    weighted = (1.0 + entropy) * bce * boundary

    # 4. 在边界像素上 average
    loss = weighted.sum() / (boundary.sum() + eps)
    return loss


def dice_loss(pred_logits, gt_mask, eps=1.0):
    pred = torch.sigmoid(pred_logits)
    inter = (pred * gt_mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def compute_loss(pred_logits, gt_mask, use_ubl: bool, ubl_weight: float = 2.0):
    """组合损失: BCE + Dice (+ U-BLoss)"""
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask)
    dice = dice_loss(pred_logits, gt_mask)
    loss = bce + dice
    if use_ubl:
        ubl = uncertainty_guided_boundary_loss_v2(pred_logits, gt_mask)
        loss = loss + ubl_weight * ubl
    return loss, {'bce': bce.item(), 'dice': dice.item()}


# =============================================================================
# 训练
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=['unet', 'swin_unetr'])
    parser.add_argument('--use-ubl', action='store_true', help='Add U-BLoss term')
    parser.add_argument('--ubl-weight', type=float, default=2.0)
    parser.add_argument('--exp-name', required=True, help='Output dir name in ./runs/')
    parser.add_argument(
        '--dataset-root',
        default='/home/zhengsongming/jupyterworkspace/datasets/BUSI_for_SAM2',
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 输出目录 (与 SAM2 训练保持一致的位置: ./runs/<exp_name>/)
    output_dir = Path('runs') / args.exp_name
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Output dir: {output_dir}")

    # 数据集
    train_set = BUSIDataset(
        args.dataset_root,
        os.path.join(args.dataset_root, 'ImageSets', 'train.txt'),
        size=args.size, augment=True,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # 模型
    model = build_model(args.backbone, args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.2f} M")
    print(f"Use U-BLoss: {args.use_ubl} (weight={args.ubl_weight})")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # log file
    log_path = output_dir / 'train.log'
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"=== Training start ===")
    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"Trainable params: {n_params / 1e6:.2f} M")

    for epoch in range(args.epochs):
        model.train()
        losses = []
        bce_losses, dice_losses = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for img, gt in pbar:
            img, gt = img.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            pred = model(img)
            loss, parts = compute_loss(pred, gt, use_ubl=args.use_ubl, ubl_weight=args.ubl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            bce_losses.append(parts['bce'])
            dice_losses.append(parts['dice'])
            pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{parts['bce']:.3f}", dice=f"{parts['dice']:.3f}")

        scheduler.step()
        msg = (
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={np.mean(losses):.4f} | bce={np.mean(bce_losses):.4f} | "
            f"dice={np.mean(dice_losses):.4f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        log(msg)

        # 每个 epoch 都保存（覆盖最新的）
        ckpt_path = output_dir / 'checkpoints' / 'checkpoint.pt'
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'args': vars(args),
        }, ckpt_path)

    log(f"=== Training done. Final ckpt: {ckpt_path} ===")
    log_file.close()
    print(f"\n✅ Done. Checkpoint at: {ckpt_path}")


if __name__ == '__main__':
    main()
