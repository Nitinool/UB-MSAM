"""
BIBM E3: Cross-backbone training script with DDP support.

跑 U-Net 或 Swin-UNETR，可选叠加 U-BLoss，用于证明 U-BLoss 是 backbone-agnostic 的训练目标 (回应 R2.2)。

支持单卡和多卡 DDP：
    # 单卡 (用 CUDA_VISIBLE_DEVICES 指定显卡)
    CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train.py \
        --backbone unet --exp-name bibm_e3_unet_baseline

    # 4 卡 DDP (用 torchrun)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiments/cross_backbone/train.py --backbone unet --exp-name bibm_e3_unet_baseline

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm


# =============================================================================
# DDP 工具
# =============================================================================
def setup_ddp():
    """根据环境变量初始化 DDP. 单卡运行时返回 (False, 0, 1)."""
    if 'WORLD_SIZE' not in os.environ:
        return False, 0, 1  # 单卡

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return True, rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_mean(tensor, world_size):
    """跨进程求平均."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / world_size


# =============================================================================
# 数据集 (BUSI, 复用 SAM2 的目录结构)
# =============================================================================
class BUSIDataset(Dataset):
    def __init__(self, root, split_txt, size=1024, augment=True, verbose=True):
        self.root = root
        self.size = size
        self.augment = augment
        with open(split_txt) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.img_dir = os.path.join(root, "JPEGImages")
        self.gt_dir = os.path.join(root, "Annotations")
        # 训练时过滤掉空 mask 的样本 (BUSI normal 已被剔除, 但保险起见)
        self.names = [n for n in self.names if self._has_lesion(n)]
        if verbose:
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

        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        gt = (gt > 127).astype(np.float32)

        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float()
        gt_t = torch.from_numpy(gt).float().unsqueeze(0)

        if self.augment and np.random.rand() < 0.5:
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
        # 新版 MONAI 移除了 img_size 参数, 用 use_v2=True 走新版 SwinV2 backbone
        try:
            return SwinUNETR(
                in_channels=3,
                out_channels=1,
                spatial_dims=2,
                feature_size=24,
                use_v2=True,
            )
        except TypeError:
            # 旧版 MONAI 仍然需要 img_size
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
# U-BLoss
# =============================================================================
def uncertainty_guided_boundary_loss_v2(pred_logits, gt_mask, eps=1e-6):
    prob = torch.sigmoid(pred_logits)
    prob_c = torch.clamp(prob, eps, 1 - eps)
    entropy = -prob_c * torch.log2(prob_c) - (1 - prob_c) * torch.log2(1 - prob_c)
    entropy = entropy.detach()

    sobel_kernel = torch.tensor(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        dtype=torch.float32, device=gt_mask.device
    ).view(1, 1, 3, 3)
    boundary = F.conv2d(gt_mask, sobel_kernel, padding=1)
    boundary = (boundary.abs() > 0.1).float()

    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, reduction='none')
    weighted = (1.0 + entropy) * bce * boundary

    loss = weighted.sum() / (boundary.sum() + eps)
    return loss


def dice_loss(pred_logits, gt_mask, eps=1.0):
    pred = torch.sigmoid(pred_logits)
    inter = (pred * gt_mask).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + gt_mask.sum(dim=(2, 3))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def compute_loss(pred_logits, gt_mask, use_ubl: bool, ubl_weight: float = 2.0, pos_weight: float = 7.0):
    """组合损失: weighted BCE + Dice (+ U-BLoss)

    Args:
        pos_weight: BCE 中前景像素的权重, 用于对抗类别不平衡
                    (BUSI 前景约 14%, 故 pos_weight ≈ 1/0.14 ≈ 7)
    """
    # 加 pos_weight 的 BCE: 让前景像素的 loss 权重高 7 倍
    # BCEWithLogitsLoss 内部对前景和背景分别加权
    pw = torch.tensor([pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, pos_weight=pw)
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
    parser.add_argument('--batch-size', type=int, default=4, help='per-GPU batch size')
    parser.add_argument('--epochs', type=int, default=150)  # 50 → 150, from-scratch U-Net 需要更多 epoch
    parser.add_argument('--lr', type=float, default=1e-3)   # 1e-4 → 1e-3, 原 lr 太低训练不收敛
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # === DDP 初始化 ===
    use_ddp, rank, world_size = setup_ddp()
    is_main = is_main_process(rank)

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if use_ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_main:
        print(f"Device: {device}, DDP: {use_ddp}, world_size: {world_size}")
        if device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(device)}")

    # === 输出目录 ===
    output_dir = Path('runs') / args.exp_name
    if is_main:
        (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            cfg = vars(args).copy()
            cfg['use_ddp'] = use_ddp
            cfg['world_size'] = world_size
            json.dump(cfg, f, indent=2)
        print(f"Output dir: {output_dir}")

    # === 数据集 ===
    train_set = BUSIDataset(
        args.dataset_root,
        os.path.join(args.dataset_root, 'ImageSets', 'train.txt'),
        size=args.size, augment=True, verbose=is_main,
    )

    if use_ddp:
        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    # === 模型 ===
    model = build_model(args.backbone, args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable params: {n_params / 1e6:.2f} M")
        print(f"Use U-BLoss: {args.use_ubl} (weight={args.ubl_weight})")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 用 SequentialLR 实现 warmup + cosine: 前 5 epoch 线性升温到 lr, 之后 cosine 衰减到 0
    warmup_epochs = max(1, args.epochs // 10)  # 默认前 10% epoch 用于 warmup
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )

    # === log file (只在 rank 0 写) ===
    if is_main:
        log_file = open(output_dir / 'train.log', 'w')
        def log(msg):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()
    else:
        def log(msg):
            pass

    log(f"=== Training start ===")
    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {use_ddp}, world_size: {world_size}, effective batch size: {args.batch_size * world_size}")
    log(f"Trainable params: {n_params / 1e6:.2f} M")

    for epoch in range(args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)  # DDP 必需, 确保每 epoch shuffle 不同

        losses = []
        bce_losses, dice_losses = [], []

        # rank 0 显示 tqdm, 其他 rank 安静
        if is_main:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = train_loader

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

            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}", bce=f"{parts['bce']:.3f}", dice=f"{parts['dice']:.3f}")

        scheduler.step()

        # 跨进程聚合 loss 用于日志
        mean_loss = np.mean(losses)
        if use_ddp:
            loss_tensor = torch.tensor(mean_loss, device=device)
            mean_loss = reduce_mean(loss_tensor, world_size).item()

        msg = (
            f"Epoch {epoch+1}/{args.epochs} | "
            f"loss={mean_loss:.4f} | bce={np.mean(bce_losses):.4f} | "
            f"dice={np.mean(dice_losses):.4f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        log(msg)

        # 只在 rank 0 保存 checkpoint
        if is_main:
            ckpt_path = output_dir / 'checkpoints' / 'checkpoint.pt'
            # 去掉 DDP 的 .module wrapper
            state_dict = model.module.state_dict() if use_ddp else model.state_dict()
            torch.save({
                'epoch': epoch + 1,
                'model': state_dict,
                'args': vars(args),
            }, ckpt_path)

    log(f"=== Training done. Final ckpt: {output_dir / 'checkpoints' / 'checkpoint.pt'} ===")

    if is_main:
        log_file.close()
        print(f"\n[OK] Done. Checkpoint at: {output_dir / 'checkpoints' / 'checkpoint.pt'}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
