"""Cross-backbone 实验共享代码: 数据集、模型构建、损失、指标、训练循环.

被 train_scratch.py / train_pretrained.py / eval.py 共用.
"""
import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path


# =============================================================================
# 数据集默认路径 (服务器). 不一致时用 --dataset-root override.
# =============================================================================
DATASET_ROOTS = {
    'BUSI':   '/home/zhengsongming/jupyterworkspace/datasets/BUSI_for_SAM2',
    'ISIC':   '/home/zhengsongming/jupyterworkspace/datasets/ISIC_for_SAM2',
    'CVC':    '/home/zhengsongming/jupyterworkspace/datasets/CVC_for_SAM2',
    'Kvasir': '/home/zhengsongming/jupyterworkspace/datasets/Kvasir_for_SAM2',
}


class SegDataset(Dataset):
    """通用分割数据集. 目录结构 (与 SAM2 训练一致):
        root/
            JPEGImages/<name>/00000.jpg
            Annotations/<name>/00000.png
            ImageSets/{train,val}.txt
    """
    def __init__(self, root, split='train', size=1024, augment=True, verbose=True):
        self.root = root
        self.size = size
        self.augment = augment and (split == 'train')
        split_txt = os.path.join(root, 'ImageSets', f'{split}.txt')
        with open(split_txt) as f:
            self.names = [l.strip() for l in f if l.strip()]
        self.img_dir = os.path.join(root, 'JPEGImages')
        self.gt_dir = os.path.join(root, 'Annotations')
        self.names = [n for n in self.names if self._has_lesion(n)]
        if verbose:
            print(f"Dataset [{split}]: {len(self.names)} samples from {root}")

    def _has_lesion(self, name):
        gt_path = os.path.join(self.gt_dir, name, '00000.png')
        if not os.path.exists(gt_path):
            return False
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        return gt is not None and (gt > 127).sum() > 0

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.img_dir, name, '00000.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(os.path.join(self.gt_dir, name, '00000.png'), cv2.IMREAD_GRAYSCALE)

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
# 模型构建 (两个版本)
# =============================================================================
def build_model_scratch(backbone, img_size):
    """MONAI from-scratch 模型 (无预训练)."""
    if backbone == 'unet':
        from monai.networks.nets import UNet
        return UNet(
            spatial_dims=2, in_channels=3, out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    elif backbone == 'swin_unetr':
        from monai.networks.nets import SwinUNETR
        try:
            return SwinUNETR(in_channels=3, out_channels=1, spatial_dims=2,
                             feature_size=24, use_v2=True)
        except TypeError:
            # 旧版 MONAI 仍需 img_size
            return SwinUNETR(img_size=(img_size, img_size), in_channels=3, out_channels=1,
                             spatial_dims=2, feature_size=24, use_v2=True)
    raise ValueError(f"Unknown backbone: {backbone}")


def build_model_pretrained(backbone, img_size):
    """smp + ImageNet 预训练 encoder."""
    import segmentation_models_pytorch as smp
    if backbone == 'unet':
        return smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                        in_channels=3, classes=1)
    elif backbone == 'swin_unetr':
        # timm 1.0.x 里的 SwinV2-Base 标准名, ImageNet 预训练
        # (SwinV2 支持任意输入尺寸, 1024 上窗口自适应, 显存吃紧可换 swinv2_tiny_window8_256)
        return smp.Unet(encoder_name='tu-swinv2_base_window8_256', encoder_weights='imagenet',
                        in_channels=3, classes=1)
    raise ValueError(f"Unknown backbone: {backbone}")


# =============================================================================
# 损失 (与 SAM2 训练框架中的 U-BLoss 实现一致)
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


def compute_loss(pred_logits, gt_mask, use_ubl, ubl_weight=2.0, pos_weight=7.0):
    """组合损失: weighted BCE + Dice (+ U-BLoss). pos_weight 对抗前景类别不平衡."""
    pw = torch.tensor([pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_mask, pos_weight=pw)
    dice = dice_loss(pred_logits, gt_mask)
    loss = bce + dice
    if use_ubl:
        ubl = uncertainty_guided_boundary_loss_v2(pred_logits, gt_mask)
        loss = loss + ubl_weight * ubl
    return loss, {'bce': bce.item(), 'dice': dice.item()}


# =============================================================================
# 指标
# =============================================================================
def calculate_metrics(gt_mask, pred_mask):
    """Dice, IoU, 95HD. 输入是 numpy array (H,W)."""
    from medpy.metric.binary import hd95
    gt_b = gt_mask > 0
    pred_b = pred_mask > 0
    inter = np.logical_and(gt_b, pred_b).sum()
    dice = 2 * inter / (gt_b.sum() + pred_b.sum() + 1e-8)
    union = gt_b.sum() + pred_b.sum() - inter
    iou = inter / (union + 1e-8)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        h95 = np.nan
    else:
        h95 = hd95(pred_b, gt_b)
    return float(dice), float(iou), float(h95)


# =============================================================================
# DDP 工具
# =============================================================================
def setup_ddp():
    if 'WORLD_SIZE' not in os.environ:
        return False, 0, 1
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
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / world_size


# =============================================================================
# 通用训练循环 (被 train_scratch.py / train_pretrained.py 调用)
# =============================================================================
def train_loop(model, build_fn, args, device, use_ddp, rank, world_size, output_dir):
    """通用训练循环. build_fn 是 build_model_scratch 或 build_model_pretrained."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
    from tqdm import tqdm

    is_main = is_main_process(rank)

    # 数据集
    dataset_root = args.dataset_root or DATASET_ROOTS[args.dataset]
    train_set = SegDataset(dataset_root, split='train', size=args.size,
                           augment=True, verbose=is_main)

    if use_ddp:
        local_rank = int(os.environ['LOCAL_RANK'])
        sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        sampler = None
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 模型
    model = build_fn(args.backbone, args.size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Trainable params: {n_params / 1e6:.2f} M")
        print(f"Use U-BLoss: {args.use_ubl} (weight={args.ubl_weight})")
        print(f"Dataset: {args.dataset} @ {dataset_root}")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, args.epochs // 10)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs),
        ],
        milestones=[warmup_epochs],
    )

    if is_main:
        log_file = open(output_dir / 'train.log', 'w')
        def log(msg):
            print(msg); log_file.write(msg + '\n'); log_file.flush()
    else:
        def log(msg): pass

    log(f"=== Training start ===")
    log(f"Args: {json.dumps(vars(args), indent=2)}")
    log(f"DDP: {use_ddp}, world_size: {world_size}, effective batch: {args.batch_size * world_size}")
    log(f"Trainable params: {n_params / 1e6:.2f} M")

    for epoch in range(args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        losses, bce_losses, dice_losses = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") if is_main else train_loader
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
        mean_loss = np.mean(losses)
        if use_ddp:
            loss_tensor = torch.tensor(mean_loss, device=device)
            mean_loss = reduce_mean(loss_tensor, world_size).item()

        msg = (f"Epoch {epoch+1}/{args.epochs} | loss={mean_loss:.4f} | "
               f"bce={np.mean(bce_losses):.4f} | dice={np.mean(dice_losses):.4f} | "
               f"lr={optimizer.param_groups[0]['lr']:.2e}")
        log(msg)

        if is_main:
            ckpt_path = output_dir / 'checkpoints' / 'checkpoint.pt'
            state_dict = model.module.state_dict() if use_ddp else model.state_dict()
            torch.save({'epoch': epoch + 1, 'model': state_dict, 'args': vars(args)}, ckpt_path)

    log(f"=== Training done. Final ckpt: {output_dir / 'checkpoints' / 'checkpoint.pt'} ===")
    if is_main:
        log_file.close()
    cleanup_ddp()
