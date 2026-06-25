"""BIBM E3: Cross-backbone 训练 — smp + ImageNet 预训练 encoder.

用法:
    # 单卡
    CUDA_VISIBLE_DEVICES=0 python experiments/cross_backbone/train_pretrained.py \
        --backbone unet --dataset BUSI --exp-name bibm_e3_busi_unet_pretrained_baseline

    # 4 卡 DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiments/cross_backbone/train_pretrained.py \
        --backbone unet --dataset BUSI --exp-name bibm_e3_busi_unet_pretrained_baseline

    # + U-BLoss
    python experiments/cross_backbone/train_pretrained.py \
        --backbone unet --use-ubl --dataset BUSI --exp-name bibm_e3_busi_unet_pretrained_ubl

支持 --dataset: BUSI / ISIC / CVC / Kvasir (路径在 common.DATASET_ROOTS)
输出: ./runs/<exp_name>/{checkpoint.pt, train.log, config.json}
依赖: segmentation_models_pytorch, timm, einops

模型:
  - unet       → smp.Unet(encoder=ResNet-34, weights=ImageNet)
  - swin_unetr → smp.Unet(encoder=SwinV2-Base, weights=ImageNet)
"""
import argparse
import json
from pathlib import Path

from common import (setup_ddp, is_main_process, train_loop, build_model_pretrained,
                    get_output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=['unet', 'swin_unetr'])
    parser.add_argument('--use-ubl', action='store_true')
    parser.add_argument('--ubl-weight', type=float, default=2.0)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--dataset', default='BUSI', choices=['BUSI', 'ISIC', 'CVC', 'Kvasir'])
    parser.add_argument('--dataset-root', default=None, help='override DATASET_ROOTS 默认路径')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)   # 预训练收敛快
    parser.add_argument('--lr', type=float, default=1e-4)    # 预训练用小 lr 保护特征
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import torch, numpy as np
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    use_ddp, rank, world_size = setup_ddp()
    if use_ddp:
        local_rank = int(__import__('os').environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main = is_main_process(rank)
    output_dir = get_output_dir(args.dataset, args.exp_name)
    if is_main:
        (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            cfg = vars(args).copy(); cfg['model_type'] = 'pretrained'; cfg['use_ddp'] = use_ddp
            json.dump(cfg, f, indent=2)
        print(f"Output dir: {output_dir} | Model: smp + ImageNet pretrained")

    train_loop(None, build_model_pretrained, args, device, use_ddp, rank, world_size, output_dir)


if __name__ == '__main__':
    main()
