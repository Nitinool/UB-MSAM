"""BIBM E3: 批量评测 — 遍历指定数据集的 runs 目录, 自动评测所有 ckpt.

用法:
    # 评测 BUSI 的所有实验 (扫描 runs/)
    python experiments/cross_backbone/eval_all.py --dataset BUSI

    # 评测 ISIC 的所有实验 (扫描 ISIC_runs/)
    python experiments/cross_backbone/eval_all.py --dataset ISIC

    # 评测 CVC
    python experiments/cross_backbone/eval_all.py --dataset CVC

输出:
    - 终端打印汇总表 (每个实验一行: exp_name / backbone / model_type / use_ubl / Dice / IoU / 95HD)
    - 存 CSV 到 <runs_dir>/eval_summary.csv
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASET_ROOTS, SegDataset, build_model_scratch, build_model_pretrained,
                    calculate_metrics, get_runs_dir, load_model_from_ckpt)


def evaluate_ckpt(ckpt_path, dataset_root, split='val', size=1024):
    """评测单个 ckpt, 返回 metrics dict. 自动判断 scratch/pretrained."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 自动判断 model_type, 兼容旧 ckpt
    model, backbone, model_type, use_ubl, epoch = load_model_from_ckpt(ckpt_path, size)
    model = model.to(device).eval()

    eval_set = SegDataset(dataset_root, split=split, size=size, augment=False, verbose=False)

    results = []
    for img_t, _ in eval_set:
        name = eval_set.names[len(results)]
        gt_path = os.path.join(eval_set.gt_dir, name, '00000.png')
        gt_orig = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        H, W = gt_orig.shape

        img = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
            pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        pred_mask = (pred > 0.5).astype(np.uint8) * 255

        dice, iou, h95 = calculate_metrics(gt_orig, pred_mask)
        results.append({'dice': dice, 'iou': iou, 'hd95': h95})

    df = pd.DataFrame(results)
    return {
        'backbone': backbone,
        'model_type': model_type,
        'use_ubl': use_ubl,
        'epoch': epoch,
        'dice_mean': df['dice'].mean(),
        'dice_std': df['dice'].std(),
        'iou_mean': df['iou'].mean(),
        'iou_std': df['iou'].std(),
        'hd95_mean': df['hd95'].mean(),
        'hd95_std': df['hd95'].std(),
        'count': len(df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['BUSI', 'ISIC', 'CVC', 'Kvasir'])
    parser.add_argument('--dataset-root', default=None, help='override DATASET_ROOTS')
    parser.add_argument('--split', default='val', help='评测哪个 split (默认 val=held-out test)')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--filter', default=None, help='只评测 exp_name 含此字符串的实验 (可选)')
    args = parser.parse_args()

    runs_dir = get_runs_dir(args.dataset)
    dataset_root = args.dataset_root or DATASET_ROOTS[args.dataset]

    # 扫描所有含 ckpt 的子目录 (优先 best_checkpoint.pt, 没有就用 checkpoint.pt)
    ckpts = []
    if not runs_dir.exists():
        print(f"Directory not found: {runs_dir}/")
        return
    for sub in sorted(runs_dir.iterdir()):
        if not sub.is_dir():
            continue
        if args.filter and args.filter not in sub.name:
            continue
        best_path = sub / 'checkpoints' / 'best_checkpoint.pt'
        last_path = sub / 'checkpoints' / 'checkpoint.pt'
        if best_path.exists():
            ckpts.append((sub.name, best_path, 'best'))
        elif last_path.exists():
            ckpts.append((sub.name, last_path, 'last'))

    if not ckpts:
        print(f"No checkpoints found in {runs_dir}/" + (f" matching '{args.filter}'" if args.filter else ""))
        return

    print(f"Found {len(ckpts)} checkpoints in {runs_dir}/:")
    for name, _, kind in ckpts:
        print(f"  - {name} ({kind})")
    print()

    # 逐个评测
    summary = []
    for exp_name, ckpt_path, kind in ckpts:
        print(f"\n=== Evaluating: {exp_name} ({kind}) ===")
        try:
            metrics = evaluate_ckpt(ckpt_path, dataset_root, args.split, args.size)
            metrics['exp_name'] = exp_name
            metrics['ckpt_kind'] = kind
            summary.append(metrics)
            print(f"  backbone={metrics['backbone']} | model={metrics['model_type']} | "
                  f"use_ubl={metrics['use_ubl']} | epoch={metrics['epoch']}")
            print(f"  Dice={metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f}  "
                  f"IoU={metrics['iou_mean']:.4f}±{metrics['iou_std']:.4f}  "
                  f"95HD={metrics['hd95_mean']:.2f}±{metrics['hd95_std']:.2f}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            summary.append({'exp_name': exp_name, 'ckpt_kind': kind, 'error': f"{type(e).__name__}: {e}"})

    # 输出汇总表
    df = pd.DataFrame(summary)
    print('\n' + '=' * 110)
    print(f'=== {args.dataset} Evaluation Summary ({len(df)} experiments) ===')
    print('=' * 110)
    # 只显示关键列, 数值保留 4 位
    display_cols = ['exp_name', 'ckpt_kind', 'backbone', 'model_type', 'use_ubl',
                    'dice_mean', 'iou_mean', 'hd95_mean']
    available_cols = [c for c in display_cols if c in df.columns]
    df_display = df[available_cols].copy()
    for c in ['dice_mean', 'iou_mean', 'hd95_mean']:
        if c in df_display.columns:
            df_display[c] = df_display[c].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
    print(df_display.to_string(index=False))

    # 存 CSV
    csv_path = runs_dir / 'eval_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved to: {csv_path}')


if __name__ == '__main__':
    main()
