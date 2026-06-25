"""BIBM E3: 统一评测脚本 — 任意 ckpt × 任意 dataset.

用法:
    # 评测 BUSI 上的某个 ckpt
    python experiments/cross_backbone/eval.py \
        --ckpt runs/bibm_e3_busi_unet_pretrained_ubl/checkpoints/checkpoint.pt \
        --dataset BUSI \
        --output results/bibm_e3_busi_unet_pretrained_ubl.json

    # 评测 ISIC 上的某个 ckpt
    python experiments/cross_backbone/eval.py \
        --ckpt runs/bibm_e3_isic_swin_pretrained_ubl/checkpoints/checkpoint.pt \
        --dataset ISIC

    # 不写 --output 只打印, 写 --output 同时存 json

输出:
    - 终端打印 final_summary 表 (Dice/IoU/95HD, 按 benign/malignant + Overall)
    - 若 --output 指定, 同时存 json (方便后续填论文)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 让脚本能直接 import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASET_ROOTS, SegDataset, build_model_scratch, build_model_pretrained,
                    calculate_metrics, get_runs_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='checkpoint.pt 路径')
    parser.add_argument('--dataset', required=True, choices=['BUSI', 'ISIC', 'CVC', 'Kvasir'])
    parser.add_argument('--dataset-root', default=None, help='override DATASET_ROOTS')
    parser.add_argument('--split', default='val', help='评测哪个 split (默认 val, 即 held-out test)')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--output', default=None, help='结果 json 输出路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 从 ckpt 里读训练时的 args, 自动判断 model_type / backbone ===
    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})
    backbone = ckpt_args.get('backbone')
    model_type = ckpt_args.get('model_type', 'pretrained')  # scratch 或 pretrained
    print(f"  backbone={backbone}, model_type={model_type}, epoch={ckpt.get('epoch', '?')}")

    if model_type == 'scratch':
        model = build_model_scratch(backbone, args.size)
    else:
        model = build_model_pretrained(backbone, args.size)
    model.load_state_dict(ckpt['model'])
    model = model.to(device).eval()

    # === 加载评测集 ===
    dataset_root = args.dataset_root or DATASET_ROOTS[args.dataset]
    eval_set = SegDataset(dataset_root, split=args.split, size=args.size,
                          augment=False, verbose=True)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # === 推理 + 评测 ===
    results = []
    for img_t, gt_t in tqdm(eval_set, desc=f'Eval {args.dataset} [{args.split}]'):
        # 取原图分辨率算指标 (更真实的临床指标)
        name = eval_set.names[len(results)]  # 当前样本名
        gt_path = os.path.join(eval_set.gt_dir, name, '00000.png')
        gt_orig = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        H, W = gt_orig.shape

        img = img_t.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img)
            pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

        # resize 回原图分辨率
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        pred_mask = (pred > 0.5).astype(np.uint8) * 255

        dice, iou, h95 = calculate_metrics(gt_orig, pred_mask)
        # BUSI 文件名含 benign/malignant 用于分类汇总, 其他数据集只看 Overall
        category = 'benign' if 'benign' in name.lower() else (
            'malignant' if 'malignant' in name.lower() else 'all')
        results.append({'category': category, 'sample_name': name,
                        'dice': dice, 'iou': iou, 'hd95': h95})

    # === 汇总 ===
    df = pd.DataFrame(results)
    agg = {'dice': ['mean', 'std'], 'iou': ['mean', 'std'],
           'hd95': ['mean', 'std'], 'sample_name': ['count']}

    # BUSI 有 benign/malignant 分类, 其他数据集只有 'all'
    if df['category'].iloc[0] != 'all' and len(df['category'].unique()) > 1:
        cat_summary = df.groupby('category').agg(agg).reset_index()
        cat_summary.columns = ['_'.join(c).strip() for c in cat_summary.columns.values]
        cat_summary.rename(columns={'category_': 'Category', 'sample_name_count': 'Count'},
                           inplace=True)
    else:
        cat_summary = pd.DataFrame()

    overall = {
        'Category': 'Overall',
        'dice_mean': df['dice'].mean(), 'dice_std': df['dice'].std(),
        'iou_mean': df['iou'].mean(), 'iou_std': df['iou'].std(),
        'hd95_mean': df['hd95'].mean(), 'hd95_std': df['hd95'].std(),
        'Count': len(df),
    }
    final = pd.concat([cat_summary, pd.DataFrame([overall])], ignore_index=True)

    print('\n' + '=' * 80)
    print(f'--- {backbone.upper()} ({model_type}) | {args.dataset} [{args.split}] | {os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))} ---')
    print('=' * 80)
    print(final.to_string(index=False))
    print('=' * 80)

    # === 存 json ===
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            'ckpt': args.ckpt,
            'dataset': args.dataset,
            'split': args.split,
            'backbone': backbone,
            'model_type': model_type,
            'epoch': ckpt.get('epoch'),
            'overall': {k: float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                        for k, v in overall.items() if k != 'Category'},
            'per_sample': results,
        }
        with open(out_path, 'w') as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
