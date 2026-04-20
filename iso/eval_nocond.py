import os, sys, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_psnr

from model_injection import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split, parse_scene_folder

CKPT = 'checkpoints/iso/best.pth'

BUCKETS = [(50, 200, 'Low'), (201, 800, 'Mid'),
           (801, 3200, 'High'), (3201, 10000, 'VeryHigh')]

def tile_image(img_chw, patch=256):
    _, H, W = img_chw.shape
    patches = []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img_chw[:, y:y+patch, x:x+patch])
    return patches

def main():
    device = torch.device('cuda')

    print(f"Loading model from {CKPT}")
    model = ConditionedDenoiser(channels=32, param_dim=9).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['denoiser_state_dict'])
    model.eval()

    _, val_folders = get_split()
    val_ds = SIDDIsoDataset(val_folders, mode='val')

    records = []

    with torch.no_grad():
        for i, (noisy, gt, _, iso) in enumerate(val_ds):
            noisy_patches = tile_image(noisy)
            gt_patches = tile_image(gt)

            psnrs = []
            for np_p, gt_p in zip(noisy_patches, gt_patches):
                inp = np_p.unsqueeze(0).to(device)
                zero_embed = torch.zeros(1, 9).to(device)
                out = torch.clamp(model(inp, zero_embed), 0, 1)

                d = out.squeeze(0).cpu().numpy()
                g = gt_p.numpy()
                psnrs.append(get_psnr(d, g))
                records.append((int(iso), float(psnrs[-1])))

            print(f"  [{i+1:3d}/{len(val_ds)}] iso={iso:5d} "
                  f"n_patch={len(psnrs):3d} mean_psnr={np.mean(psnrs):.2f}")

    with open('results_nocond.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iso', 'psnr'])
        w.writerows(records)
    print(f"\nSaved {len(records)} records to results_nocond.csv")

    print(f"\n{'='*60}")
    print(f"Zero-conditioning ablation (same checkpoint, zero ISO embed)")
    print(f"{'='*60}")
    print(f"{'Bucket':<12} {'N':<6} {'NoCond PSNR':<14} {'std':<8}")
    print('-' * 60)
    for lo, hi, name in BUCKETS:
        vals = [r[1] for r in records if lo <= r[0] <= hi]
        if vals:
            print(f"{name:<12} {len(vals):<6} {np.mean(vals):<14.3f} {np.std(vals):<8.3f}")
    overall = [r[1] for r in records]
    print('-' * 60)
    print(f"{'Overall':<12} {len(overall):<6} {np.mean(overall):<14.3f} {np.std(overall):<8.3f}")

    if os.path.exists('results_iso_only.csv'):
        iso_records = []
        with open('results_iso_only.csv') as f:
            for row in csv.DictReader(f):
                iso_records.append((int(row['iso']), float(row['psnr'])))

        print(f"\n{'='*60}")
        print(f"Conditioning contribution (ISO-conditioned minus zero-cond)")
        print(f"{'='*60}")
        print(f"{'Bucket':<12} {'With ISO':<12} {'No Cond':<12} {'Δ PSNR':<10}")
        print('-' * 60)
        for lo, hi, name in BUCKETS:
            v_iso = [r[1] for r in iso_records if lo <= r[0] <= hi]
            v_zero = [r[1] for r in records if lo <= r[0] <= hi]
            if v_iso and v_zero:
                delta = np.mean(v_iso) - np.mean(v_zero)
                print(f"{name:<12} {np.mean(v_iso):<12.2f} {np.mean(v_zero):<12.2f} "
                      f"{'+' if delta > 0 else ''}{delta:<10.2f}")
        d_all = np.mean([r[1] for r in iso_records]) - np.mean(overall)
        print('-' * 60)
        print(f"{'Overall':<12} {np.mean([r[1] for r in iso_records]):<12.2f} "
              f"{np.mean(overall):<12.2f} {'+' if d_all > 0 else ''}{d_all:<10.2f}")

if __name__ == '__main__':
    main()
