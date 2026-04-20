import os, sys, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_psnr

from model_iso_only import ConditionedDenoiser
from data_sidd_sensor import SIDDSensorDataset, get_split, get_tier

CKPT = 'checkpoints/sensor/best.pth'

TIERS = [('High', ['GP']), ('Mid', ['S6', 'G4']), ('Low', ['IP', 'N6'])]

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
    val_ds = SIDDSensorDataset(val_folders, mode='val')

    records = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            noisy, gt, sensor_embed, sensor_val, phone, iso = val_ds[i]
            tier = get_tier(phone)

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
                records.append({
                    'phone': phone,
                    'tier': tier,
                    'sensor_val': sensor_val,
                    'iso': iso,
                    'psnr': float(psnrs[-1])
                })

            print(f"  [{i+1:3d}/{len(val_ds)}] phone={phone} tier={tier:>4s} "
                  f"iso={iso:5d} n_patch={len(psnrs):3d} mean_psnr={np.mean(psnrs):.2f}")

    with open('results_sensor_nocond.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"\nSaved {len(records)} records to results_sensor_nocond.csv")

    print(f"\n{'='*60}")
    print(f"Zero-conditioning ablation (sensor model, zero embed)")
    print(f"{'='*60}")
    print(f"{'Tier':<8} {'N':<6} {'NoCond PSNR':<14} {'std':<8}")
    print('-' * 60)
    for tier_name, phones in TIERS:
        vals = [r['psnr'] for r in records if r['tier'] == tier_name]
        if vals:
            print(f"{tier_name:<8} {len(vals):<6} {np.mean(vals):<14.3f} {np.std(vals):<8.3f}")
    overall = [r['psnr'] for r in records]
    print('-' * 60)
    print(f"{'Overall':<8} {len(overall):<6} {np.mean(overall):<14.3f} {np.std(overall):<8.3f}")

    if os.path.exists('results_sensor_inject.csv'):
        sensor_records = []
        with open('results_sensor_inject.csv') as f:
            for row in csv.DictReader(f):
                sensor_records.append(row)

        print(f"\n{'='*60}")
        print(f"Conditioning contribution (Sensor-conditioned minus zero-cond)")
        print(f"{'='*60}")
        print(f"{'Tier':<8} {'With Sensor':<14} {'No Cond':<14} {'Δ PSNR':<10}")
        print('-' * 60)
        for tier_name, phones in TIERS:
            v_sensor = [float(r['psnr']) for r in sensor_records if r['tier'] == tier_name]
            v_zero = [r['psnr'] for r in records if r['tier'] == tier_name]
            if v_sensor and v_zero:
                delta = np.mean(v_sensor) - np.mean(v_zero)
                print(f"{tier_name:<8} {np.mean(v_sensor):<14.2f} {np.mean(v_zero):<14.2f} "
                      f"{'+' if delta > 0 else ''}{delta:<10.2f}")
        d_all = np.mean([float(r['psnr']) for r in sensor_records]) - np.mean(overall)
        print('-' * 60)
        print(f"{'Overall':<8} {np.mean([float(r['psnr']) for r in sensor_records]):<14.2f} "
              f"{np.mean(overall):<14.2f} {'+' if d_all > 0 else ''}{d_all:<10.2f}")

if __name__ == '__main__':
    main()
