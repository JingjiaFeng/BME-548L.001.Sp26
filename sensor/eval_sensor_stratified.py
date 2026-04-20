import os, sys, argparse, csv
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BaseDenoiser_SIDD
from utils import embed_param, get_psnr

from model_injection import ConditionedDenoiser
from data_sidd_sensor import SIDDSensorDataset, get_split, get_tier, SENSOR_QUALITY, SENSOR_MIN, SENSOR_MAX

def tile_image(img_chw, patch=256):
    _, H, W = img_chw.shape
    patches = []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img_chw[:, y:y+patch, x:x+patch])
    return patches

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'sensor'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--out_csv', type=str, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.ckpt is None:
        if args.model == 'baseline':
            args.ckpt = os.path.join(repo_root, 'pretrained_models/SIDD_baseline.pth')
        else:
            raise ValueError("--ckpt required for sensor model")

    if args.out_csv is None:
        args.out_csv = f'./results_sensor_{args.model}.csv'

    device = torch.device('cuda')

    print(f"Building {args.model} model...")
    if args.model == 'baseline':
        model = BaseDenoiser_SIDD(channels=32).to(device)
    else:
        model = ConditionedDenoiser(channels=32, param_dim=9).to(device)

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['denoiser_state_dict'])
    model.eval()

    _, val_folders = get_split()
    val_ds = SIDDSensorDataset(val_folders, mode='val')

    records = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            noisy, gt, sensor_embed, sensor_val, phone, iso = val_ds[i]
            tier = get_tier(phone)

            noisy_patches = tile_image(noisy, patch=256)
            gt_patches = tile_image(gt, patch=256)

            psnrs = []
            for np_p, gt_p in zip(noisy_patches, gt_patches):
                inp = np_p.unsqueeze(0).to(device)

                if args.model == 'baseline':
                    dummy_param = torch.zeros(1, 18).to(device)
                    dummy_phone = torch.zeros(1, 1).long().to(device)
                    out = model(inp, dummy_param, dummy_phone)
                else:
                    se = sensor_embed.unsqueeze(0).to(device)
                    out = model(inp, se)

                out = torch.clamp(out, 0, 1)
                d = out.squeeze(0).cpu().numpy()
                g = gt_p.numpy()
                p = get_psnr(d, g)
                psnrs.append(p)
                records.append({
                    'phone': phone,
                    'tier': tier,
                    'sensor_val': sensor_val,
                    'iso': iso,
                    'psnr': p
                })

            print(f"  [{i+1:3d}/{len(val_ds)}] phone={phone} tier={tier:>4s} "
                  f"sensor={sensor_val:.1f} iso={iso:5d} "
                  f"n_patch={len(psnrs):3d} mean_psnr={np.mean(psnrs):.2f}")

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"\nSaved {len(records)} records to {args.out_csv}")

    print(f"\n{'='*60}")
    print(f"Sensor-tier stratified PSNR ({args.model})")
    print(f"{'='*60}")
    print(f"{'Tier':<8} {'Phones':<12} {'Sensor Val':<12} {'N':<6} {'PSNR':<12} {'std':<8}")
    print('-' * 60)
    for tier_name, phones in [('High', ['GP']), ('Mid', ['S6','G4']), ('Low', ['IP','N6'])]:
        vals = [r['psnr'] for r in records if r['tier'] == tier_name]
        sv = SENSOR_QUALITY[phones[0]]
        if vals:
            print(f"{tier_name:<8} {','.join(phones):<12} {sv:<12.1f} "
                  f"{len(vals):<6} {np.mean(vals):<12.3f} {np.std(vals):<8.3f}")
    overall = [r['psnr'] for r in records]
    print('-' * 60)
    print(f"{'Overall':<8} {'':<12} {'':<12} "
          f"{len(overall):<6} {np.mean(overall):<12.3f} {np.std(overall):<8.3f}")

if __name__ == '__main__':
    main()
