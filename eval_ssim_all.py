import os, sys, csv, argparse
import numpy as np
import torch
from skimage.metrics import structural_similarity as compute_ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BaseDenoiser_SIDD
from utils import embed_param, get_psnr

from model_injection import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split as get_split_iso, ISO_MIN, ISO_MAX
from data_sidd_sensor import SIDDSensorDataset, get_split as get_split_sensor, get_tier, SENSOR_QUALITY, SENSOR_MIN, SENSOR_MAX

ISO_BUCKETS = [(50, 200, 'Low'), (201, 800, 'Mid'),
               (801, 3200, 'High'), (3201, 10000, 'VeryHigh')]

SENSOR_TIERS = [('High', ['GP']), ('Mid', ['S6', 'G4']), ('Low', ['IP', 'N6'])]

def tile_image(img_chw, patch=256):
    _, H, W = img_chw.shape
    patches = []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img_chw[:, y:y+patch, x:x+patch])
    return patches

def ssim_on_patches(pred, gt):
    
    p = pred.transpose(1, 2, 0)
    g = gt.transpose(1, 2, 0)
    return compute_ssim(p, g, channel_axis=2, data_range=1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['iso_cond', 'iso_nocond', 'sensor_cond', 'sensor_nocond', 'baseline'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--out_csv', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda')
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    auto_ckpt = {
        'iso_cond': 'checkpoints/iso/best.pth',
        'iso_nocond': 'checkpoints/iso/best.pth',
        'sensor_cond': 'checkpoints/sensor/best.pth',
        'sensor_nocond': 'checkpoints/sensor/best.pth',
        'baseline': os.path.join(repo_root, 'pretrained_models/SIDD_baseline.pth'),
    }
    if args.ckpt is None:
        args.ckpt = auto_ckpt[args.mode]
    if args.out_csv is None:
        args.out_csv = f'results_ssim_{args.mode}.csv'

    print(f"Mode: {args.mode}")
    print(f"Loading: {args.ckpt}")

    if args.mode == 'baseline':
        model = BaseDenoiser_SIDD(channels=32).to(device)
    else:
        model = ConditionedDenoiser(channels=32, param_dim=9).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['denoiser_state_dict'])
    model.eval()

    use_sensor = args.mode in ('sensor_cond', 'sensor_nocond')
    if use_sensor:
        _, val_folders = get_split_sensor()
        val_ds = SIDDSensorDataset(val_folders, mode='val')
    else:
        _, val_folders = get_split_iso()
        val_ds = SIDDIsoDataset(val_folders, mode='val')

    records = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            if use_sensor:
                noisy, gt, sensor_embed, sensor_val, phone, iso = val_ds[i]
                tier = get_tier(phone)
            else:
                noisy, gt, iso_embed, iso = val_ds[i]
                phone = ''
                tier = ''

            noisy_patches = tile_image(noisy)
            gt_patches = tile_image(gt)

            psnrs, ssims = [], []
            for np_p, gt_p in zip(noisy_patches, gt_patches):
                inp = np_p.unsqueeze(0).to(device)

                if args.mode == 'baseline':
                    dummy_param = torch.zeros(1, 18).to(device)
                    dummy_phone = torch.zeros(1, 1).long().to(device)
                    out = model(inp, dummy_param, dummy_phone)
                elif args.mode == 'iso_cond':
                    out = model(inp, iso_embed.unsqueeze(0).to(device))
                elif args.mode == 'iso_nocond':
                    out = model(inp, torch.zeros(1, 9).to(device))
                elif args.mode == 'sensor_cond':
                    out = model(inp, sensor_embed.unsqueeze(0).to(device))
                elif args.mode == 'sensor_nocond':
                    out = model(inp, torch.zeros(1, 9).to(device))

                out = torch.clamp(out, 0, 1)
                d = out.squeeze(0).cpu().numpy()
                g = gt_p.numpy()
                psnrs.append(get_psnr(d, g))
                ssims.append(ssim_on_patches(d, g))

                records.append({
                    'iso': int(iso),
                    'phone': phone,
                    'tier': tier,
                    'psnr': float(psnrs[-1]),
                    'ssim': float(ssims[-1])
                })

            print(f"  [{i+1:3d}/{len(val_ds)}] iso={iso:5d} "
                  f"psnr={np.mean(psnrs):.2f} ssim={np.mean(ssims):.4f}")

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"\nSaved {len(records)} records to {args.out_csv}")

    print(f"\n{'='*70}")
    print(f"ISO-stratified ({args.mode})")
    print(f"{'='*70}")
    print(f"{'Bucket':<12} {'N':<6} {'PSNR':<12} {'SSIM':<12}")
    print('-' * 70)
    for lo, hi, name in ISO_BUCKETS:
        p = [r['psnr'] for r in records if lo <= r['iso'] <= hi]
        s = [r['ssim'] for r in records if lo <= r['iso'] <= hi]
        if p:
            print(f"{name:<12} {len(p):<6} {np.mean(p):<12.3f} {np.mean(s):<12.4f}")
    overall_p = [r['psnr'] for r in records]
    overall_s = [r['ssim'] for r in records]
    print('-' * 70)
    print(f"{'Overall':<12} {len(overall_p):<6} {np.mean(overall_p):<12.3f} {np.mean(overall_s):<12.4f}")

    if use_sensor:
        print(f"\n{'='*70}")
        print(f"Sensor-tier stratified ({args.mode})")
        print(f"{'='*70}")
        print(f"{'Tier':<8} {'N':<6} {'PSNR':<12} {'SSIM':<12}")
        print('-' * 70)
        for tier_name, phones in SENSOR_TIERS:
            p = [r['psnr'] for r in records if r['tier'] == tier_name]
            s = [r['ssim'] for r in records if r['tier'] == tier_name]
            if p:
                print(f"{tier_name:<8} {len(p):<6} {np.mean(p):<12.3f} {np.mean(s):<12.4f}")

if __name__ == '__main__':
    main()
