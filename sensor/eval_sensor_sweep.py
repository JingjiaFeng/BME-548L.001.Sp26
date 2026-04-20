import os, sys, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import embed_param, get_psnr

from model_iso_only import ConditionedDenoiser
from data_sidd_sensor import SIDDSensorDataset, get_split, get_tier, SENSOR_MIN, SENSOR_MAX

CKPT = 'checkpoints/sensor/best.pth'

INJECT_SENSORS = {
    'High (67.4)': 67.4,
    'Mid (28.5)': 28.5,
    'Low (20.8)': 20.8,
}

ISO_BUCKETS = [(50, 200, 'Low'), (201, 800, 'Mid'),
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
    val_ds = SIDDSensorDataset(val_folders, mode='val')

    sensor_embeds = {}
    for name, val in INJECT_SENSORS.items():
        t = torch.tensor([[float(val)]])
        sensor_embeds[name] = embed_param(t, SENSOR_MIN, SENSOR_MAX).to(device)

    records = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            noisy, gt, _, sensor_val, phone, iso = val_ds[i]
            true_tier = get_tier(phone)

            noisy_patches = tile_image(noisy)
            gt_patches = tile_image(gt)

            for inj_name, inj_emb in sensor_embeds.items():
                psnrs = []
                for np_p, gt_p in zip(noisy_patches, gt_patches):
                    inp = np_p.unsqueeze(0).to(device)
                    out = torch.clamp(model(inp, inj_emb), 0, 1)
                    d = out.squeeze(0).cpu().numpy()
                    g = gt_p.numpy()
                    psnrs.append(get_psnr(d, g))

                records.append({
                    'img_idx': i,
                    'true_phone': phone,
                    'true_tier': true_tier,
                    'true_sensor': sensor_val,
                    'iso': int(iso),
                    'injected_sensor': inj_name,
                    'mean_psnr': float(np.mean(psnrs)),
                })

            print(f"  [{i+1:3d}/{len(val_ds)}] phone={phone} tier={true_tier} iso={iso:5d} | "
                  f"High={records[-3]['mean_psnr']:.2f}  "
                  f"Mid={records[-2]['mean_psnr']:.2f}  "
                  f"Low={records[-1]['mean_psnr']:.2f}  "
                  f"spread={max(records[-3]['mean_psnr'],records[-2]['mean_psnr'],records[-1]['mean_psnr']) - min(records[-3]['mean_psnr'],records[-2]['mean_psnr'],records[-1]['mean_psnr']):.3f}")

    with open('results_sensor_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"\nSaved {len(records)} records to results_sensor_sweep.csv")

    print(f"\n{'='*70}")
    print("Overall PSNR by injected sensor value")
    print(f"{'='*70}")
    print(f"{'Injected':<20} {'Mean PSNR':<12} {'Std':<10}")
    print('-' * 70)
    for inj_name in INJECT_SENSORS:
        vals = [r['mean_psnr'] for r in records if r['injected_sensor'] == inj_name]
        print(f"{inj_name:<20} {np.mean(vals):<12.3f} {np.std(vals):<10.3f}")

    all_means = [np.mean([r['mean_psnr'] for r in records if r['injected_sensor'] == n])
                 for n in INJECT_SENSORS]
    print(f"\nOverall spread: {max(all_means) - min(all_means):.3f} dB")

    print(f"\n{'='*70}")
    print("PSNR by ISO bucket × injected sensor value")
    print(f"{'='*70}")
    print(f"{'ISO Bucket':<12} " + "".join(f"{n:<20}" for n in INJECT_SENSORS) + "Spread")
    print('-' * 70)
    for lo, hi, bname in ISO_BUCKETS:
        row = f"{bname:<12} "
        means = []
        for inj_name in INJECT_SENSORS:
            vals = [r['mean_psnr'] for r in records
                    if r['injected_sensor'] == inj_name and lo <= r['iso'] <= hi]
            m = np.mean(vals) if vals else 0
            means.append(m)
            row += f"{m:<20.3f}"
        row += f"{max(means) - min(means):.3f}"
        print(row)

    print(f"\n{'='*70}")
    print("COMPARISON: ISO sweep spread vs Sensor sweep spread")
    print(f"{'='*70}")
    print(f"{'Bucket':<12} {'ISO sweep spread':<20} {'Sensor sweep spread':<20}")
    print('-' * 70)
    iso_spreads = {'Low': 3.03, 'Mid': 1.82, 'High': 1.95, 'VeryHigh': 2.86}
    for lo, hi, bname in ISO_BUCKETS:
        means = []
        for inj_name in INJECT_SENSORS:
            vals = [r['mean_psnr'] for r in records
                    if r['injected_sensor'] == inj_name and lo <= r['iso'] <= hi]
            if vals:
                means.append(np.mean(vals))
        sensor_spread = max(means) - min(means) if len(means) >= 2 else 0
        iso_sp = iso_spreads.get(bname, 0)
        print(f"{bname:<12} {iso_sp:<20.3f} {sensor_spread:<20.3f}")

if __name__ == '__main__':
    main()
