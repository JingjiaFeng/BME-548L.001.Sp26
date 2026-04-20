import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_psnr, embed_param
from model_injection import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split as get_split_iso, ISO_MIN, ISO_MAX
from data_sidd_sensor import SIDDSensorDataset, get_split as get_split_sensor, SENSOR_MIN, SENSOR_MAX

ISO_CKPT = 'checkpoints/iso/best.pth'
SENSOR_CKPT = 'checkpoints/sensor/best.pth'

ISO_BUCKETS = [(50, 200, 'Low'), (201, 800, 'Mid'),
               (801, 3200, 'High'), (3201, 10000, 'VeryHigh')]

def tile_image(img, patch=256):
    _, H, W = img.shape
    patches = []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img[:, y:y+patch, x:x+patch])
    return patches

def run_eval(model, val_ds, embed_fn, device, label):
    
    records = []
    with torch.no_grad():
        for i in range(len(val_ds)):
            item = val_ds[i]
            if len(item) == 4:
                noisy, gt, real_embed, iso = item
            else:
                noisy, gt, real_embed, _, _, iso = item
            iso = int(iso)

            noisy_patches = tile_image(noisy)
            gt_patches = tile_image(gt)

            for np_p, gt_p in zip(noisy_patches, gt_patches):
                inp = np_p.unsqueeze(0).to(device)
                emb = embed_fn(real_embed, device)
                out = torch.clamp(model(inp, emb), 0, 1)
                psnr = get_psnr(out.squeeze(0).cpu().numpy(), gt_p.numpy())
                records.append((iso, psnr))

            if (i + 1) % 10 == 0:
                print(f"  [{label}] [{i+1}/{len(val_ds)}] running avg = {np.mean([r[1] for r in records]):.3f}")

    return records

def print_stratified(records, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"{'Bucket':<12} {'N':<6} {'PSNR':<12}")
    print('-' * 60)
    for lo, hi, name in ISO_BUCKETS:
        vals = [r[1] for r in records if lo <= r[0] <= hi]
        if vals:
            print(f"{name:<12} {len(vals):<6} {np.mean(vals):<12.3f}")
    overall = [r[1] for r in records]
    print('-' * 60)
    print(f"{'Overall':<12} {len(overall):<6} {np.mean(overall):<12.3f}")
    return np.mean(overall)

def main():
    device = torch.device('cuda')
    ones_emb = torch.ones(1, 9).to(device)

    print("\n" + "="*60)
    print("ISO MODEL")
    print("="*60)

    iso_model = ConditionedDenoiser(channels=32, param_dim=9).to(device)
    ck = torch.load(ISO_CKPT, map_location=device, weights_only=False)
    iso_model.load_state_dict(ck['denoiser_state_dict'])
    iso_model.eval()

    _, val_folders_iso = get_split_iso()
    val_ds_iso = SIDDIsoDataset(val_folders_iso, mode='val')

    print("\nRunning: ISO model + all-ones embedding...")
    iso_ones = run_eval(iso_model, val_ds_iso,
                        lambda real_emb, dev: ones_emb,
                        device, "ISO all-ones")
    iso_ones_psnr = print_stratified(iso_ones, "ISO model + All Ones (baseline)")

    print("\nRunning: ISO model + real ISO embedding...")
    iso_real = run_eval(iso_model, val_ds_iso,
                        lambda real_emb, dev: real_emb.unsqueeze(0).to(dev),
                        device, "ISO real")
    iso_real_psnr = print_stratified(iso_real, "ISO model + Real ISO")

    print("\n" + "="*60)
    print("SENSOR MODEL")
    print("="*60)

    sensor_model = ConditionedDenoiser(channels=32, param_dim=9).to(device)
    ck = torch.load(SENSOR_CKPT, map_location=device, weights_only=False)
    sensor_model.load_state_dict(ck['denoiser_state_dict'])
    sensor_model.eval()

    _, val_folders_sensor = get_split_sensor()
    val_ds_sensor = SIDDSensorDataset(val_folders_sensor, mode='val')

    print("\nRunning: Sensor model + all-ones embedding...")
    sensor_ones = run_eval(sensor_model, val_ds_sensor,
                           lambda real_emb, dev: ones_emb,
                           device, "Sensor all-ones")
    sensor_ones_psnr = print_stratified(sensor_ones, "Sensor model + All Ones (baseline)")

    print("\nRunning: Sensor model + real sensor embedding...")
    sensor_real = run_eval(sensor_model, val_ds_sensor,
                           lambda real_emb, dev: real_emb.unsqueeze(0).to(dev),
                           device, "Sensor real")
    sensor_real_psnr = print_stratified(sensor_real, "Sensor model + Real Sensor")

    print("\n" + "="*60)
    print("FINAL COMPARISON: Physical Information Contribution")
    print("="*60)
    print(f"{'Model':<15} {'All Ones':<12} {'Real Value':<12} {'Δ PSNR':<12} {'Conclusion':<20}")
    print('-' * 60)

    iso_delta = iso_real_psnr - iso_ones_psnr
    sensor_delta = sensor_real_psnr - sensor_ones_psnr

    iso_conclusion = "ISO IS USEFUL ✓" if iso_delta > 0.5 else "ISO not useful ✗"
    sensor_conclusion = "Sensor NOT useful ✗" if sensor_delta < 0.5 else "Sensor is useful ✓"

    print(f"{'ISO':<15} {iso_ones_psnr:<12.2f} {iso_real_psnr:<12.2f} "
          f"{'+' if iso_delta > 0 else ''}{iso_delta:<12.2f} {iso_conclusion}")
    print(f"{'Sensor':<15} {sensor_ones_psnr:<12.2f} {sensor_real_psnr:<12.2f} "
          f"{'+' if sensor_delta > 0 else ''}{sensor_delta:<12.2f} {sensor_conclusion}")

    print(f"\n{'='*70}")
    print("Per-bucket Δ PSNR (Real - All Ones)")
    print(f"{'='*70}")
    print(f"{'Bucket':<12} {'ISO Δ':<12} {'Sensor Δ':<12}")
    print('-' * 70)
    for lo, hi, name in ISO_BUCKETS:
        iso_o = [r[1] for r in iso_ones if lo <= r[0] <= hi]
        iso_r = [r[1] for r in iso_real if lo <= r[0] <= hi]
        sen_o = [r[1] for r in sensor_ones if lo <= r[0] <= hi]
        sen_r = [r[1] for r in sensor_real if lo <= r[0] <= hi]
        iso_d = np.mean(iso_r) - np.mean(iso_o) if iso_o and iso_r else 0
        sen_d = np.mean(sen_r) - np.mean(sen_o) if sen_o and sen_r else 0
        print(f"{name:<12} {'+' if iso_d > 0 else ''}{iso_d:<12.2f} "
              f"{'+' if sen_d > 0 else ''}{sen_d:<12.2f}")

if __name__ == '__main__':
    main()
