import os, sys, argparse, csv, time
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import embed_param, get_psnr

from model_injection import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split, parse_scene_folder, ISO_MIN, ISO_MAX

INJECT_ISOS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 10000]

def tile_image(img_chw, patch=256):
    _, H, W = img_chw.shape
    patches, coords = [], []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img_chw[:, y:y+patch, x:x+patch])
            coords.append((y, x))
    return patches, coords

def compute_ssim_simple(img1, img2):
    
    from skimage.metrics import structural_similarity
    i1 = img1.transpose(1, 2, 0)
    i2 = img2.transpose(1, 2, 0)
    return structural_similarity(i1, i2, channel_axis=2, data_range=1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default='results_iso_sweep.csv')
    parser.add_argument('--compute_ssim', action='store_true',
                        help='Also compute SSIM (+ ~25% time)')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    
    print(f"Loading model from {args.ckpt}")
    model = ConditionedDenoiser(channels=32, param_dim=9).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['denoiser_state_dict'])
    model.eval()
    
    _, val_folders = get_split()
    val_ds = SIDDIsoDataset(val_folders, mode='val')
    
    iso_embeds = {}
    for iso in INJECT_ISOS:
        iso_t = torch.tensor([[float(iso)]])
        iso_embeds[iso] = embed_param(iso_t, ISO_MIN, ISO_MAX).to(device)  # [1, 9]
    
    records = []
    t_start = time.time()
    
    with torch.no_grad():
        for img_idx, (noisy, gt, _, true_iso) in enumerate(val_ds):
            folder = val_folders[img_idx // 2]
            phone = parse_scene_folder(folder)['phone']
            
            noisy_patches, _ = tile_image(noisy, patch=256)
            gt_patches, _ = tile_image(gt, patch=256)
            n_patches = len(noisy_patches)
            
            for inj_iso in INJECT_ISOS:
                iso_emb = iso_embeds[inj_iso]
                psnrs = []
                ssims = []
                
                for np_p, gt_p in zip(noisy_patches, gt_patches):
                    inp = np_p.unsqueeze(0).to(device)
                    out = torch.clamp(model(inp, iso_emb), 0, 1)
                    
                    d = out.squeeze(0).cpu().numpy()
                    g = gt_p.numpy()
                    psnrs.append(get_psnr(d, g))
                    if args.compute_ssim:
                        ssims.append(compute_ssim_simple(d, g))
                
                mean_psnr = float(np.mean(psnrs))
                mean_ssim = float(np.mean(ssims)) if ssims else None
                
                records.append({
                    'img_idx': img_idx,
                    'scene': folder,
                    'phone': phone,
                    'true_iso': int(true_iso),
                    'injected_iso': inj_iso,
                    'n_patches': n_patches,
                    'mean_psnr': mean_psnr,
                    'mean_ssim': mean_ssim if mean_ssim else -1,
                })
            
            elapsed = time.time() - t_start
            eta = elapsed / (img_idx + 1) * (len(val_ds) - img_idx - 1)
            row = [r for r in records[-len(INJECT_ISOS):]]
            peak = max(row, key=lambda x: x['mean_psnr'])
            print(f"[{img_idx+1:3d}/{len(val_ds)}] true_iso={true_iso:5d} phone={phone} "
                  f"| peak@inj={peak['injected_iso']:>5d} psnr={peak['mean_psnr']:.2f} "
                  f"| elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
    
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved {len(records)} records to {args.out_csv}")
    
    BUCKETS = [(50, 200, 'Low'), (201, 800, 'Mid'),
               (801, 3200, 'High'), (3201, 10000, 'VeryHigh')]
    
    print("\n" + "="*100)
    print("Mean PSNR matrix: rows = true ISO bucket, cols = injected ISO")
    print("="*100)
    print(f"{'bucket':<12}" + "".join(f"{iso:>9}" for iso in INJECT_ISOS))
    print("-"*100)
    for lo, hi, name in BUCKETS:
        bucket_records = [r for r in records if lo <= r['true_iso'] <= hi]
        if not bucket_records:
            continue
        row = f"{name:<12}"
        for inj in INJECT_ISOS:
            vals = [r['mean_psnr'] for r in bucket_records if r['injected_iso'] == inj]
            row += f"{np.mean(vals):>9.2f}" if vals else f"{'--':>9}"
        print(row)
    print("="*100)
    print("  ↑ Read each ROW left-to-right: where does PSNR peak?")
    print("    Peak position should be close to that row's true ISO range.")

if __name__ == '__main__':
    main()
