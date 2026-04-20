import os, sys, argparse, math
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import embed_param, get_psnr

from model_injection import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split, parse_scene_folder, ISO_MIN, ISO_MAX

def phone2num(val):
    return {'G4': 0, 'GP': 1, 'IP': 2, 'N6': 3, 'S6': 4}[val]

def tile_image(img_chw, patch=256):
    
    _, H, W = img_chw.shape
    patches, coords = [], []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img_chw[:, y:y+patch, x:x+patch])
            coords.append((y, x))
    return patches, coords

def run_model(model, noisy_patch, iso, phone_str, model_type):
    
    B = 1
    noisy = noisy_patch.unsqueeze(0).cuda()  # [1,3,256,256]
    
    if model_type == 'baseline':
        dummy_param = torch.zeros(1, 18).cuda()
        dummy_phone = torch.zeros(1, 1).long().cuda()
        out = model(noisy, dummy_param, dummy_phone)
        iso_t = torch.tensor([[float(iso)]]).cuda()
        shutter_t = torch.tensor([[1000.0]]).cuda()
        iso_emb = embed_param(iso_t, 50, 10000)
        shutter_emb = embed_param(shutter_t, 20, 8460)
        param = torch.cat([iso_emb, shutter_emb], dim=1)  # [1,18]
        phone_idx = torch.tensor([[phone2num(phone_str)]]).int().cuda()
        out = model(noisy, param, phone_idx)
    elif model_type == 'iso_only':
        iso_t = torch.tensor([[float(iso)]]).cuda()
        iso_emb = embed_param(iso_t, ISO_MIN, ISO_MAX).cuda()  # [1,9]
        out = model(noisy, iso_emb)
    
    return torch.clamp(out, 0., 1.)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'iso_only'])
    parser.add_argument('--ckpt', type=str, default=None,
    parser.add_argument('--out_csv', type=str, default=None)
    args = parser.parse_args()
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.ckpt is None:
        auto_ckpt = {
            'baseline': os.path.join(repo_root, 'pretrained_models/SIDD_baseline.pth'),
        }
        args.ckpt = auto_ckpt.get(args.model)
        if args.ckpt is None:
            raise ValueError("--ckpt required for iso_only")
    
    if args.out_csv is None:
        args.out_csv = f'./results_{args.model}.csv'
    
    print(f"Building {args.model} model...")
    if args.model == 'baseline':
        model = BaseDenoiser_SIDD(channels=32).cuda()
    elif args.model == 'iso_only':
        model = ConditionedDenoiser(channels=32, param_dim=9).cuda()
    
    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['denoiser_state_dict'])
    model.eval()
    
    _, val_folders = get_split()
    val_ds = SIDDIsoDataset(val_folders, mode='val')
    
    records = []  # list of (iso, phone, psnr)
    
    with torch.no_grad():
        for i, (noisy, gt, _, iso) in enumerate(val_ds):
            folder = val_folders[i // 2]  # 2 pairs per folder
            phone_str = parse_scene_folder(folder)['phone']
            
            noisy_patches, _ = tile_image(noisy, patch=256)
            gt_patches, _ = tile_image(gt, patch=256)
            
            psnrs = []
            for np_patch, gt_patch in zip(noisy_patches, gt_patches):
                denoised = run_model(model, np_patch, iso, phone_str, args.model)
                d = denoised.squeeze(0).cpu().numpy()
                g = gt_patch.numpy()
                psnr = get_psnr(d, g)
                psnrs.append(psnr)
                records.append((int(iso), phone_str, float(psnr)))
            
            print(f"  [{i+1:3d}/{len(val_ds)}] iso={iso:5d} phone={phone_str} "
                  f"n_patch={len(psnrs):3d} mean_psnr={np.mean(psnrs):.2f}")
    
    import csv
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iso', 'phone', 'psnr'])
        w.writerows(records)
    print(f"\nSaved {len(records)} patch records to {args.out_csv}")
    
    BUCKETS = [(50, 200, 'Low'), (201, 800, 'Mid'),
               (801, 3200, 'High'), (3201, 10000, 'VeryHigh')]
    
    print(f"\n{'='*60}")
    print(f"ISO-stratified PSNR ({args.model})")
    print(f"{'='*60}")
    print(f"{'Bucket':<10} {'ISO range':<15} {'N':<6} {'mean PSNR':<12} {'std':<8}")
    print('-' * 60)
    for lo, hi, name in BUCKETS:
        vals = [r[2] for r in records if lo <= r[0] <= hi]
        if len(vals) > 0:
            print(f"{name:<10} [{lo:>5}, {hi:>5}]  {len(vals):<6} "
                  f"{np.mean(vals):<12.3f} {np.std(vals):<8.3f}")
        else:
            print(f"{name:<10} [{lo:>5}, {hi:>5}]  0      (no samples)")
    
    overall = [r[2] for r in records]
    print('-' * 60)
    print(f"{'Overall':<10} {'':<15} {len(overall):<6} "
          f"{np.mean(overall):<12.3f} {np.std(overall):<8.3f}")

if __name__ == '__main__':
    main()
