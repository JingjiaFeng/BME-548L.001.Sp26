import os, sys
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BaseDenoiser_SIDD
from utils import embed_param

from model_injection import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split, parse_scene_folder, ISO_MIN, ISO_MAX

TARGET_ISOS = [100, 800, 3200, 6400]

CROP_SIZE = 256

BASELINE_CKPT = '../pretrained_models/SIDD_baseline.pth'
ISO_CKPT = 'checkpoints/iso/best.pth'

def center_crop(img_chw, size=256):
    
    _, H, W = img_chw.shape
    y = H // 2 - size // 2
    x = W // 2 - size // 2
    return img_chw[:, y:y+size, x:x+size]

def tensor_to_uint8(t):
    
    arr = t.cpu().numpy().transpose(1, 2, 0)  # HWC
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def psnr(a, b):
    
    mse = np.mean((a - b) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)

def main():
    device = torch.device('cuda')
    os.makedirs('visual_results', exist_ok=True)
    
    print("Loading baseline model...")
    baseline = BaseDenoiser_SIDD(channels=32).to(device)
    ck = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    baseline.load_state_dict(ck['denoiser_state_dict'])
    baseline.eval()
    
    print("Loading ISO Injection model...")
    iso_model = ConditionedDenoiser(channels=32, param_dim=9).to(device)
    ck = torch.load(ISO_CKPT, map_location=device, weights_only=False)
    iso_model.load_state_dict(ck['denoiser_state_dict'])
    iso_model.eval()
    
    _, val_folders = get_split()
    val_ds = SIDDIsoDataset(val_folders, mode='val')
    
    picks = {}
    for i in range(len(val_ds)):
        _, _, _, iso = val_ds[i]
        iso = int(iso)
        if iso in TARGET_ISOS and iso not in picks:
            picks[iso] = i
        if len(picks) == len(TARGET_ISOS):
            break
    
    print(f"\nPicked images: {picks}")
    
    crops_all = []  # will be [ {noisy, baseline, ours, gt, iso}, ... ]
    
    with torch.no_grad():
        for target_iso in TARGET_ISOS:
            idx = picks[target_iso]
            noisy, gt, iso_embed, iso = val_ds[idx]
            folder = val_folders[idx // 2]
            phone = parse_scene_folder(folder)['phone']
            
            noisy_c = center_crop(noisy, CROP_SIZE).unsqueeze(0).to(device)
            gt_c = center_crop(gt, CROP_SIZE).unsqueeze(0).to(device)
            iso_embed_b = iso_embed.unsqueeze(0).to(device)
            
            dummy_param = torch.zeros(1, 18).to(device)
            dummy_phone = torch.zeros(1, 1).long().to(device)
            out_base = torch.clamp(baseline(noisy_c, dummy_param, dummy_phone), 0, 1)
            
            out_iso = torch.clamp(iso_model(noisy_c, iso_embed_b), 0, 1)
            
            noisy_np = noisy_c.squeeze(0).cpu().numpy()
            gt_np = gt_c.squeeze(0).cpu().numpy()
            base_np = out_base.squeeze(0).cpu().numpy()
            iso_np = out_iso.squeeze(0).cpu().numpy()
            
            psnr_noisy = psnr(noisy_np, gt_np)
            psnr_base = psnr(base_np, gt_np)
            psnr_iso = psnr(iso_np, gt_np)
            
            print(f"\nISO={target_iso} ({phone}):")
            print(f"  Noisy PSNR:        {psnr_noisy:.2f} dB")
            print(f"  Baseline PSNR:     {psnr_base:.2f} dB")
            print(f"  ISO Injection PSNR: {psnr_iso:.2f} dB  (+{psnr_iso-psnr_base:.2f} dB)")
            
            crops_all.append({
                'iso': target_iso,
                'phone': phone,
                'noisy': noisy_c.squeeze(0),
                'baseline': out_base.squeeze(0),
                'ours': out_iso.squeeze(0),
                'gt': gt_c.squeeze(0),
                'psnr_noisy': psnr_noisy,
                'psnr_base': psnr_base,
                'psnr_ours': psnr_iso,
            })
            
            iso_dir = f'visual_results/ISO_{target_iso}_{phone}'
            os.makedirs(iso_dir, exist_ok=True)
            cv2.imwrite(f'{iso_dir}/noisy.png', tensor_to_uint8(crops_all[-1]['noisy']))
            cv2.imwrite(f'{iso_dir}/baseline.png', tensor_to_uint8(crops_all[-1]['baseline']))
            cv2.imwrite(f'{iso_dir}/ours.png', tensor_to_uint8(crops_all[-1]['ours']))
            cv2.imwrite(f'{iso_dir}/gt.png', tensor_to_uint8(crops_all[-1]['gt']))
            print(f"  → saved to {iso_dir}/")
    
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(TARGET_ISOS), 4, figsize=(14, 3.2 * len(TARGET_ISOS)))
    
    col_titles = ['Noisy Input', 'Baseline', 'ISO Injection (ours)', 'Ground Truth']
    
    for r, crop in enumerate(crops_all):
        imgs = [crop['noisy'], crop['baseline'], crop['ours'], crop['gt']]
        psnrs = [crop['psnr_noisy'], crop['psnr_base'], crop['psnr_ours'], None]
        
        for c in range(4):
            ax = axes[r, c]
            img_rgb = imgs[c].cpu().numpy().transpose(1, 2, 0)
            img_rgb = np.clip(img_rgb, 0, 1)
            ax.imshow(img_rgb)
            ax.set_xticks([]); ax.set_yticks([])
            
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12, fontweight='bold', pad=8)
            
            if c == 0:
                ax.set_ylabel(f"True ISO = {crop['iso']}\n({crop['phone']})",
                              fontsize=11, fontweight='bold')
            
            if psnrs[c] is not None:
                color = '#D62728' if c == 2 else '#333333'
                weight = 'bold' if c == 2 else 'normal'
                ax.text(0.03, 0.97, f'{psnrs[c]:.2f} dB',
                        transform=ax.transAxes, fontsize=10.5,
                        color='white', fontweight=weight,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor=color, edgecolor='none', alpha=0.88))
    
    fig.suptitle('Visual Comparison: Baseline vs ISO Injection Across ISO Levels',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('visual_results/grid_comparison.png', dpi=170, bbox_inches='tight')
    print(f"\n✅ Grid figure saved to visual_results/grid_comparison.png")

if __name__ == '__main__':
    main()
