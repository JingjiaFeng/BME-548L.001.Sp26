"""
Fourier Domain Analysis — averaged over ALL val images per ISO bucket.
Same checkpoint, NoCond (zero embed) vs ISO Injection (real ISO).
"""

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import embed_param, get_psnr

from model_iso_only import ConditionedDenoiser
from data_sidd_iso import SIDDIsoDataset, get_split, ISO_MIN, ISO_MAX

ISO_CKPT = 'checkpoints/iso/best.pth'
PATCH_SIZE = 256

ISO_BUCKETS = [
    (50, 200, 'Low [50-200]'),
    (201, 800, 'Mid [200-800]'),
    (801, 3200, 'High [800-3200]'),
    (3201, 10000, 'VeryHigh [3200+]'),
]


def to_gray(img_chw):
    return 0.299 * img_chw[0] + 0.587 * img_chw[1] + 0.114 * img_chw[2]


def radial_profile(gray):
    """Compute radially averaged power spectrum from grayscale [H,W]."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    H, W = magnitude.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(int)
    max_r = min(cy, cx)
    power = magnitude ** 2
    profile = np.zeros(max_r)
    for i in range(max_r):
        mask = r == i
        if mask.any():
            profile[i] = power[mask].mean()
    return profile


def freq_band_energy(gray):
    """Compute low/mid/high frequency energy ratios."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    H, W = magnitude.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    power = magnitude ** 2
    total = power.sum()

    bands = {
        'Low': (0, max_r * 0.15),
        'Mid': (max_r * 0.15, max_r * 0.5),
        'High': (max_r * 0.5, max_r),
    }
    result = {}
    for name, (r_lo, r_hi) in bands.items():
        mask = (r >= r_lo) & (r < r_hi)
        result[name] = power[mask].sum() / total
    return result


def tile_image(img_chw, patch=256):
    _, H, W = img_chw.shape
    patches = []
    for y in range(0, H - patch + 1, patch):
        for x in range(0, W - patch + 1, patch):
            patches.append(img_chw[:, y:y+patch, x:x+patch])
    return patches


def main():
    device = torch.device('cuda')
    os.makedirs('fourier_results', exist_ok=True)

    # Load model
    print("Loading model...")
    model = ConditionedDenoiser(channels=32, param_dim=9).to(device)
    ck = torch.load(ISO_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck['denoiser_state_dict'])
    model.eval()

    _, val_folders = get_split()
    val_ds = SIDDIsoDataset(val_folders, mode='val')

    # Collect per-patch radial profiles and band energies, grouped by ISO bucket
    # Keys: (bucket_name, version_name) → list of profiles/energies
    radial_data = defaultdict(list)    # (bucket, version) → [profile, profile, ...]
    band_data = defaultdict(list)      # (bucket, version) → [{'Low':x, 'Mid':y, 'High':z}, ...]

    n_patches_total = 0

    with torch.no_grad():
        for i in range(len(val_ds)):
            noisy, gt, iso_embed, iso = val_ds[i]
            iso = int(iso)

            # Find bucket
            bucket_name = None
            for lo, hi, name in ISO_BUCKETS:
                if lo <= iso <= hi:
                    bucket_name = name
                    break
            if bucket_name is None:
                continue

            # Tile into patches
            noisy_patches = tile_image(noisy, PATCH_SIZE)
            gt_patches = tile_image(gt, PATCH_SIZE)

            for np_p, gt_p in zip(noisy_patches, gt_patches):
                inp = np_p.unsqueeze(0).to(device)

                # NoCond
                zero_emb = torch.zeros(1, 9).to(device)
                out_nc = torch.clamp(model(inp, zero_emb), 0, 1).squeeze(0).cpu().numpy()

                # ISO Injection
                iso_emb = iso_embed.unsqueeze(0).to(device)
                out_iso = torch.clamp(model(inp, iso_emb), 0, 1).squeeze(0).cpu().numpy()

                versions = {
                    'Noisy': np_p.numpy(),
                    'NoCond': out_nc,
                    'ISO Injection': out_iso,
                    'Ground Truth': gt_p.numpy(),
                }

                for vname, img in versions.items():
                    gray = to_gray(img)
                    radial_data[(bucket_name, vname)].append(radial_profile(gray))
                    band_data[(bucket_name, vname)].append(freq_band_energy(gray))

                n_patches_total += 1

            print(f"  [{i+1:3d}/{len(val_ds)}] iso={iso:5d} bucket={bucket_name} "
                  f"patches={len(noisy_patches)} total={n_patches_total}")

    print(f"\nTotal patches processed: {n_patches_total}")

    # Average radial profiles per (bucket, version)
    avg_radials = {}
    for key, profiles in radial_data.items():
        avg_radials[key] = np.mean(profiles, axis=0)

    # Average band energies per (bucket, version)
    avg_bands = {}
    for key, bands_list in band_data.items():
        avg_bands[key] = {
            'Low': np.mean([b['Low'] for b in bands_list]),
            'Mid': np.mean([b['Mid'] for b in bands_list]),
            'High': np.mean([b['High'] for b in bands_list]),
        }

    version_names = ['Noisy', 'NoCond', 'ISO Injection', 'Ground Truth']
    colors = {'Noisy': '#E63946', 'NoCond': '#808080',
              'ISO Injection': '#2E86AB', 'Ground Truth': '#06A77D'}

    # ==================== FIGURE 1: Averaged Radial Power Spectrum ====================
    print("\nPlotting averaged radial power spectrum...")
    fig, axes = plt.subplots(1, len(ISO_BUCKETS), figsize=(22, 6))

    for ax_idx, (lo, hi, bucket_name) in enumerate(ISO_BUCKETS):
        ax = axes[ax_idx]
        n_samples = len(radial_data.get((bucket_name, 'Noisy'), []))

        for vname in version_names:
            key = (bucket_name, vname)
            if key not in avg_radials:
                continue
            profile = avg_radials[key]
            freqs = np.arange(len(profile))
            profile_db = 10 * np.log10(profile + 1e-10)
            lw = 2.5 if vname in ('ISO Injection', 'Ground Truth') else 1.5
            ls = '-' if vname != 'NoCond' else '--'
            ax.plot(freqs, profile_db, color=colors[vname], lw=lw, ls=ls,
                    label=vname, alpha=0.9)

        max_r = len(avg_radials.get((bucket_name, 'Noisy'), [0]))
        ax.axvspan(0, max_r * 0.15, alpha=0.06, color='blue')
        ax.axvspan(max_r * 0.15, max_r * 0.5, alpha=0.06, color='green')
        ax.axvspan(max_r * 0.5, max_r, alpha=0.06, color='red')

        if ax_idx == 0:
            ax.text(max_r * 0.07, ax.get_ylim()[1] - 3, 'Low', ha='center',
                    fontsize=8, color='blue', alpha=0.7)
            ax.text(max_r * 0.32, ax.get_ylim()[1] - 3, 'Mid', ha='center',
                    fontsize=8, color='green', alpha=0.7)
            ax.text(max_r * 0.75, ax.get_ylim()[1] - 3, 'High', ha='center',
                    fontsize=8, color='red', alpha=0.7)

        ax.set_title(f'{bucket_name}\n(N={n_samples} patches)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Spatial Frequency', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('Power (dB)', fontsize=11)
        ax.grid(True, alpha=0.3)
        if ax_idx == len(ISO_BUCKETS) - 1:
            ax.legend(fontsize=9, loc='upper right', framealpha=0.95)

    fig.suptitle('Averaged Radial Power Spectrum: NoCond vs ISO Injection (same checkpoint)\n'
                 f'(Averaged over all {n_patches_total} val patches — '
                 'Blue closer to Green = conditioning preserves more detail)',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('fourier_results/radial_power_averaged.png', dpi=170, bbox_inches='tight')
    print("✅ Saved fourier_results/radial_power_averaged.png")
    plt.close()

    # ==================== FIGURE 2: High-Freq Preservation (Averaged) ====================
    print("Plotting high-frequency preservation (averaged)...")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    nocond_devs = []
    ours_devs = []
    bucket_labels = []
    n_counts = []

    for lo, hi, bucket_name in ISO_BUCKETS:
        key_gt = (bucket_name, 'Ground Truth')
        key_nc = (bucket_name, 'NoCond')
        key_iso = (bucket_name, 'ISO Injection')

        if key_gt not in avg_bands:
            continue

        gt_hf = avg_bands[key_gt]['High']
        nc_hf = avg_bands[key_nc]['High']
        iso_hf = avg_bands[key_iso]['High']

        nocond_devs.append(abs(nc_hf - gt_hf) / gt_hf * 100)
        ours_devs.append(abs(iso_hf - gt_hf) / gt_hf * 100)
        bucket_labels.append(bucket_name.replace(' ', '\n'))
        n_counts.append(len(band_data.get(key_gt, [])))

    x = np.arange(len(bucket_labels))
    w = 0.35

    bars_nc = ax.bar(x - w/2, nocond_devs, w, label='NoCond (zero embedding)',
                     color='#808080', alpha=0.88, edgecolor='black', linewidth=0.5)
    bars_o = ax.bar(x + w/2, ours_devs, w, label='ISO Injection (real ISO)',
                    color='#2E86AB', alpha=0.88, edgecolor='black', linewidth=0.5)

    for i in range(len(bucket_labels)):
        diff = nocond_devs[i] - ours_devs[i]
        top_y = max(nocond_devs[i], ours_devs[i]) + 1
        if diff > 0:
            ax.annotate(f'{diff:.1f}% closer\nto GT',
                        xy=(x[i], top_y), ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='#D62728',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#FFEB99', edgecolor='#D62728',
                                  linewidth=1.2))
        else:
            ax.annotate(f'{-diff:.1f}% worse',
                        xy=(x[i], top_y), ha='center', va='bottom',
                        fontsize=10, color='#555555',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#E8E8E8', edgecolor='#999999',
                                  linewidth=1))

        # N label
        ax.text(x[i], -1.5, f'N={n_counts[i]}', ha='center', fontsize=8, color='#666')

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_xlabel('ISO Bucket', fontsize=12)
    ax.set_ylabel('High-Freq Energy Deviation from GT (%)\n(lower = better)', fontsize=11)
    ax.set_title('High-Frequency Preservation: NoCond vs ISO Injection\n'
                 f'(Averaged over {n_patches_total} patches — pure conditioning contribution)',
                 fontsize=12, pad=10)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fourier_results/hf_preservation_averaged.png', dpi=170, bbox_inches='tight')
    print("✅ Saved fourier_results/hf_preservation_averaged.png")
    plt.close()

    # Summary table
    print(f"\n{'='*70}")
    print(f"High-Frequency Energy Deviation from GT (%) — Averaged over all patches")
    print(f"{'='*70}")
    print(f"{'Bucket':<20} {'N':<8} {'NoCond':<12} {'ISO Inj':<12} {'Δ':<12}")
    print('-' * 70)
    for i, (_, _, name) in enumerate(ISO_BUCKETS):
        if i < len(nocond_devs):
            diff = nocond_devs[i] - ours_devs[i]
            print(f"{name:<20} {n_counts[i]:<8} {nocond_devs[i]:<12.2f} {ours_devs[i]:<12.2f} "
                  f"{'+' if diff > 0 else ''}{diff:<12.2f}")


if __name__ == '__main__':
    main()