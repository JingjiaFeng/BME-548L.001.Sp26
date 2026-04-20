import os, sys, time, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_psnr

from model_iso_only import ConditionedDenoiser, load_warmstart
from data_sidd_sensor import SIDDSensorDataset, get_split

def quick_eval(model, val_loader, device, max_batches=32):
    
    model.eval()
    psnrs = []
    with torch.no_grad():
        for i, (noisy, gt, sensor_embed, _, _, _) in enumerate(val_loader):
            if i >= max_batches:
                break
            _, _, H, W = noisy.shape
            ch, cw = H // 2 - 128, W // 2 - 128
            noisy = noisy[:, :, ch:ch+256, cw:cw+256].to(device)
            gt = gt[:, :, ch:ch+256, cw:cw+256].to(device)
            sensor_embed = sensor_embed.to(device)

            out = torch.clamp(model(noisy, sensor_embed), 0, 1)
            for b in range(out.shape[0]):
                psnr = get_psnr(out[b].cpu().numpy(), gt[b].cpu().numpy())
                psnrs.append(psnr)
    model.train()
    return float(np.mean(psnrs)) if psnrs else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_iter', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Initial LR. NAFNet-AdaLN paper used 1e-6 for 200k iter from scratch; '
                             'we use higher LR for shorter fine-tune.')
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='iso_only_v1')
    parser.add_argument('--warmstart_ckpt', type=str,
                        default='../pretrained_models/SIDD_warmstart.pth')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda')
    work_dir = f'/scratch/ch594/projects/NAFNet-AdaLN_ISO/experiments/{args.exp_name}'
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(f'{work_dir}/ckpts', exist_ok=True)
    
    train_folders, val_folders = get_split()
    train_ds = SIDDSensorDataset(train_folders, mode='train', patch_size=256)
    val_ds = SIDDSensorDataset(val_folders, mode='val')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch
    train_iter = infinite_loader(train_loader)
    
    model = ConditionedDenoiser(channels=32, num_blocks=[4,4,4,8], param_dim=9).to(device)
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['denoiser_state_dict'])
        start_iter = ckpt['iteration']
    else:
        print(f"Warm-start from {args.warmstart_ckpt}")
        model = load_warmstart(model, args.warmstart_ckpt)
        model = model.to(device)
        start_iter = 0
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_iter, eta_min=args.lr * 0.01)
    
    log_csv = f'{work_dir}/train_log.csv'
    with open(log_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['iter', 'loss', 'lr', 'val_psnr'])
    
    model.train()
    loss_meter = []
    t0 = time.time()
    best_psnr = 0
    
    for it in range(start_iter, args.total_iter):
        noisy, gt, sensor_embed, _, _, _ = next(train_iter)
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        sensor_embed = sensor_embed.to(device, non_blocking=True)
        
        pred = model(noisy, sensor_embed)
        loss = F.l1_loss(pred, gt)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loss_meter.append(loss.item())
        
        if (it + 1) % 50 == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            avg_loss = np.mean(loss_meter[-50:])
            elapsed = time.time() - t0
            eta = elapsed / (it + 1 - start_iter) * (args.total_iter - it - 1)
            print(f"[iter {it+1:6d}/{args.total_iter}] "
                  f"loss={avg_loss:.5f} lr={cur_lr:.2e} "
                  f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min")
        
        if (it + 1) % args.eval_every == 0:
            val_psnr = quick_eval(model, val_loader, device, max_batches=32)
            print(f"    >>> val_psnr @ iter {it+1}: {val_psnr:.3f} dB")
            with open(log_csv, 'a', newline='') as f:
                csv.writer(f).writerow([it+1, np.mean(loss_meter[-50:]),
                                        optimizer.param_groups[0]['lr'], val_psnr])
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({'iteration': it+1,
                            'denoiser_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_psnr': val_psnr},
                           f'{work_dir}/ckpts/best.pth')
                print(f"    >>> New best! Saved to best.pth")
        
        if (it + 1) % args.save_every == 0:
            torch.save({'iteration': it+1,
                        'denoiser_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       f'{work_dir}/ckpts/iter_{it+1}.pth')
            print(f"    >>> Saved ckpt iter_{it+1}.pth")
    
    torch.save({'iteration': args.total_iter,
                'denoiser_state_dict': model.state_dict()},
               f'{work_dir}/ckpts/final.pth')
    print(f"\n✅ Training complete. Best val PSNR: {best_psnr:.3f} dB")
    print(f"Checkpoints in {work_dir}/ckpts/")
    print(f"Log in {log_csv}")

if __name__ == '__main__':
    main()
