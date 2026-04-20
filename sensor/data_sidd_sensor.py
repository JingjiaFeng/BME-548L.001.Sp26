import os, sys, glob, random, cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import embed_param

SIDD_ROOT = '/scratch/ch594/datasets/SIDD/mnt/d/SIDD_Medium_Srgb/Data'

SENSOR_QUALITY = {
    'GP': 67.4,   # High tier
    'S6': 28.5,   # Mid tier
    'G4': 28.5,   # Mid tier
    'IP': 25.7,   # Low tier
    'N6': 20.8,   # Low tier
}

SENSOR_MIN = 20.0   # slightly below N6
SENSOR_MAX = 70.0   # slightly above GP

def get_tier(phone):
    if phone == 'GP':
        return 'High'
    elif phone in ('S6', 'G4'):
        return 'Mid'
    else:
        return 'Low'

def parse_scene_folder(folder_name):
    parts = folder_name.split('_')
    return {
        'scene_id': parts[0],
        'scene_num': parts[1],
        'phone': parts[2],
        'iso': int(parts[3]),
        'shutter': int(parts[4]),
        'illum': parts[5],
        'brightness': parts[6],
    }

def get_split(seed=42, val_ratio=0.2):
    all_folders = sorted(os.listdir(SIDD_ROOT))
    rng = random.Random(seed)
    rng.shuffle(all_folders)
    n_val = int(len(all_folders) * val_ratio)
    val_folders = sorted(all_folders[:n_val])
    train_folders = sorted(all_folders[n_val:])
    return train_folders, val_folders

class SIDDSensorDataset(Dataset):
    
    def __init__(self, folder_list, mode='train', patch_size=256):
        self.folder_list = folder_list
        self.mode = mode
        self.patch_size = patch_size
        
        self.samples = []
        for folder in folder_list:
            folder_path = os.path.join(SIDD_ROOT, folder)
            meta = parse_scene_folder(folder)
            noisy_files = sorted(glob.glob(os.path.join(folder_path, '*NOISY*.PNG')))
            gt_files = sorted(glob.glob(os.path.join(folder_path, '*GT*.PNG')))
            assert len(noisy_files) == len(gt_files)
            sensor_val = SENSOR_QUALITY[meta['phone']]
            for nf, gf in zip(noisy_files, gt_files):
                self.samples.append((nf, gf, sensor_val, meta['phone'], meta['iso']))
        
        print(f"[SIDDSensorDataset/{mode}] {len(folder_list)} scenes, {len(self.samples)} pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_img(self, path):
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0
    
    def __getitem__(self, idx):
        noisy_path, gt_path, sensor_val, phone, iso = self.samples[idx]
        noisy = self._load_img(noisy_path)
        gt = self._load_img(gt_path)
        
        H, W, _ = noisy.shape
        
        if self.mode == 'train':
            ps = self.patch_size
            top = random.randint(0, H - ps)
            left = random.randint(0, W - ps)
            noisy = noisy[top:top+ps, left:left+ps]
            gt = gt[top:top+ps, left:left+ps]
            
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=0).copy()
                gt = np.flip(gt, axis=0).copy()
            if random.random() < 0.5:
                noisy = np.flip(noisy, axis=1).copy()
                gt = np.flip(gt, axis=1).copy()
            k = random.randint(0, 3)
            if k > 0:
                noisy = np.rot90(noisy, k=k).copy()
                gt = np.rot90(gt, k=k).copy()
        else:
            H8 = H // 8 * 8
            W8 = W // 8 * 8
            noisy = noisy[:H8, :W8]
            gt = gt[:H8, :W8]
        
        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1))
        gt_t = torch.from_numpy(gt.transpose(2, 0, 1))
        
        sensor_tensor = torch.tensor([[float(sensor_val)]])
        sensor_embed = embed_param(sensor_tensor, SENSOR_MIN, SENSOR_MAX).squeeze(0)
        
        return noisy_t, gt_t, sensor_embed, sensor_val, phone, iso

if __name__ == '__main__':
    from collections import Counter
    
    train_folders, val_folders = get_split()
    print(f"\nSplit: {len(train_folders)} train / {len(val_folders)} val")
    
    train_tiers = Counter(get_tier(parse_scene_folder(f)['phone']) for f in train_folders)
    val_tiers = Counter(get_tier(parse_scene_folder(f)['phone']) for f in val_folders)
    print(f"\nTrain tier distribution: {sorted(train_tiers.items())}")
    print(f"Val tier distribution:   {sorted(val_tiers.items())}")
    
    val_phones = Counter(parse_scene_folder(f)['phone'] for f in val_folders)
    print(f"Val phone distribution:  {sorted(val_phones.items())}")
    
    train_ds = SIDDSensorDataset(train_folders, mode='train')
    noisy, gt, sensor_embed, sensor_val, phone, iso = train_ds[0]
    print(f"\n[train sample 0]")
    print(f"  noisy: {noisy.shape}")
    print(f"  sensor_embed: {sensor_embed.shape} = {sensor_embed.numpy().round(3)}")
    print(f"  sensor_val: {sensor_val} ({phone}, tier={get_tier(phone)})")
    print(f"  iso: {iso}")
    
    print("\n✅ Sensor dataset sanity check passed!")
