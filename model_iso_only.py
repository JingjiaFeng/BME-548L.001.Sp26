import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from model import NAFBlock_AdaLN, Downsample, Upsample

class ConditionedDenoiser(nn.Module):
    
    def __init__(self, channels=32, num_blocks=[4, 4, 4, 8], param_dim=9):
        super(ConditionedDenoiser, self).__init__()
        
        self.param_dim = param_dim  # 9 for ISO-only
        
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.NAFBlock_enc1 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels, param_dim=param_dim) for _ in range(num_blocks[0])])
        self.NAFBlock_enc2 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 2, param_dim=param_dim) for _ in range(num_blocks[1])])
        self.NAFBlock_enc3 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 4, param_dim=param_dim) for _ in range(num_blocks[2])])
        self.NAFBlock_mid = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 8, param_dim=param_dim) for _ in range(num_blocks[3])])
        self.NAFBlock_dec3 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 4, param_dim=param_dim) for _ in range(num_blocks[2])])
        self.NAFBlock_dec2 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels * 2, param_dim=param_dim) for _ in range(num_blocks[1])])
        self.NAFBlock_dec1 = nn.ModuleList(
            [NAFBlock_AdaLN(c=channels, param_dim=param_dim) for _ in range(num_blocks[0])])
        
        self.down1_2 = Downsample(channels)
        self.down2_3 = Downsample(channels * 2)
        self.down3_4 = Downsample(channels * 4)
        
        self.up4_3 = Upsample(channels * 8)
        self.channel_reduce3 = nn.Conv2d(channels * 8, channels * 4, kernel_size=1, bias=False)
        self.up3_2 = Upsample(channels * 4)
        self.channel_reduce2 = nn.Conv2d(channels * 4, channels * 2, kernel_size=1, bias=False)
        self.up2_1 = Upsample(channels * 2)
        self.channel_reduce1 = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for module_list in [self.NAFBlock_enc1, self.NAFBlock_enc2, self.NAFBlock_enc3,
                            self.NAFBlock_mid,
                            self.NAFBlock_dec3, self.NAFBlock_dec2, self.NAFBlock_dec1]:
            for block in module_list:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
    
    def forward(self, inp, iso_embed):
        
        param_vector = iso_embed  # shape [B, 9], no phone/shutter concat
        
        x = self.conv_in(inp)
        
        for block in self.NAFBlock_enc1:
            x = block(x, param_vector)
        down1 = x
        x = self.down1_2(x)
        
        for block in self.NAFBlock_enc2:
            x = block(x, param_vector)
        down2 = x
        x = self.down2_3(x)
        
        for block in self.NAFBlock_enc3:
            x = block(x, param_vector)
        down3 = x
        x = self.down3_4(x)
        
        for block in self.NAFBlock_mid:
            x = block(x, param_vector)
        
        x = self.up4_3(x)
        x = torch.cat([x, down3], 1)
        x = self.channel_reduce3(x)
        for block in self.NAFBlock_dec3:
            x = block(x, param_vector)
        
        x = self.up3_2(x)
        x = torch.cat([x, down2], 1)
        x = self.channel_reduce2(x)
        for block in self.NAFBlock_dec2:
            x = block(x, param_vector)
        
        x = self.up2_1(x)
        x = torch.cat([x, down1], 1)
        x = self.channel_reduce1(x)
        for block in self.NAFBlock_dec1:
            x = block(x, param_vector)
        
        x = self.conv_out(x) + inp
        
        return x

def load_warmstart(iso_only_model, warmstart_ckpt_path):
    
    ckpt = torch.load(warmstart_ckpt_path, map_location='cpu')
    src_state = ckpt['denoiser_state_dict']
    dst_state = iso_only_model.state_dict()
    
    loaded, skipped = [], []
    for key, src_tensor in src_state.items():
        if key in dst_state:
            if dst_state[key].shape == src_tensor.shape:
                dst_state[key] = src_tensor
                loaded.append(key)
            else:
                skipped.append(f"{key}: shape mismatch {tuple(src_tensor.shape)} vs {tuple(dst_state[key].shape)}")
        else:
            skipped.append(f"{key}: not in ISO-only model (likely phone2vector)")
    
    iso_only_model.load_state_dict(dst_state)
    
    print(f"[warm-start] Loaded {len(loaded)} params from NAFNet-AdaLN checkpoint.")
    print(f"[warm-start] Skipped {len(skipped)} params (kept at zero-init):")
    for s in skipped[:10]:
        print(f"    {s}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped) - 10} more")
    
    return iso_only_model

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')  # so we can import from parent model.py
    
    print("Building ConditionedDenoiser...")
    model = ConditionedDenoiser(channels=32, num_blocks=[4, 4, 4, 8], param_dim=9)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params / 1e6:.2f}M")
    
    ckpt_path = '../pretrained_models/SIDD_warmstart.pth'
    import os
    if os.path.exists(ckpt_path):
        print(f"\nWarm-starting from {ckpt_path}...")
        model = load_warmstart(model, ckpt_path)
    else:
        print(f"\nCheckpoint not found at {ckpt_path}, skipping warm-start test.")
    
    model = model.cuda().eval()
    print("\nRunning forward pass test (B=2, 3x256x256)...")
    with torch.no_grad():
        x = torch.randn(2, 3, 256, 256).cuda()
        iso_embed = torch.randn(2, 9).cuda()  # fake ISO embedding
        out = model(x, iso_embed)
        print(f"Input shape:  {tuple(x.shape)}")
        print(f"Output shape: {tuple(out.shape)}")
        assert out.shape == x.shape, "Output shape mismatch!"
        print("\n✅ Sanity check passed!")
