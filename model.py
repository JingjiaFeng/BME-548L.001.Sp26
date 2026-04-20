import torch
import torch.nn as nn

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFBlock_AdaLN(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., param_dim=9):
        super().__init__()
        dw = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, 1, 1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.adaLN_modulation = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(param_dim, c, bias=True),
            nn.SiLU(),
            nn.Linear(c, 4 * c, bias=True))
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp, control_param):
        shift_a, scale_a, shift_m, scale_m = self.adaLN_modulation(
            control_param).unsqueeze(-1).unsqueeze(-1).chunk(4, dim=1)
        x = modulate(self.norm1(inp), shift_a, scale_a)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = modulate(self.norm2(y), shift_m, scale_m)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class BaseDenoiser_SIDD(nn.Module):
    def __init__(self, channels=32, num_blocks=[4, 4, 4, 8]):
        super().__init__()
        self.conv_in = nn.Conv2d(3, channels, 3, 1, 1, bias=True)
        self.conv_out = nn.Conv2d(channels, 3, 3, 1, 1, bias=True)
        self.enc1 = nn.ModuleList([NAFBlock(channels) for _ in range(num_blocks[0])])
        self.enc2 = nn.ModuleList([NAFBlock(channels * 2) for _ in range(num_blocks[1])])
        self.enc3 = nn.ModuleList([NAFBlock(channels * 4) for _ in range(num_blocks[2])])
        self.mid = nn.ModuleList([NAFBlock(channels * 8) for _ in range(num_blocks[3])])
        self.dec3 = nn.ModuleList([NAFBlock(channels * 4) for _ in range(num_blocks[2])])
        self.dec2 = nn.ModuleList([NAFBlock(channels * 2) for _ in range(num_blocks[1])])
        self.dec1 = nn.ModuleList([NAFBlock(channels) for _ in range(num_blocks[0])])
        self.down1 = Downsample(channels)
        self.down2 = Downsample(channels * 2)
        self.down3 = Downsample(channels * 4)
        self.up3 = Upsample(channels * 8)
        self.reduce3 = nn.Conv2d(channels * 8, channels * 4, 1, bias=False)
        self.up2 = Upsample(channels * 4)
        self.reduce2 = nn.Conv2d(channels * 4, channels * 2, 1, bias=False)
        self.up1 = Upsample(channels * 2)
        self.reduce1 = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, inp, param=None, phone=None):
        x = self.conv_in(inp)
        for blk in self.enc1: x = blk(x)
        d1 = x; x = self.down1(x)
        for blk in self.enc2: x = blk(x)
        d2 = x; x = self.down2(x)
        for blk in self.enc3: x = blk(x)
        d3 = x; x = self.down3(x)
        for blk in self.mid: x = blk(x)
        x = self.reduce3(torch.cat([self.up3(x), d3], 1))
        for blk in self.dec3: x = blk(x)
        x = self.reduce2(torch.cat([self.up2(x), d2], 1))
        for blk in self.dec2: x = blk(x)
        x = self.reduce1(torch.cat([self.up1(x), d1], 1))
        for blk in self.dec1: x = blk(x)
        return self.conv_out(x) + inp
