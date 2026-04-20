import math
import numpy as np
import torch

def get_psnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0 if np.max(img1) <= 1.0 else 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def embed_param(param: torch.Tensor, min_val, max_val) -> torch.Tensor:
    return torch.cat([
        (param / max_val),
        (param / max_val).sqrt(),
        (param / max_val) ** .25,
        (1. / (param / min_val)),
        (1. / (param / min_val)).sqrt(),
        (1. / (param / min_val)) ** .25,
        (param.log2() - math.log2(min_val)) / math.log2(max_val / min_val),
        (param.log2()).sin(),
        (param.log2()).cos()
    ], dim=1)
