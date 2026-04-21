

# Physics-Conditioned Image Denoising

<p align="center">
  <img src="figures/Gemini_Generated_Image_xdbthnxdbthnxdbt.png" width="400"/>
  <br>
  <em>Image generated with <a href="https://gemini.google.com/">Google Gemini</a></em>
</p>

Investigating how camera physical parameters affect deep learning-based image denoising. We inject ISO (dynamic, per-shot) and sensor quality (static, per-device) as conditioning signals into a NAFNet-based U-Net via Adaptive Layer Normalization (AdaLN), and evaluate their impact on denoising performance using the SIDD Medium dataset.

## Setup

### Dataset

Download [SIDD Medium sRGB](https://abdokamel.github.io/sidd/) and extract to your preferred path.

### Checkpoints

Our models are initialized from a pretrained NAFBlock-based U-Net backbone provided by [Oh et al.](https://github.com/OBAKSA/CPADNet). We injected single-parameter conditioning (ISO-only or sensor-only, 9-dim), zero-initialize the new AdaLN layers, and fine-tune for 10K iterations on SIDD.

Download trained checkpoints from https://drive.google.com/drive/folders/1R_-tSXJ5pEbvV-FP2Y51ecVrOcPlErA-?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto and place as:
    checkpoints/
    ├── iso/
    │   └── best.pth
    └── sensor/
        └── best.pth

### Environment

    conda create -n nafnet_iso python=3.10
    conda activate nafnet_iso
    pip install torch torchvision opencv-python scikit-image numpy matplotlib

## Training

    # ISO conditioning
    cd iso
    python train_iso_only.py --total_iter 10000 --exp_name iso_run1

    # Sensor conditioning
    cd ../sensor
    python train_sensor.py --total_iter 10000 --exp_name sensor_run1

## Evaluation

### PSNR

    # ISO: stratified eval / zero-embedding ablation / sweep experiment
    cd iso
    python eval_iso_stratified.py --model iso_only
    python eval_nocond.py
    python eval_iso_sweep.py

    # Sensor: stratified eval / all-ones baseline / sweep experiment
    cd ../sensor
    python eval_sensor_stratified.py --model iso_only
    python eval_nocond_sensor.py
    python eval_sensor_sweep.py

### SSIM

    python eval_ssim_all.py --mode iso_cond
    python eval_ssim_all.py --mode iso_nocond
    python eval_ssim_all.py --mode sensor_cond
    python eval_ssim_all.py --mode sensor_nocond

### Fourier Analysis

    cd fourier
    python eval_fourier_iso.py
    python eval_fourier_sensor.py
