# VID-Trans-ReID Camera-Removed + CamStyle-Inspired Augmentation

This repo is a **camera-removed baseline** for VID-Trans-ReID with an additional **generator-free, tracklet-consistent CamStyle-inspired augmentation** during training.

It is intended for a clean experiment where you:
- remove camera metadata from the architecture
- keep the no-camera baseline evaluation protocol intact
- add only training-time camera-style augmentation
- do **not** add MetaBIN or GRL in this repo

## What this repo is

This is **not the original CamStyle** implementation based on CycleGAN camera-to-camera image translation.
Instead, this repo uses a **CamStyle-inspired photometric camera-style perturbation** that is:
- applied only during training
- shared across all frames in a sampled tracklet
- designed to improve robustness to camera-style appearance shifts

A tracklet-consistent transform is important for video Re-ID because it preserves temporal coherence inside the sampled clip.

## What was cleaned

- removed unused EMA logic from training
- set deterministic cuDNN mode cleanly with `benchmark=False`
- kept evaluation aligned with the no-camera baseline
- clarified naming so the method is described as **CamStyle-inspired**, not original CamStyle
- aligned `RandomErasing3` fill values with the normalized tensor range `[-1, 1]` by using a neutral fill of `(0.0, 0.0, 0.0)`

## Training

```bash
python VID_Trans_ReID.py   --Dataset_name Mars   --model_path /path/to/jx_vit_base_p16_224-80ecf9dd.pth   --output_dir ./output_camera_removed_camstyle   --epochs 120   --eval_every 10
```

## Disable CamStyle-inspired augmentation

To recover the plain no-camera baseline from the same repo:

```bash
python VID_Trans_ReID.py   --Dataset_name Mars   --model_path /path/to/jx_vit_base_p16_224-80ecf9dd.pth   --output_dir ./output_camera_removed_plain   --disable_camstyle   --epochs 120   --eval_every 10
```

## Test

```bash
python VID_Test.py   --Dataset_name Mars   --model_path ./output_camera_removed_camstyle/Mars_camera_removed_camstyle_best.pth
```

## Important note for thesis/report writing

Describe this method as:

> **generator-free, tracklet-consistent camera-style augmentation inspired by CamStyle**

That wording is accurate and avoids overstating it as the original CycleGAN-based CamStyle method.
