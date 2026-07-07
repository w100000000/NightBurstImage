#!/usr/bin/env python3
"""Generate NBINet RKNN test input from REAL IMX585 validation data.

Usage (server):
    python gen_real_input.py [--index 0] [--height 544] [--width 960]

Output:
    short_cat.bin  — INT8 NHWC [1, 544, 960, 8]
    long_raw.bin   — INT8 NHWC [1, 544, 960, 4]
    gt_rgb.png     — Ground truth RGB for comparison (from GT_mid.npy demosaic)

Pipeline:
    1. python gen_real_input.py --index 5
    2. scp short_cat.bin long_raw.bin root@192.168.0.232:/tmp/
    3. ssh root@192.168.0.232 ./nbinet_infer /tmp/nbinet.rknn /tmp/short_cat.bin /tmp/long_raw.bin /tmp/output.rgb
    4. scp root@192.168.0.232:/tmp/output.rgb .
    5. python vis.py output.rgb test_output.png
    6. Compare test_output.png with gt_rgb.png
"""
import os
import sys
import numpy as np
import argparse
from PIL import Image

# Add project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# IMX585 Bayer patterns
def bayer_to_rggb(bayer_uint16, black_level=96, max_val=4095):
    """Convert IMX585 uint16 Bayer (RGGB) → [4, H, W] float32 [0, 1]"""
    bayer = bayer_uint16.astype(np.float32)
    bayer = np.clip((bayer - black_level) / (max_val - black_level), 0.0, 1.0)
    R  = bayer[0::2, 0::2]
    Gr = bayer[0::2, 1::2]
    Gb = bayer[1::2, 0::2]
    B  = bayer[1::2, 1::2]
    return np.stack([R, Gr, Gb, B], axis=0)  # [4, H_raw, W_raw]


def demosaic_rggb_to_rgb(bayer_float32, max_val=4095):
    """Demosaic RGGB Bayer float32 → RGB [3, H, W] float32 [0, 1]"""
    import cv2
    bayer_clip = np.clip(bayer_float32, 0, max_val)
    bayer_16 = (bayer_clip / max_val * 65535.0).astype(np.uint16)
    rgb_16 = cv2.cvtColor(bayer_16, cv2.COLOR_BayerRG2RGB_EA)
    rgb = rgb_16.astype(np.float32) / 65535.0
    return rgb.transpose(2, 0, 1)  # [3, H, W]


def float_to_int8_nhwc(data_4ch, zp=-128, scale=0.003922):
    """[C, H, W] float32 [0,1] → [1, H, W, C] INT8 NHWC"""
    nhwc = data_4ch.transpose(1, 2, 0)          # [H, W, C]
    nhwc = np.expand_dims(nhwc, 0)               # [1, H, W, C]
    int8 = np.clip(np.round(nhwc / scale + zp), -128, 127).astype(np.int8)
    return int8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0, help='Sample index in val split')
    parser.add_argument('--height', type=int, default=544, help='RAW crop height (must be mult of 8)')
    parser.add_argument('--width', type=int, default=960, help='RAW crop width')
    parser.add_argument('--dataset', type=str,
                        default='/home/dc2026_2/dataset/raw_d2hnet_asi585mc',
                        help='IMX585 dataset root')
    parser.add_argument('--crop_y', type=int, default=None, help='Manual crop y offset in RAW domain')
    parser.add_argument('--crop_x', type=int, default=None, help='Manual crop x offset in RAW domain')
    parser.add_argument('--black_level', type=int, default=96)
    parser.add_argument('--max_val', type=int, default=4095)
    args = parser.parse_args()

    # Read val split
    split_file = os.path.join(args.dataset, 'splits', 'val.txt')
    if os.path.exists(split_file):
        with open(split_file) as f:
            folders = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: find all scene directories
        folders = sorted([
            d for d in os.listdir(args.dataset)
            if os.path.isdir(os.path.join(args.dataset, d))
            and os.path.exists(os.path.join(args.dataset, d, 'S1.npy'))
        ])

    if args.index >= len(folders):
        print(f"ERROR: index {args.index} >= {len(folders)} samples")
        sys.exit(1)

    folder = folders[args.index]
    folder_path = os.path.join(args.dataset, folder)
    print(f"Sample [{args.index}]: {folder}")

    # Load S1, S2, L (uint16 Bayer 2160×3840)
    s1 = np.load(os.path.join(folder_path, 'S1.npy'))
    s2 = np.load(os.path.join(folder_path, 'S2.npy'))
    lf = np.load(os.path.join(folder_path, 'L.npy'))
    print(f"  S1: {s1.shape} {s1.dtype}, range [{s1.min()}, {s1.max()}]")
    print(f"  S2: {s2.shape} {s2.dtype}, range [{s2.min()}, {s2.max()}]")
    print(f"  L:  {lf.shape} {lf.dtype}, range [{lf.min()}, {lf.max()}]")

    # Convert to RGGB [4, 2160, 3840]
    s1_rggb = bayer_to_rggb(s1, args.black_level, args.max_val)
    s2_rggb = bayer_to_rggb(s2, args.black_level, args.max_val)
    l_rggb  = bayer_to_rggb(lf, args.black_level, args.max_val)
    print(f"  RGGB shapes: S1={s1_rggb.shape}, S2={s2_rggb.shape}, L={l_rggb.shape}")

    # Center crop to [4, height, width]
    H, W = s1_rggb.shape[1], s1_rggb.shape[2]
    if args.crop_y is None:
        cy = (H - args.height) // 2
    else:
        cy = args.crop_y
    if args.crop_x is None:
        cx = (W - args.width) // 2
    else:
        cx = args.crop_x
    print(f"  Crop: y={cy}, x={cx}  ({args.height}×{args.width} RAW → {args.height*2}×{args.width*2} RGB)")

    s1_crop = s1_rggb[:, cy:cy+args.height, cx:cx+args.width]
    s2_crop = s2_rggb[:, cy:cy+args.height, cx:cx+args.width]
    l_crop  = l_rggb[:,  cy:cy+args.height, cx:cx+args.width]

    # Short cat: [8, H, W]
    short_cat = np.concatenate([s1_crop, s2_crop], axis=0)

    # Quantize to INT8 NHWC
    short_int8 = float_to_int8_nhwc(short_cat)
    long_int8  = float_to_int8_nhwc(l_crop)

    # Save
    short_int8.tofile('short_cat.bin')
    long_int8.tofile('long_raw.bin')
    print(f"\nshort_cat.bin: {short_int8.shape} INT8, {short_int8.nbytes} bytes")
    print(f"long_raw.bin:  {long_int8.shape} INT8, {long_int8.nbytes} bytes")

    # Save GT RGB for comparison
    gt_path = os.path.join(folder_path, 'GT_mid.npy')
    if os.path.exists(gt_path):
        gt_bayer = np.load(gt_path)
        print(f"\nGT_mid: {gt_bayer.shape} {gt_bayer.dtype}")
        # GT is full-frame Bayer float32 — demosaic first, then crop in RGB domain
        gt_rgb_full = demosaic_rggb_to_rgb(gt_bayer, args.max_val)
        # Crop in RGB domain (2x RAW coordinates)
        gt_crop = gt_rgb_full[:,
                              cy*2:(cy+args.height)*2,
                              cx*2:(cx+args.width)*2]
        gt_hwc = gt_crop.transpose(1, 2, 0)
        gt_uint8 = np.clip(gt_hwc * 255, 0, 255).astype(np.uint8)
        Image.fromarray(gt_uint8).save('gt_rgb.png')
        print(f"  gt_rgb.png saved ({gt_uint8.shape[1]}×{gt_uint8.shape[0]})")
    else:
        print(f"\nWARNING: GT_mid.npy not found at {gt_path}")

    # Also save a simple S1 preview for reference (demosaiced single short frame)
    s1_bayer_full = s1.astype(np.float32)
    s1_preview = demosaic_rggb_to_rgb(s1_bayer_full, args.max_val)
    s1_crop_rgb = s1_preview[:,
                             cy*2:(cy+args.height)*2,
                             cx*2:(cx+args.width)*2]
    s1_hwc = s1_crop_rgb.transpose(1, 2, 0)
    s1_uint8 = np.clip(s1_hwc * 255, 0, 255).astype(np.uint8)
    Image.fromarray(s1_uint8).save('s1_preview.png')
    print(f"  s1_preview.png saved ({s1_uint8.shape[1]}×{s1_uint8.shape[0]}) — single short frame, no denoise")

    print("\nDone. Next steps:")
    print("  scp short_cat.bin long_raw.bin root@192.168.0.232:/tmp/")
    print("  ssh root@192.168.0.232 /tmp/nbinet_infer /tmp/nbinet.rknn /tmp/short_cat.bin /tmp/long_raw.bin /tmp/output.rgb")
    print("  scp root@192.168.0.232:/tmp/output.rgb .")
    print("  python vis.py output.rgb test_output.png")
    print("  Compare: test_output.png (NPU output) vs gt_rgb.png (ground truth) vs s1_preview.png (noisy single frame)")


if __name__ == '__main__':
    main()
