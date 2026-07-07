#!/usr/bin/env python3
"""Convert RKNN output.rgb (float32 NCHW) → raw framebuffer (XRGB8888).

Usage (server):
    python rgb_to_fb.py output.rgb fb.bin --fb_w 1024 --fb_h 600

Then on board:
    dd if=/tmp/fb.bin of=/dev/fb0
"""
import numpy as np
import sys
import argparse
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='RKNN output .rgb file (float32 NCHW)')
    parser.add_argument('output', help='Raw framebuffer output .bin')
    parser.add_argument('--fb_w', type=int, default=1024)
    parser.add_argument('--fb_h', type=int, default=600)
    parser.add_argument('--crop', action='store_true', help='Center-crop before resize')
    args = parser.parse_args()

    # Read float32 output
    data = np.fromfile(args.input, dtype=np.float32)

    # Try to infer shape
    total = data.size
    if total == 3 * 1088 * 1920:
        h, w = 1088, 1920
    elif total == 3 * 544 * 960:
        h, w = 544, 960
    else:
        raise ValueError(f"Cannot infer shape from {total} floats")

    img = data.reshape(1, 3, h, w)[0]  # [3, H, W] float32
    img = img.transpose(1, 2, 0)       # [H, W, 3]

    # Center-crop to 16:10 aspect if requested
    if args.crop:
        target_ratio = args.fb_w / args.fb_h
        current_ratio = w / h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            cx = (w - new_w) // 2
            img = img[:, cx:cx+new_w, :]
        else:
            new_h = int(w / target_ratio)
            cy = (h - new_h) // 2
            img = img[cy:cy+new_h, :, :]

    # Resize to framebuffer size
    pil_img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    pil_img = pil_img.resize((args.fb_w, args.fb_h), Image.BILINEAR)
    rgb = np.array(pil_img)

    # Convert to XRGB8888 (B G R X in little-endian)
    fb = np.zeros((args.fb_h, args.fb_w, 4), dtype=np.uint8)
    fb[:, :, 0] = rgb[:, :, 2]  # B
    fb[:, :, 1] = rgb[:, :, 1]  # G
    fb[:, :, 2] = rgb[:, :, 0]  # R
    fb[:, :, 3] = 0              # X (unused)

    fb.tofile(args.output)
    print(f"Saved {args.output}: {fb.shape[1]}×{fb.shape[0]} XRGB8888, {fb.nbytes} bytes")
    print(f"On board: dd if=/tmp/fb.bin of=/dev/fb0")

if __name__ == '__main__':
    main()
