#!/usr/bin/env python3
"""
Convert pipeline dump files to viewable PNG.
Usage:
  python3 view_dump.py <npu_frame0.rgb or fb_dump.rgb>

For NPU output (NCHW float32 3x272x480):
  python3 view_dump.py npu_frame0.rgb --npu

For fb_dump (XRGB8888 1024x600):
  python3 view_dump.py fb_dump.rgb --fb
"""
import numpy as np
import sys
import os
from PIL import Image

def view_npu(path):
    """NPU output: float32 NCHW [3, 272, 480], model outputs RGB"""
    data = np.fromfile(path, dtype=np.float32)
    print(f"File size: {len(data)} floats")
    print(f"Expected: {3 * 272 * 480} = {3*272*480}")

    if len(data) != 3 * 272 * 480:
        print(f"ERROR: unexpected size. Trying reshape anyway...")

    # NCHW -> CHW -> HWC
    rgb = data.reshape(3, 272, 480).transpose(1, 2, 0)  # [272, 480, 3]
    print(f"Value range: [{rgb.min():.4f}, {rgb.max():.4f}]")
    print(f"Mean per channel: R={rgb[:,:,0].mean():.4f} G={rgb[:,:,1].mean():.4f} B={rgb[:,:,2].mean():.4f}")

    # Clip and convert to uint8
    rgb = np.clip(rgb, 0, 1)
    img = (rgb * 255).astype(np.uint8)

    out = path.replace('.rgb', '.png')
    Image.fromarray(img).save(out)
    print(f"Saved: {out} ({img.shape[1]}x{img.shape[0]})")

    # Also try R/B swap version (in case channel order is BGR)
    bgr = rgb[:, :, ::-1]  # swap R<->B
    img_bgr = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
    out2 = path.replace('.rgb', '_bgr.png')
    Image.fromarray(img_bgr).save(out2)
    print(f"Saved (R/B swapped): {out2}")

def view_fb(path):
    """fb_dump: XRGB8888 uint32 1024x600, byte order BGRX"""
    data = np.fromfile(path, dtype=np.uint32)
    print(f"File size: {len(data)} pixels, expected {1024*600}={1024*600}")

    # Extract RGB from XRGB8888 (BGRX byte order)
    img = np.zeros((600, 1024, 3), dtype=np.uint8)
    for i, px in enumerate(data):
        y, x = divmod(i, 1024)
        if y >= 600:
            break
        img[y, x, 2] = (px >> 16) & 0xFF  # R (byte2)
        img[y, x, 1] = (px >> 8) & 0xFF   # G (byte1)
        img[y, x, 0] = px & 0xFF           # B (byte0)

    print(f"Value range: [{img.min()}, {img.max()}]")
    print(f"Mean: R={img[:,:,2].mean():.1f} G={img[:,:,1].mean():.1f} B={img[:,:,0].mean():.1f}")

    out = path.replace('.rgb', '.png')
    Image.fromarray(img).save(out)
    print(f"Saved: {out}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    if '--fb' in sys.argv:
        view_fb(path)
    else:
        view_npu(path)
