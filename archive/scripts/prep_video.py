#!/usr/bin/env python3
"""
prep_video.py — 把视频转成 fb0 raw 帧序列 (使用 OpenCV, 不需要 ffmpeg)

用法:
  python3 prep_video.py input.mp4 frames/ [fps=30] [width=1024] [height=600]

输出:
  frames/000000.raw  frames/000001.raw  ...

每个 .raw 文件: W×H × 4 bytes XRGB8888 (BGRX byte order)
"""
import numpy as np
import cv2
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 prep_video.py <input.mp4> <output_dir> [fps=30] [width=1024] [height=600]")
        sys.exit(1)

    video   = sys.argv[1]
    out_dir = sys.argv[2]
    fps_out = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    fb_w    = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    fb_h    = int(sys.argv[5]) if len(sys.argv) > 5 else 600

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video}")
        sys.exit(1)

    fps_in  = cap.get(cv2.CAP_PROP_FPS)
    n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {video} ({n_frames_in} frames @ {fps_in:.1f} fps)")
    print(f"Output: {fb_w}x{fb_h} @ {fps_out} fps → {out_dir}/")

    # Figure out frame step to match target fps
    step = max(1, int(fps_in / fps_out))
    frame_n = 0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step != 0:
            idx += 1
            continue
        idx += 1

        # Resize to display resolution
        resized = cv2.resize(frame, (fb_w, fb_h), interpolation=cv2.INTER_LINEAR)
        # OpenCV reads as BGR → convert to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Pack as XRGB8888 BGRX: byte0=B, byte1=G, byte2=R, byte3=0
        out = np.zeros((fb_h, fb_w, 4), dtype=np.uint8)
        out[:, :, 0] = rgb[:, :, 2]  # B  (was R in RGB)
        out[:, :, 1] = rgb[:, :, 1]  # G
        out[:, :, 2] = rgb[:, :, 0]  # R  (was B in RGB)
        # byte3 stays 0 (X channel, unused)

        path = os.path.join(out_dir, f"{frame_n:06d}.raw")
        out.tofile(path)
        frame_n += 1

        if frame_n % 50 == 0:
            print(f"  {frame_n} frames...", end='\r')

    cap.release()
    size_mb = frame_n * fb_w * fb_h * 4 / 1024 / 1024
    print(f"\nDone: {frame_n} frames ({size_mb:.1f} MB) → {out_dir}/")

    # Also create a looping version (forward + reverse for smooth loop)
    print("\nTo create smooth loop: add --reverse and re-run")

    print(f"\n=== Deploy to board ===")
    print(f"1. Copy frames:")
    print(f"   scp -r {out_dir} root@<BOARD_IP>:/tmp/")
    print(f"")
    print(f"2. Copy fb_player (cross-compile first):")
    print(f"   cd ~/NightBurstImage/rknn_infer/pipeline && make fb_player")
    print(f"   scp fb_player root@<BOARD_IP>:/tmp/")
    print(f"")
    print(f"3. Run on board:")
    print(f"   killall weston 2>/dev/null")
    print(f"   /tmp/fb_player /tmp/{os.path.basename(out_dir.rstrip('/'))} {fps_out}")


if __name__ == '__main__':
    main()
