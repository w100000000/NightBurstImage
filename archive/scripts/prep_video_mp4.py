#!/usr/bin/env python3
"""
prep_video_mp4.py — 把视频缩放+降帧，输出 MP4 供 ELF2 板子硬解码播放

用法:
  python3 prep_video_mp4.py input.mp4 [output.mp4] [fps=6] [width=1024] [height=600]

输出:
  H.264 MP4, 1024x600, 指定帧率 (模拟 NPU 推理 ~6fps)

板子上:
  gst-play-1.0 /mnt/sdcard/video_1024x600.mp4
  或者:
  gst-launch-1.0 filesrc location=/mnt/sdcard/video_1024x600.mp4 ! qtdemux ! h264parse ! mppvideodec ! waylandsink
"""
import numpy as np
import cv2
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 prep_video_mp4.py <input.mp4> [output.mp4] [fps=6] [width=1024] [height=600]")
        sys.exit(1)

    video_in = sys.argv[1]
    video_out = sys.argv[2] if len(sys.argv) > 2 else None
    fps_out  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    fb_w     = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    fb_h     = int(sys.argv[5]) if len(sys.argv) > 5 else 600

    # Default output name
    if video_out is None:
        base = os.path.splitext(os.path.basename(video_in))[0]
        video_out = f"{base}_{fb_w}x{fb_h}_{fps_out}fps.mp4"

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_in}")
        sys.exit(1)

    fps_in  = cap.get(cv2.CAP_PROP_FPS)
    n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input:  {video_in}")
    print(f"        {n_frames_in} frames @ {fps_in:.1f} fps, {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"Output: {video_out}")
    print(f"        {fb_w}x{fb_h} @ {fps_out} fps")

    # Try H.264 codecs in order of preference
    codecs = [
        ('avc1', 'H.264 (avc1)'),
        ('H264', 'H.264 (H264)'),
        ('mp4v', 'MPEG-4 (mp4v)'),
        ('XVID', 'Xvid'),
    ]

    fourcc = None
    codec_name = None
    for cc, name in codecs:
        test_fourcc = cv2.VideoWriter_fourcc(*cc)
        test_path = video_out + ".test.mp4"
        writer = cv2.VideoWriter(test_path, test_fourcc, fps_out, (fb_w, fb_h))
        if writer.isOpened():
            fourcc = test_fourcc
            codec_name = name
            writer.release()
            os.remove(test_path)
            break
        writer.release()

    if fourcc is None:
        print("ERROR: No usable codec found. Try installing ffmpeg or gstreamer opencv backend.")
        sys.exit(1)

    print(f"Codec:  {codec_name}")

    writer = cv2.VideoWriter(video_out, fourcc, fps_out, (fb_w, fb_h))
    if not writer.isOpened():
        print(f"ERROR: Cannot open VideoWriter for {video_out}")
        sys.exit(1)

    # Frame step to match target fps
    step = max(1, int(fps_in / fps_out))
    frame_in = 0
    frame_out = 0

    print(f"Step:   keep 1 every {step} frames (30→{fps_out}fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_in % step != 0:
            frame_in += 1
            continue
        frame_in += 1

        # Resize to display resolution
        resized = cv2.resize(frame, (fb_w, fb_h), interpolation=cv2.INTER_LINEAR)
        writer.write(resized)
        frame_out += 1

        if frame_out % 20 == 0:
            print(f"  {frame_out} frames written...", end='\r')

    cap.release()
    writer.release()

    size_mb = os.path.getsize(video_out) / 1024 / 1024
    duration_s = frame_out / fps_out
    print(f"\nDone: {frame_out} frames, {size_mb:.1f} MB, ~{duration_s:.1f}s @ {fps_out}fps")
    print(f"Saved: {video_out}")

    print(f"\n=== Deploy to board ===")
    print(f"1. Copy to SD card:")
    print(f"   cp {video_out} /mnt/sdcard/")
    print(f"")
    print(f"2. Test playback on board (SSH):")
    print(f"   gst-play-1.0 /mnt/sdcard/{os.path.basename(video_out)}")
    print(f"")
    print(f"   Or with gst-launch:")
    print(f"   gst-launch-1.0 filesrc location=/mnt/sdcard/{os.path.basename(video_out)} ! qtdemux ! h264parse ! mppvideodec ! waylandsink")


if __name__ == '__main__':
    main()
