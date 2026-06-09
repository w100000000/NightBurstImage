#!/usr/bin/env python3
"""
NBINet 训练数据采集 Pipeline — ELF2 + OV13855

架构:
    Sensor (V4L2) → 环形缓冲 → S-L-S triplet → 写盘

用法:
    # 采 100 组训练数据
    python3 collect.py --count 100 --output /data/train

    # 连续采集 (Ctrl+C 停止)
    python3 collect.py --continuous --output /data/train

    # 预览模式 (只显示不存盘)
    python3 collect.py --preview

硬件:
    ELF2 RK3588 + OV13855 @ 4224×3136 × 30fps
    VTS=14314 → ~12fps (长帧83ms), 短帧16ms, 交替曝光
"""

import os
import sys
import time
import argparse
import subprocess

# Add pipeline dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raw_processor import TripletSaver


# ── Sensor control ───────────────────────────────────────────

SUBDEV = '/dev/v4l-subdev2'
VIDEO  = '/dev/video0'

SENSOR_30FPS_SETUP = [
    ['v4l2-ctl', '-d', SUBDEV, '--set-subdev-fmt',
     'pad=0,width=4224,height=3136,code=0x3007'],
]

VBLANK_LONG = 11178   # VTS = 3136 + 11178 = 14314, ~83ms frame (12fps)
EXPOSURE_SHORT = 3000   # ~16ms
EXPOSURE_LONG  = 14300  # ~82ms


def run_cmd(cmd, check=True):
    """Run a shell command, print output on error."""
    print('  $', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print('  ERROR:', result.stderr.strip())
    return result


def setup_sensor():
    """One-time sensor initialization: format + VTS."""
    print('[Setup] Configuring OV13855...')

    # Set format to 30fps full-res
    run_cmd(SENSOR_30FPS_SETUP[0])
    time.sleep(0.3)

    # Set VTS for ~83ms frame (long exposure capable)
    run_cmd(['v4l2-ctl', '-d', SUBDEV, '-c',
             'vertical_blanking=%d' % VBLANK_LONG])
    time.sleep(0.2)

    # Verify
    result = subprocess.run(['v4l2-ctl', '-d', SUBDEV, '-C', 'vertical_blanking'],
                            capture_output=True, text=True)
    print('  VTS status:', result.stdout.strip())

    result = subprocess.run(['v4l2-ctl', '-d', SUBDEV, '-C', 'exposure'],
                            capture_output=True, text=True)
    print('  Exposure range:', result.stdout.strip())


def set_exposure(exp_val):
    """Set sensor exposure (in lines)."""
    run_cmd(['v4l2-ctl', '-d', SUBDEV, '-c', 'exposure=%d' % exp_val])


def capture_one_raw(output_path):
    """Capture a single RAW frame via v4l2-ctl subprocess."""
    result = subprocess.run(
        ['v4l2-ctl', '-d', VIDEO, '--stream-mmap',
         '--stream-count', '1', '--stream-to', output_path],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        print('  Capture ERROR:', result.stderr.strip()[-200:])
        return None
    return output_path


# ── Main loop ────────────────────────────────────────────────

def collect(args):
    """Main capture loop: alternating exposure, ring buffer, save triplets."""
    output_dir = args.output
    target_count = args.count
    save_format = args.format

    setup_sensor()

    saver = TripletSaver(
        output_dir=output_dir,
        raw_width=4224, raw_height=3136,
        crop_width=1920, crop_height=1080,
        bayer_pattern='BGGR',
        save_format=save_format,
    )

    print('\n[Collect] Starting capture...')
    print('  Pattern: S(16ms) → L(82ms) → S(16ms) → ...')
    print('  Output:  %s/' % output_dir)
    if target_count:
        print('  Target:  %d triplets' % target_count)
    else:
        print('  Target:  continuous (Ctrl+C to stop)')
    print()

    frame_idx = 0
    exp_type = 'S'
    start_time = time.time()

    try:
        while True:
            # Set exposure for this frame
            if exp_type == 'S':
                set_exposure(EXPOSURE_SHORT)
            else:
                set_exposure(EXPOSURE_LONG)

            # Wait for frame to complete (frame period ~83ms)
            time.sleep(0.1)

            # Capture
            tmp_path = '/tmp/collect_frame.raw'
            result = capture_one_raw(tmp_path)

            if result is None:
                print('  Frame %d capture failed, retrying...' % frame_idx)
                continue

            # Read raw bytes
            with open(tmp_path, 'rb') as f:
                raw_bytes = f.read()

            # Feed to triplet saver
            saver.feed(raw_bytes, exp_type)
            frame_idx += 1

            # Swap exposure for next frame
            exp_type = 'L' if exp_type == 'S' else 'S'

            # Check target
            if target_count and saver.triplet_count >= target_count:
                break

    except KeyboardInterrupt:
        print('\n[Collect] Interrupted.')

    elapsed = time.time() - start_time
    print('\n[Done] %d triplets in %.1f seconds (%.1f triplets/min)' %
          (saver.triplet_count, elapsed,
           saver.triplet_count / elapsed * 60))


def preview(args):
    """Preview mode: capture and print frame stats without saving."""
    setup_sensor()

    print('\n[Preview] Capturing frames (Ctrl+C to stop)...')
    frame_idx = 0
    exp_type = 'S'

    try:
        while True:
            if exp_type == 'S':
                set_exposure(EXPOSURE_SHORT)
            else:
                set_exposure(EXPOSURE_LONG)

            time.sleep(0.1)
            tmp_path = '/tmp/preview_frame.raw'
            capture_one_raw(tmp_path)

            with open(tmp_path, 'rb') as f:
                raw_bytes = f.read()

            size_mb = len(raw_bytes) / (1024 * 1024)
            # Quick brightness check: average first 10000 bytes
            sample = raw_bytes[:10000]
            avg_byte = sum(sample) / len(sample)

            print('  Frame %3d [%s]: %6.2f MB, avg_byte=%3d' %
                  (frame_idx, exp_type, size_mb, avg_byte))

            frame_idx += 1
            exp_type = 'L' if exp_type == 'S' else 'S'

    except KeyboardInterrupt:
        print('\n[Preview] Done.')


# ── CLI ──────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NBINet Training Data Collector')
    parser.add_argument('--output', '-o', type=str, default='/tmp/nbinet_collect',
                        help='Output directory for training data')
    parser.add_argument('--count', '-n', type=int, default=0,
                        help='Number of triplets to collect (0 = continuous)')
    parser.add_argument('--format', '-f', type=str, default='raw',
                        choices=['raw', 'npy'], help='Output format')
    parser.add_argument('--preview', action='store_true',
                        help='Preview mode: show frame stats, do not save')

    args = parser.parse_args()

    if args.preview:
        preview(args)
    else:
        collect(args)
