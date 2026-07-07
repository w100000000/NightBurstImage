"""
NBINet ONNX → RKNN 转换脚本

用法:
    python export/convert_to_rknn.py \
        --onnx_path export/nbinet.onnx \
        --output_path export/nbinet.rknn \
        --calib_dir data/raw/val \
        --target rk3588

依赖:
    pip install rknn-toolkit2  (从 Rockchip 官方获取)

说明:
    - INT8 量化需要校准数据集（至少 100 张代表性 RAW 输入）
    - 校准数据格式: numpy .npy 文件, shape [4, H, W], float32 [0, 1]
    - 目标平台: rk3588
"""

import argparse
import os
import sys
import glob
import numpy as np


def collect_calibration_data(calib_dir, max_samples=200, height=540, width=960):
    """收集校准数据并保存为 npy 文件，生成 RKNN 所需的校准数据列表文件。

    RKNN-Toolkit2 的 build(dataset=...) 要求传入一个文本文件路径，
    文本文件每行列出一个校准样本的文件路径。对于多输入模型，每行用空格
    分隔各输入的文件路径。

    支持两种格式:
      1. IMX585 场景目录 (推荐): calib_dir 下有 scene_xxx/S1.npy + scene_xxx/L.npy
      2. 单个 .npy 文件: 对每个文件自动生成 S2=S1+noise 的模拟配对

    参数:
        calib_dir: 校准数据目录
        max_samples: 最大样本数
        height: RAW RGGB 平面高度
        width: RAW RGGB 平面宽度
    返回:
        str: 校准数据列表文件的路径
    """
    calib_output_dir = os.path.join(os.path.dirname(calib_dir), 'calib_npy')
    os.makedirs(calib_output_dir, exist_ok=True)

    # 清理旧的校准文件
    for old in glob.glob(os.path.join(calib_output_dir, '*.npy')):
        os.remove(old)

    dataset_lines = []

    # 策略1: 检测 IMX585 场景目录 (含 S1.npy + L.npy 子目录)
    scene_dirs = sorted(glob.glob(os.path.join(calib_dir, '*/')))
    scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]

    paired_samples = []
    for sd in scene_dirs:
        s1_path = os.path.join(sd, 'S1.npy')
        s2_path = os.path.join(sd, 'S2.npy')
        l_path  = os.path.join(sd, 'L.npy')
        if os.path.exists(s1_path) and os.path.exists(s2_path):
            paired_samples.append((s1_path, s2_path, l_path))

    if paired_samples:
        print('Found %d paired S1/S2/L scene directories for calibration.' % len(paired_samples))
        for idx, (s1, s2, l_path) in enumerate(paired_samples[:max_samples]):
            s1_data = np.load(s1)
            s2_data = np.load(s2)
            l_data  = np.load(l_path) if os.path.exists(l_path) else np.zeros_like(s1_data)

            # RGGB Bayer → 4-channel 并归一化到 [0,1]
            if s1_data.ndim == 2:
                s1_rggb = _bayer_to_rggb_calib(s1_data, height, width)
                s2_rggb = _bayer_to_rggb_calib(s2_data, height, width)
                l_rggb  = _bayer_to_rggb_calib(l_data, height, width)
            else:
                # 已经是 [C, H, W] 格式
                s1_rggb = s1_data.astype(np.float32)[:4, :height, :width]
                s2_rggb = s2_data.astype(np.float32)[:4, :height, :width]
                l_rggb  = l_data.astype(np.float32)[:4, :height, :width]

            short_cat = np.concatenate([s1_rggb, s2_rggb], axis=0)  # [8, H, W]

            short_path = os.path.join(calib_output_dir, 'short_%04d.npy' % idx)
            long_path  = os.path.join(calib_output_dir, 'long_%04d.npy' % idx)
            np.save(short_path, short_cat.astype(np.float32))
            np.save(long_path,  l_rggb.astype(np.float32))
            dataset_lines.append('%s %s' % (short_path, long_path))

    # 策略2: 单个 .npy 文件 (旧格式, 用加噪模拟 S1≠S2)
    if not paired_samples:
        npy_files = sorted(glob.glob(os.path.join(calib_dir, '*.npy')))
        if len(npy_files) == 0:
            raw_files = sorted(glob.glob(os.path.join(calib_dir, '**/*.raw'), recursive=True))
            npy_files = raw_files[:max_samples]

        if len(npy_files) == 0:
            print('WARNING: No calibration data found in %s' % calib_dir)
            print('Generating random calibration data for testing...')
            for i in range(min(max_samples, 100)):
                data = np.random.rand(4, height, width).astype(np.float32)
                path = os.path.join(calib_output_dir, 'random_%04d.npy' % i)
                np.save(path, data)
                npy_files.append(path)

        print('Found %d single .npy files (S2 will be S1+noise for realistic INT8 calib).' % len(npy_files[:max_samples]))
        for idx, f in enumerate(npy_files[:max_samples]):
            if f.endswith('.npy'):
                data = np.load(f)
            else:
                data = np.fromfile(f, dtype=np.float32).reshape(4, height, width)
            if data.ndim == 4:
                data = data[0]  # [4, H, W]
            data = data.astype(np.float32)

            s1_data = data[:4, :height, :width]
            # S2 = S1 + 小噪声, 模拟独立短帧 (对 INT8 量化更真实)
            noise = np.random.randn(*s1_data.shape).astype(np.float32) * 0.01
            s2_data = np.clip(s1_data + noise, 0.0, 1.0)
            l_data  = s1_data  # 长帧用同一帧近似

            short_cat = np.concatenate([s1_data, s2_data], axis=0)  # [8, H, W]

            short_path = os.path.join(calib_output_dir, 'short_%04d.npy' % idx)
            long_path  = os.path.join(calib_output_dir, 'long_%04d.npy' % idx)
            np.save(short_path, short_cat.astype(np.float32))
            np.save(long_path,  l_data.astype(np.float32))
            dataset_lines.append('%s %s' % (short_path, long_path))

    # 写入校准列表文件
    dataset_txt = os.path.join(calib_output_dir, 'dataset.txt')
    with open(dataset_txt, 'w') as f:
        f.write('\n'.join(dataset_lines))

    print('Collected %d calibration samples, list saved to %s' % (len(dataset_lines), dataset_txt))
    return dataset_txt


def _bayer_to_rggb_calib(bayer, height, width):
    """快速 RGGB Bayer → 4-plane RGGB (同 IMX585 的 Bayer 排列)"""
    bayer = bayer.astype(np.float32)
    bayer = np.clip(bayer / 4095.0, 0.0, 1.0)  # 12-bit → [0,1]
    R  = bayer[0::2, 0::2][:height, :width]
    Gr = bayer[0::2, 1::2][:height, :width]
    Gb = bayer[1::2, 0::2][:height, :width]
    B  = bayer[1::2, 1::2][:height, :width]
    return np.stack([R, Gr, Gb, B], axis=0)


def convert_onnx_to_rknn(onnx_path, output_path, calib_dir, target='rk3588',
                          do_quantize=True, height=540, width=960):
    """
    将 ONNX 模型转换为 RKNN 格式

    参数:
        onnx_path: ONNX 模型路径
        output_path: RKNN 输出路径
        calib_dir: 校准数据目录
        target: 目标平台 (rk3588/rk3576/rk3568/rk3566)
        do_quantize: 是否做 INT8 量化
        height: RAW 输入高度
        width: RAW 输入宽度
    """
    try:
        from rknn.api import RKNN
    except ImportError:
        print('ERROR: rknn-toolkit2 not installed.')
        print('Install from: https://github.com/airockchip/rknn-toolkit2')
        sys.exit(1)

    rknn = RKNN(verbose=True)

    # ========================
    # 1. 配置
    # ========================
    print('Configuring RKNN...')
    rknn.config(
        mean_values=[[0]*8, [0]*4],       # 两个输入: short_cat(8ch), long_raw(4ch)
        std_values=[[1]*8, [1]*4],        # 训练已归一化到[0,1]，RKNN不再归一化
        target_platform=target,
    )

    # ========================
    # 2. 加载 ONNX
    # ========================
    print('Loading ONNX model: %s' % onnx_path)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print('Load ONNX failed!')
        return

    # ========================
    # 3. 构建 RKNN（含量化）
    # ========================
    print('Building RKNN model...')
    dataset_txt = None
    if do_quantize:
        dataset_txt = collect_calibration_data(calib_dir, height=height, width=width)

    ret = rknn.build(
        do_quantization=do_quantize,
        dataset=dataset_txt,
    )
    if ret != 0:
        print('Build RKNN failed!')
        return

    # ========================
    # 4. 导出
    # ========================
    print('Exporting RKNN model: %s' % output_path)
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export RKNN failed!')
        return

    print('RKNN export done: %s' % output_path)

    # ========================
    # 5. 性能评估（在 x86 上模拟）
    # ========================
    print('\n--- Performance Summary ---')
    print('Target platform: %s' % target)
    print('Quantization: %s' % ('INT8' if do_quantize else 'FP16'))
    print('Input: [1, 4, %d, %d] (RAW Bayer)' % (height, width))
    print('Output: [1, 3, %d, %d] (RGB)' % (height * 2, width * 2))
    print('Expected latency on RK3588 NPU: < 40ms (for 25fps)')
    print('Note: actual latency depends on RK3588 hardware, run on device to verify')

    rknn.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', type=str, required=True, help='ONNX model path')
    parser.add_argument('--output_path', type=str, default='export/nbinet.rknn', help='RKNN output path')
    parser.add_argument('--calib_dir', type=str, default='data/raw/val', help='Calibration data directory')
    parser.add_argument('--target', type=str, default='rk3588', help='Target platform')
    parser.add_argument('--no_quantize', action='store_true', help='Skip INT8 quantization (use FP16)')
    parser.add_argument('--height', type=int, default=544, help='RAW input height (1080p→544, must be mult of 8)')
    parser.add_argument('--width', type=int, default=960, help='RAW input width (1080p→960)')
    args = parser.parse_args()

    convert_onnx_to_rknn(
        onnx_path=args.onnx_path,
        output_path=args.output_path,
        calib_dir=args.calib_dir,
        target=args.target,
        do_quantize=not args.no_quantize,
        height=args.height,
        width=args.width,
    )
