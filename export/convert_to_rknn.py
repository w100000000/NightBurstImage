"""
D2HNet_RK ONNX → RKNN 转换脚本

用法:
    python export/convert_to_rknn.py \
        --onnx_path export/d2hnet_rk.onnx \
        --output_path export/d2hnet_rk.rknn \
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

    参数:
        calib_dir: 校准数据目录，包含 .npy 或 .raw 文件
        max_samples: 最大样本数
        height: RAW RGGB 平面高度
        width: RAW RGGB 平面宽度
    返回:
        str: 校准数据列表文件的路径
    """
    calib_output_dir = os.path.join(os.path.dirname(calib_dir), 'calib_npy')
    os.makedirs(calib_output_dir, exist_ok=True)

    npy_files = sorted(glob.glob(os.path.join(calib_dir, '*.npy')))
    if len(npy_files) == 0:
        raw_files = sorted(glob.glob(os.path.join(calib_dir, '**/*.raw'), recursive=True))
        if len(raw_files) == 0:
            print('WARNING: No calibration data found in %s' % calib_dir)
            print('Generating random calibration data for testing...')
            raw_files = []
            for i in range(min(max_samples, 100)):
                data = np.random.rand(4, height, width).astype(np.float32)
                path = os.path.join(calib_output_dir, 'random_%04d.npy' % i)
                np.save(path, data)
                raw_files.append(path)
            npy_files = raw_files
        else:
            npy_files = raw_files[:max_samples]

    # 生成校准样本 npy 文件（双输入: short_cat + long_raw）
    dataset_lines = []
    for idx, f in enumerate(npy_files[:max_samples]):
        if f.endswith('.npy'):
            data = np.load(f)
        else:
            data = np.fromfile(f, dtype=np.float32).reshape(4, height, width)
        if data.ndim == 4:
            data = data[0]  # [4, H, W]

        short_cat = np.concatenate([data, data], axis=0)  # [8, H, W]
        long_raw = data  # [4, H, W]

        short_path = os.path.join(calib_output_dir, 'short_%04d.npy' % idx)
        long_path = os.path.join(calib_output_dir, 'long_%04d.npy' % idx)
        np.save(short_path, short_cat)
        np.save(long_path, long_raw)
        dataset_lines.append('%s %s' % (short_path, long_path))

    # 写入校准列表文件
    dataset_txt = os.path.join(calib_output_dir, 'dataset.txt')
    with open(dataset_txt, 'w') as f:
        f.write('\n'.join(dataset_lines))

    print('Collected %d calibration samples, list saved to %s' % (len(dataset_lines), dataset_txt))
    return dataset_txt


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
    parser.add_argument('--output_path', type=str, default='export/d2hnet_rk.rknn', help='RKNN output path')
    parser.add_argument('--calib_dir', type=str, default='data/raw/val', help='Calibration data directory')
    parser.add_argument('--target', type=str, default='rk3588', help='Target platform')
    parser.add_argument('--no_quantize', action='store_true', help='Skip INT8 quantization (use FP16)')
    parser.add_argument('--height', type=int, default=540, help='RAW input height')
    parser.add_argument('--width', type=int, default=960, help='RAW input width')
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
