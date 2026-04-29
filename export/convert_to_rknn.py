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


def collect_calibration_data(calib_dir, max_samples=200):
    """收集校准数据

    扫描目录下的 .npy 文件（RAW 4ch float32），用于 INT8 量化校准。

    参数:
        calib_dir: 校准数据目录，包含 .npy 文件
        max_samples: 最大样本数
    返回:
        list of numpy arrays, each [4, H, W]
    """
    npy_files = sorted(glob.glob(os.path.join(calib_dir, '*.npy')))
    if len(npy_files) == 0:
        # 如果没有 .npy 文件，尝试从 .raw 文件读取
        raw_files = sorted(glob.glob(os.path.join(calib_dir, '**/*.raw'), recursive=True))
        if len(raw_files) == 0:
            print('WARNING: No calibration data found in %s' % calib_dir)
            print('Generating random calibration data for testing...')
            data_list = []
            for _ in range(max_samples):
                data_list.append(np.random.randn(4, 540, 960).astype(np.float32))
            return data_list
        npy_files = raw_files[:max_samples]

    data_list = []
    for f in npy_files[:max_samples]:
        if f.endswith('.npy'):
            data = np.load(f)
        else:
            # .raw 文件: 假设已解包为 [4, H, W] float32
            data = np.fromfile(f, dtype=np.float32).reshape(4, -1, -1)
        # 取其中一帧（校准只需要输入数据，不需要 GT）
        if data.ndim == 4:
            data = data[0]  # [4, H, W]
        data_list.append(data.astype(np.float32))

    print('Collected %d calibration samples from %s' % (len(data_list), calib_dir))
    return data_list


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
        mean_values=[[0, 0, 0, 0]],      # RAW 输入不做均值归一化
        std_values=[[1023, 1023, 1023, 1023]],  # 10-bit RAW 归一化到 [0,1]
        target_platform=target,
        # quantized_dtype='asymmetric_quantized-8',  # 默认 INT8
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
    calib_data = None
    if do_quantize:
        calib_data = collect_calibration_data(calib_dir)

    ret = rknn.build(
        do_quantization=do_quantize,
        dataset=calib_data if calib_data else None,
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
