"""
NBINet ONNX 导出脚本

用法:
    python export/export_to_onnx.py \
        --model_path snapshot/nbinet_distill/GNet/GNet-epoch-149.pkl \
        --output_path export/nbinet.onnx \
        --height 540 --width 960

说明:
    - RAW 输入分辨率是 sRGB 的一半（Bayer 2x2 RGGB pack）
    - 1080p (1920x1080) 对应 RAW RGGB 输入 540x960
    - 网络内部 PixelShuffle(2) 实现 demosaic 2x 上采样
    - 输出是全分辨率 RGB 1080x1920
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from easydict import EasyDict as edict

from models.network import NBINet


def export_to_onnx(model_path, output_path, height=540, width=960, opset=11):
    """
    将 NBINet 导出为 ONNX 格式

    参数:
        model_path: PyTorch checkpoint 路径
        output_path: ONNX 输出路径
        height: RAW 输入高度（1080p → 540）
        width: RAW 输入宽度（1080p → 960）
        opset: ONNX opset 版本
    """
    # 构建模型（与训练时配置一致）
    opt = edict({
        'in_channel': 4,
        'out_channel': 3,
        'ngf': 10,
        'activ': 'relu',
        'norm': 'none',
        'pad_type': 'zero',
        'final_activ': 'none',
        'res_num': 2,
        'bottleneck_res_num': 4,
    })

    model = NBINet(opt)

    # 加载权重
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print('Model params: %.2fK' % (total_params / 1000))

    # 创建 dummy 输入：short_cat [1,8,H,W]（两短帧拼接）+ long [1,4,H,W]
    dummy_short_cat = torch.randn(1, 8, height, width)
    dummy_long = torch.randn(1, 4, height, width)

    # 导出 ONNX（固定分辨率，RKNN 不支持动态 shape）
    print('Exporting to ONNX: %s' % output_path)
    print('  Input 1 (short_cat): [1, 8, %d, %d] (2x RAW RGGB concatenated)' % (height, width))
    print('  Input 2 (long_raw):  [1, 4, %d, %d] (RAW RGGB)' % (height, width))
    print('  Output:              [1, 3, %d, %d] (RGB full resolution)' % (height * 2, width * 2))

    torch.onnx.export(
        model,
        (dummy_short_cat, dummy_long),
        output_path,
        opset_version=opset,
        input_names=['short_cat', 'long_raw'],
        output_names=['output'],
    )
    print('ONNX export done: %s' % output_path)

    # 用 onnxruntime 验证
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        ort_inputs = {
            'short_cat': dummy_short_cat.numpy(),
            'long_raw': dummy_long.numpy(),
        }
        ort_out = sess.run(None, ort_inputs)[0]
        print('ONNX output shape: %s (expected [1, 3, %d, %d])' % (
            str(ort_out.shape), height * 2, width * 2))

        # PyTorch 推理
        with torch.no_grad():
            pt_out = model(dummy_short_cat, dummy_long).numpy()

        diff = np.max(np.abs(pt_out - ort_out))
        print('ONNX validation: max diff = %.6f (should be < 1e-4)' % diff)
        if diff < 1e-4:
            print('PASS: ONNX output matches PyTorch output')
        else:
            print('WARNING: ONNX output differs from PyTorch output')
    except ImportError:
        print('onnxruntime not installed, skipping validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='PyTorch checkpoint path')
    parser.add_argument('--output_path', type=str, default='export/nbinet.onnx', help='ONNX output path')
    parser.add_argument('--height', type=int, default=540, help='RAW input height (1080p→540)')
    parser.add_argument('--width', type=int, default=960, help='RAW input width (1080p→960)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_path, args.height, args.width, args.opset)
