"""
D2HNet_RK ONNX 导出脚本

用法:
    python export/export_to_onnx.py \
        --model_path snapshot/d2hnet_rk_distill/GNet/GNet-epoch-149.pkl \
        --output_path export/d2hnet_rk.onnx \
        --height 540 --width 960

说明:
    - RAW 输入分辨率是 sRGB 的一半（Bayer 2x2 块）
    - 1080p (1920x1080) 对应 RAW 输入 960x540
    - 输出是全分辨率 RGB 1920x1080
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from easydict import EasyDict as edict

from models.network import D2HNet_RK


def export_to_onnx(model_path, output_path, height=540, width=960, opset=11):
    """
    将 D2HNet_RK 导出为 ONNX 格式

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
        'activ': 'lrelu',
        'norm': 'none',
        'pad_type': 'zero',
        'final_activ': 'none',
        'res_num': 2,
        'bottleneck_res_num': 4,
    })

    model = D2HNet_RK(opt)

    # 加载权重
    state_dict = torch.load(model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print('Model params: %.2fK' % (total_params / 1000))

    # 创建 dummy 输入（3 帧 RAW Bayer，半分辨率）
    dummy_short1 = torch.randn(1, 4, height, width)
    dummy_long = torch.randn(1, 4, height, width)
    dummy_short2 = torch.randn(1, 4, height, width)

    # 导出 ONNX
    print('Exporting to ONNX: %s' % output_path)
    print('  Input shape: [1, 4, %d, %d] (RAW Bayer, half-res)' % (height, width))
    print('  Output shape: [1, 3, %d, %d] (RGB, full-res)' % (height * 2, width * 2))

    torch.onnx.export(
        model,
        (dummy_short1, dummy_long, dummy_short2),
        output_path,
        opset_version=opset,
        input_names=['short1_raw', 'long_raw', 'short2_raw'],
        output_names=['output'],
        dynamic_axes={
            'short1_raw': {0: 'batch'},
            'long_raw': {0: 'batch'},
            'short2_raw': {0: 'batch'},
            'output': {0: 'batch'},
        },
    )
    print('ONNX export done: %s' % output_path)

    # 用 onnxruntime 验证
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        ort_inputs = {
            'short1_raw': dummy_short1.numpy(),
            'long_raw': dummy_long.numpy(),
            'short2_raw': dummy_short2.numpy(),
        }
        ort_out = sess.run(None, ort_inputs)[0]

        # PyTorch 推理
        with torch.no_grad():
            pt_out = model(dummy_short1, dummy_long, dummy_short2).numpy()

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
    parser.add_argument('--output_path', type=str, default='export/d2hnet_rk.onnx', help='ONNX output path')
    parser.add_argument('--height', type=int, default=540, help='RAW input height (1080p→540)')
    parser.add_argument('--width', type=int, default=960, help='RAW input width (1080p→960)')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()

    export_to_onnx(args.model_path, args.output_path, args.height, args.width, args.opset)
