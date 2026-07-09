"""
可视化验证: 用训练好的模型跑几张验证集样本, 保存对比图
Usage: python validate_vis.py --opt options/nbinet_imx585.yaml --checkpoint snapshot/nbinet_imx585/final.pth --gpu 0 --samples 5
"""
import os, argparse, numpy as np, cv2, torch, yaml
from easydict import EasyDict as edict

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default='options/nbinet_imx585.yaml')
parser.add_argument('--checkpoint', type=str, default='snapshot/nbinet_imx585/final.pth')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--start', type=int, default=0, help='Start index in val dataset')
parser.add_argument('--samples', type=int, default=5)
parser.add_argument('--outdir', type=str, default='vis_output')
args = parser.parse_args()

# 加载配置和模型
with open(args.opt, 'r') as f:
    opt = edict(yaml.load(f, Loader=yaml.FullLoader))
from models.utils import create_generator_val
G = create_generator_val(opt.GNet)
ckpt = torch.load(args.checkpoint, map_location='cpu')
G.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
G.cuda(args.gpu).eval()

# 加载验证集
from dataloader import dataset_imx585
val_ds = dataset_imx585.IMX585_RAW3RGB_dataset(opt.Dataset, 'val')

os.makedirs(args.outdir, exist_ok=True)

def demosaic_rggb_fast(rggb_4ch):
    """快速 RGGB 4ch [4,H,W] → RGB [H*2,W*2,3] 用于预览（非边缘感知，但速度快）"""
    R, Gr, Gb, B = rggb_4ch[0], rggb_4ch[1], rggb_4ch[2], rggb_4ch[3]
    h, w = R.shape
    rgb = np.zeros((h*2, w*2, 3), dtype=np.float32)
    rgb[0::2, 0::2, 0] = R
    rgb[0::2, 1::2, 0] = R
    rgb[1::2, 0::2, 0] = R
    rgb[1::2, 1::2, 0] = R
    rgb[0::2, 0::2, 1] = Gr
    rgb[0::2, 1::2, 1] = Gr
    rgb[1::2, 0::2, 1] = Gb
    rgb[1::2, 1::2, 1] = Gb
    rgb[0::2, 0::2, 2] = B
    rgb[0::2, 1::2, 2] = B
    rgb[1::2, 0::2, 2] = B
    rgb[1::2, 1::2, 2] = B
    return rgb

def demosaic_rggb_ea(bayer):
    """边缘感知 demosaic，用于单通道 RAW Bayer"""
    bayer_clip = np.clip(bayer, 0, 1.0)
    bayer_16 = (bayer_clip * 65535.0).astype(np.uint16)
    rgb_16 = cv2.cvtColor(bayer_16, cv2.COLOR_BayerRG2RGB_EA)
    return rgb_16.astype(np.float32) / 65535.0

def raw_rggb_to_bayer(rggb_4ch):
    """4 通道 RGGB → 单通道 Bayer"""
    R, Gr, Gb, B = rggb_4ch[0], rggb_4ch[1], rggb_4ch[2], rggb_4ch[3]
    h, w = R.shape
    bayer = np.zeros((h*2, w*2), dtype=np.float32)
    bayer[0::2, 0::2] = R
    bayer[0::2, 1::2] = Gr
    bayer[1::2, 0::2] = Gb
    bayer[1::2, 1::2] = B
    return bayer

def to_uint8(img):
    """float [0,1] 转 uint8 [0,255]"""
    return np.clip(img * 255, 0, 255).astype(np.uint8)

for k in range(args.samples):
    idx = args.start + k
    sample = val_ds[idx]
    s_cat = sample['short_img'].unsqueeze(0).cuda(args.gpu)  # [1,8,H,W]
    l_img = sample['long_img'].unsqueeze(0).cuda(args.gpu)   # [1,4,H,W]
    gt    = sample['RGBout_img'].cpu().numpy()                # [3,H*2,W*2]

    with torch.no_grad():
        out = G(s_cat, l_img).clamp(0, 1).cpu().numpy()[0]    # [3,H*2,W*2]

    # 提取 S1/L/S2 RAW 平面用于可视化
    s1_4c = s_cat[0, :4].cpu().numpy()     # 前 4 通道 = short1
    s2_4c = s_cat[0, 4:].cpu().numpy()     # 后 4 通道 = short2
    l_4c  = l_img[0].cpu().numpy()         # long 4ch

    # Demosaic RAW → RGB 用于预览
    s1_bayer = raw_rggb_to_bayer(s1_4c)
    s2_bayer = raw_rggb_to_bayer(s2_4c)
    l_bayer  = raw_rggb_to_bayer(l_4c)
    s1_rgb = demosaic_rggb_ea(s1_bayer)
    s2_rgb = demosaic_rggb_ea(s2_bayer)
    l_rgb  = demosaic_rggb_ea(l_bayer)

    gt_rgb    = gt.transpose(1, 2, 0)      # 无白平衡 — 与训练目标一致
    out_rgb   = out.transpose(1, 2, 0)      # 无白平衡 — 与训练目标一致

    # 构建对比条带：S1 | L | S2 | Output | GT
    strip_h = s1_rgb.shape[0]
    strip_w = s1_rgb.shape[1]
    # 如果需要，将所有图像缩放到相同尺寸（模型输出与 GT 尺寸一致，RAW 分辨率减半）
    # GT 和输出分辨率是 RAW 的 2 倍（已 demosaic），RAW 帧分辨率减半
    # 调整 RAW demosaic 尺寸以匹配 GT 大小以便对比
    crop_h, crop_w = gt_rgb.shape[0], gt_rgb.shape[1]
    s1_rgb_rs = cv2.resize(s1_rgb, (crop_w, crop_h))
    s2_rgb_rs = cv2.resize(s2_rgb, (crop_w, crop_h))
    l_rgb_rs  = cv2.resize(l_rgb, (crop_w, crop_h))

    strip = np.hstack([
        to_uint8(s1_rgb_rs), to_uint8(l_rgb_rs), to_uint8(s2_rgb_rs),
        to_uint8(out_rgb), to_uint8(gt_rgb)
    ])

    # 添加标签
    labels = ['Short1', 'Long', 'Short2', 'Output', 'GT']
    label_y = 20
    x = 0
    for i, label in enumerate(labels):
        cv2.putText(strip, label, (x+5, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1, cv2.LINE_AA)
        x += crop_w

    outpath = os.path.join(args.outdir, 'sample_%03d.png' % idx)
    cv2.imwrite(outpath, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print('Saved: %s' % outpath)

    # 同时打印每个样本的 PSNR
    psnr_val = 20 * np.log10(1.0 / np.sqrt(np.mean((out_rgb - gt_rgb) ** 2)))
    print('  PSNR: %.2f dB' % psnr_val)

print('\nDone! Check %s/' % args.outdir)
