"""
IMX585 (ZWO ASI585MC) 三帧 RAW 融合数据集

目录结构:
    dataset/
    ├── 000000_<scene>/S1.npy  (uint16, 2160×3840)
    │               /L.npy   (uint16)
    │               /S2.npy  (uint16)
    │               /GT_mid.npy  (float32, 200-frame avg)
    │               /meta.json
    └── splits/
        ├── train.txt   (每行一个 sample folder name)
        ├── val.txt
        └── test.txt
"""

import os
import numpy as np
import cv2
import random
import torch
import torch.utils.data as data


class IMX585_RAW3RGB_dataset(data.Dataset):
    """IMX585 (ZWO ASI585MC) RAW 域三帧融合数据集

    加载 3 帧 RAW（S1/L/S2）+ 1 帧 GT RAW → 转换 RGB GT。
    Bayer RGGB, 12-bit, .npy 格式。
    """

    def __init__(self, dataset_opt, tag):
        self.opt = dataset_opt
        self.tag = tag
        self.crop_size = getattr(self.opt, 'crop_size', 256)
        self.patch_per_image = getattr(self.opt, 'patch_per_image', 4)
        self.random_crop = getattr(self.opt, 'random_crop', True)
        self.max_val = getattr(self.opt, 'max_val', 4095)        # 12-bit 白电平
        self.black_level = getattr(self.opt, 'black_level', 96) # IMX585 暗电平
        self.bayer_pattern = getattr(self.opt, 'bayer_pattern', 'RGGB')

        # 数据集根目录
        self.root = dataset_opt.root_path
        self.splits_dir = os.path.join(self.root, 'splits')

        # 读取 split 文件
        split_file = os.path.join(self.splits_dir, tag + '.txt')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.file_list = [line.strip() for line in f if line.strip()]
        else:
            self.file_list = []

        if len(self.file_list) == 0:
            self.file_list = ['placeholder'] * 100
            self.use_random = True
            print('[IMX585] No split file %s, using random data for smoke test.' % split_file)
        else:
            self.use_random = False

        print('%s === IMX585 dataset samples: %d (root: %s)' % (tag, len(self.file_list), self.root))

    @staticmethod
    def imx585_bayer_to_rggb(bayer, black_level=96, max_val=4095.0):
        """RGGB Bayer → 4-plane RGGB + 归一化到 [0, 1]

        IMX585 RGGB 排列 (sdk_bayer_pattern_value=0):
            行0: R  Gr R  Gr ...
            行1: Gb B  Gb B  ...
        """
        if bayer.dtype == np.uint16:
            bayer = bayer.astype(np.float32)

        # 减黑电平, 归一化
        bayer = np.clip((bayer - black_level) / (max_val - black_level), 0.0, 1.0)

        R  = bayer[0::2, 0::2]
        Gr = bayer[0::2, 1::2]
        Gb = bayer[1::2, 0::2]
        B  = bayer[1::2, 1::2]
        rggb = np.stack([R, Gr, Gb, B], axis=0)  # [4, H/2, W/2]
        return rggb

    @staticmethod
    def demosaic_gt(bayer_raw, max_val=4095.0):
        """GT RAW Bayer → RGB, 返回 [3, H, W] float32 [0, 1]

        GT 是 200 帧平均 float32, 噪声极低, 边缘感知 demosaic 效果好。
        """
        # 裁剪并缩放到 16-bit
        bayer_clip = np.clip(bayer_raw, 0, max_val)
        bayer_16 = (bayer_clip / max_val * 65535.0).astype(np.uint16)
        # RGGB → RGB (边缘感知 demosaic)
        rgb_16 = cv2.cvtColor(bayer_16, cv2.COLOR_BayerRG2RGB_EA)
        rgb = rgb_16.astype(np.float32) / 65535.0
        rgb = rgb.transpose(2, 0, 1)  # [H, W, C] → [C, H, W]
        return rgb

    @staticmethod
    def random_crop_start(h, w, crop_size, min_divide=2):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def _load_frame(self, folder, filename):
        """加载单个 .npy RAW 帧, 返回 [4, H, W] float32 RGGB"""
        path = os.path.join(self.root, folder, filename)
        data = np.load(path)
        return self.imx585_bayer_to_rggb(data,
                                         black_level=self.black_level,
                                         max_val=self.max_val)

    def _load_gt(self, folder):
        """加载 GT (RAW domain) → demosaic → RGB [3, H, W] float32"""
        path = os.path.join(self.root, folder, 'GT_mid.npy')
        bayer = np.load(path)
        return self.demosaic_gt(bayer, max_val=self.max_val)

    def __getitem__(self, index):
        if self.use_random:
            cs = self.crop_size
            short1 = np.random.randn(4, cs, cs).astype(np.float32)
            long   = np.random.randn(4, cs, cs).astype(np.float32)
            short2 = np.random.randn(4, cs, cs).astype(np.float32)
            gt = np.random.randn(3, cs * 2, cs * 2).astype(np.float32)
        else:
            folder = self.file_list[index]

            short1 = self._load_frame(folder, 'S1.npy')
            long   = self._load_frame(folder, 'L.npy')
            short2 = self._load_frame(folder, 'S2.npy')
            gt = self._load_gt(folder)

            # 随机裁切 (Bayer 对齐)
            if self.random_crop:
                H, W = short1.shape[1], short1.shape[2]
                rh, rw = self.random_crop_start(H, W, self.crop_size, min_divide=2)
                short1 = short1[:, rh:rh + self.crop_size, rw:rw + self.crop_size]
                long   = long[:,   rh:rh + self.crop_size, rw:rw + self.crop_size]
                short2 = short2[:, rh:rh + self.crop_size, rw:rw + self.crop_size]
                gt_rh, gt_rw = rh * 2, rw * 2
                gt_crop = self.crop_size * 2
                gt = gt[:, gt_rh:gt_rh + gt_crop, gt_rw:gt_rw + gt_crop]

            # 数据增强 (水平/垂直翻转)
            if self.tag == 'train':
                if random.random() > 0.5:
                    short1 = short1[:, :, ::-1].copy()
                    long   = long[:,   :, ::-1].copy()
                    short2 = short2[:, :, ::-1].copy()
                    gt     = gt[:,     :, ::-1].copy()
                if random.random() > 0.5:
                    short1 = short1[:, ::-1, :].copy()
                    long   = long[:,   ::-1, :].copy()
                    short2 = short2[:, ::-1, :].copy()
                    gt     = gt[:,     ::-1, :].copy()

        # 拼接短帧 (short1 + short2 = 8 通道)
        short_cat = np.concatenate([short1, short2], axis=0)

        sample = {
            'short_img':   torch.from_numpy(short_cat).float(),
            'long_img':    torch.from_numpy(np.asarray(long)).float(),
            'RGBout_img':  torch.from_numpy(np.asarray(gt)).float(),
            'gt_long_img': torch.from_numpy(np.asarray(gt)).float(),
        }
        return sample

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    # 冒烟测试
    import argparse
    from easydict import EasyDict as edict

    opt = edict({
        'root_path': '/mnt/e/NightBurstImage/dataset',
        'crop_size': 256,
        'patch_per_image': 4,
        'random_crop': True,
        'max_val': 4095,
        'black_level': 96,
        'bayer_pattern': 'RGGB',
    })

    ds = IMX585_RAW3RGB_dataset(opt, 'train')
    print('Dataset len:', len(ds))

    sample = ds[0]
    print('short_img:', sample['short_img'].shape, 'dtype:', sample['short_img'].dtype)
    print('long_img:',  sample['long_img'].shape,  'dtype:', sample['long_img'].dtype)
    print('RGBout_img:', sample['RGBout_img'].shape, 'dtype:', sample['RGBout_img'].dtype)
    print('Value ranges:')
    print('  short_img: [%.4f, %.4f]' % (sample['short_img'].min().item(), sample['short_img'].max().item()))
    print('  long_img:  [%.4f, %.4f]' % (sample['long_img'].min().item(),  sample['long_img'].max().item()))
    print('  RGBout:    [%.4f, %.4f]' % (sample['RGBout_img'].min().item(), sample['RGBout_img'].max().item()))
