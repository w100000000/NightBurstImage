import os
import numpy as np
import cv2
import random
import torch
import torch.utils.data as data


class RAW3RGB_dataset(data.Dataset):
    """RAW 域三帧融合数据集

    加载 3 帧 RAW（短1-长-短2）+ 1 帧 RGB GT。
    RAW 格式：V4L2 SBGGR10P（OV13855, 10-bit packed Bayer, BGGR 排列）。

    目录结构:
        data/raw/train/scene001/short1.raw
        data/raw/train/scene001/long.raw
        data/raw/train/scene001/short2.raw
        data/raw/train/scene001/gt.png
    """

    def __init__(self, dataset_opt, tag):
        self.opt = dataset_opt
        self.tag = tag
        self.crop_size = getattr(self.opt, 'crop_size', 256)
        self.patch_per_image = getattr(self.opt, 'patch_per_image', 4)
        self.random_crop = getattr(self.opt, 'random_crop', True)
        self.raw_width = getattr(self.opt, 'raw_width', 1920)
        self.raw_height = getattr(self.opt, 'raw_height', 1080)
        self.black_level = getattr(self.opt, 'black_level', 64)

        if tag == 'train':
            base_path = self.opt.train_path
        else:
            base_path = self.opt.val_path

        self.file_list = []
        if os.path.exists(base_path):
            for scene_dir in sorted(os.listdir(base_path)):
                scene_path = os.path.join(base_path, scene_dir)
                if os.path.isdir(scene_path):
                    required = ['short1.raw', 'long.raw', 'short2.raw', 'gt.png']
                    if all(os.path.exists(os.path.join(scene_path, f)) for f in required):
                        self.file_list.append(scene_path)

        if len(self.file_list) == 0:
            self.file_list = ['placeholder'] * 100
            self.use_random = True
            print('[RAW3RGB_dataset] No data found, using random data for smoke test.')
        else:
            self.use_random = False

        print('%s === RAW3RGB_dataset samples: %d' % (tag, len(self.file_list)))

    @staticmethod
    def unpack_sbggr10p(raw_bytes, width, height, black_level=64):
        """解包 V4L2 SBGGR10P 格式（OV13855 输出）

        10-bit packed: 每 5 字节存 4 个像素。
        """
        total_pixels = width * height
        num_groups = total_pixels // 4
        raw_bytes = raw_bytes[:num_groups * 5]

        data = raw_bytes.reshape(num_groups, 5)

        p0_h = data[:, 0].astype(np.uint16)
        p1_h = data[:, 1].astype(np.uint16)
        p2_h = data[:, 2].astype(np.uint16)
        p3_h = data[:, 3].astype(np.uint16)

        byte4 = data[:, 4]
        p0_l = (byte4 >> 0) & 0x03
        p1_l = (byte4 >> 2) & 0x03
        p2_l = (byte4 >> 4) & 0x03
        p3_l = (byte4 >> 6) & 0x03

        pixels = np.zeros(total_pixels, dtype=np.uint16)
        pixels[0::4] = (p0_h << 2) | p0_l
        pixels[1::4] = (p1_h << 2) | p1_l
        pixels[2::4] = (p2_h << 2) | p2_l
        pixels[3::4] = (p3_h << 2) | p3_l

        bayer = pixels.reshape(height, width).astype(np.float32)
        bayer = np.clip((bayer - black_level) / (1023.0 - black_level), 0.0, 1.0)
        return bayer

    @staticmethod
    def bayer_to_rggb(bayer):
        """从 BGGR Bayer mosaic 提取 4 平面 → RGGB 通道顺序

        OV13855 BGGR 排列:
            行0: B  Gb B  Gb ...
            行1: Gr R  Gr R  ...
        """
        R  = bayer[1::2, 1::2]
        Gr = bayer[1::2, 0::2]
        Gb = bayer[0::2, 1::2]
        B  = bayer[0::2, 0::2]
        rggb = np.stack([R, Gr, Gb, B], axis=0)
        return rggb

    def random_crop_start(self, h, w, crop_size, min_divide=2):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def _load_raw_frame(self, raw_path):
        raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
        bayer = self.unpack_sbggr10p(raw_bytes, self.raw_width, self.raw_height,
                                     black_level=self.black_level)
        rggb = self.bayer_to_rggb(bayer)
        return rggb

    def __getitem__(self, index):
        if self.use_random:
            cs = self.crop_size
            short1 = np.random.randn(4, cs, cs).astype(np.float32)
            long = np.random.randn(4, cs, cs).astype(np.float32)
            short2 = np.random.randn(4, cs, cs).astype(np.float32)
            gt = np.random.randn(3, cs * 2, cs * 2).astype(np.float32)
        else:
            scene_path = self.file_list[index]

            short1 = self._load_raw_frame(os.path.join(scene_path, 'short1.raw'))
            long = self._load_raw_frame(os.path.join(scene_path, 'long.raw'))
            short2 = self._load_raw_frame(os.path.join(scene_path, 'short2.raw'))

            gt = cv2.imread(os.path.join(scene_path, 'gt.png'))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H, W = short1.shape[1], short1.shape[2]
            gt = cv2.resize(gt, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
            gt = gt.transpose(2, 0, 1)

            if self.random_crop:
                rh, rw = self.random_crop_start(H, W, self.crop_size, min_divide=1)
                short1 = short1[:, rh:rh+self.crop_size, rw:rw+self.crop_size]
                long = long[:, rh:rh+self.crop_size, rw:rw+self.crop_size]
                short2 = short2[:, rh:rh+self.crop_size, rw:rw+self.crop_size]
                gt_rh, gt_rw = rh * 2, rw * 2
                gt_crop = self.crop_size * 2
                gt = gt[:, gt_rh:gt_rh+gt_crop, gt_rw:gt_rw+gt_crop]

            if self.tag == 'train':
                if random.random() > 0.5:
                    short1 = short1[:, :, ::-1].copy()
                    long = long[:, :, ::-1].copy()
                    short2 = short2[:, :, ::-1].copy()
                    gt = gt[:, :, ::-1].copy()
                if random.random() > 0.5:
                    short1 = short1[:, ::-1, :].copy()
                    long = long[:, ::-1, :].copy()
                    short2 = short2[:, ::-1, :].copy()
                    gt = gt[:, ::-1, :].copy()

        short_cat = np.concatenate([short1, short2], axis=0)
        sample = {
            'short_img':  torch.from_numpy(short_cat).float(),
            'long_img':   torch.from_numpy(np.asarray(long)).float(),
            'RGBout_img': torch.from_numpy(np.asarray(gt)).float(),
            'gt_long_img': torch.from_numpy(np.asarray(gt)).float(),
        }
        return sample

    def __len__(self):
        return len(self.file_list)
