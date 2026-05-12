import os
import numpy as np
import cv2
import random
import torch
import torch.utils.data as data
import torch.nn.functional as F
from copy import deepcopy
import sys

from util import utils
from . import augmentation as DA
from .utils import resize_img, read_img


def build_train_file_set(namelist, gt_postfix = '_short'):
    shortlist = []
    long8list = []
    long6list = []
    long4list = []
    long2list = []
    gtlist = []
    for i in range(len(namelist)):
        if utils.check_file(namelist[i] + '_short.png') and \
            utils.check_file(namelist[i] + '_long8.png') and \
                utils.check_file(namelist[i] + '_long6.png') and \
                    utils.check_file(namelist[i] + '_long4.png') and \
                        utils.check_file(namelist[i] + '_long2.png') and \
                            utils.check_file(namelist[i] + gt_postfix + '.png'):
            shortlist.append(namelist[i] + '_short.png')
            long8list.append(namelist[i] + '_long8.png')
            long6list.append(namelist[i] + '_long6.png')
            long4list.append(namelist[i] + '_long4.png')
            long2list.append(namelist[i] + '_long2.png')
            gtlist.append(namelist[i] + gt_postfix + '.png')
    return shortlist, long8list, long6list, long4list, long2list, gtlist


def build_file_set(namelist, gt_postfix = '_short'):
    shortlist = []
    longlist = []
    gtlist = []
    for i in range(len(namelist)):
        if utils.check_file(namelist[i] + '_short.png') and \
            utils.check_file(namelist[i] + '_long8.png') and \
                utils.check_file(namelist[i] + gt_postfix + '.png'):
            shortlist.append(namelist[i] + '_short.png')
            longlist.append(namelist[i] + '_long8.png')
            gtlist.append(namelist[i] + gt_postfix + '.png')
    return shortlist, longlist, gtlist


def add_train_sharp_file(shortlist, long8list, long6list, long4list, long2list, gtlist, src_dir, sharp_dir):
    short, long8, long6, long4, long2, gt = [], [], [], [], [], []
    for i in range(len(gtlist)):
        sharp_gt = gtlist[i].replace(src_dir, sharp_dir)
        if utils.check_file(sharp_gt):
            gt.append(sharp_gt)
            short.append(shortlist[i])
            long8.append(long8list[i])
            long6.append(long6list[i])
            long4.append(long4list[i])
            long2.append(long2list[i])
    short += shortlist
    long8 += long8list
    long6 += long6list
    long4 += long4list
    long2 += long2list
    gt += gtlist
    return short, long8, long6, long4, long2, gt


def add_sharp_file(shortlist, longlist, gtlist, src_dir, sharp_dir):
    short, long_, gt = [], [], []
    for i in range(len(gtlist)):
        sharp_gt = gtlist[i].replace(src_dir, sharp_dir)
        if utils.check_file(sharp_gt):
            gt.append(sharp_gt)
            short.append(shortlist[i])
            long_.append(longlist[i])
    short += shortlist
    long_ += longlist
    gt += gtlist
    return short, long_, gt


class SLRGB2RGB_dataset(data.Dataset):
    def __init__(self, dataset_opt, tag):
        # General
        self.opt = dataset_opt
        self.tag = tag
        if hasattr(self.opt, 'gt_postfix'):
            gt_postfix = self.opt.gt_postfix
        else:
            gt_postfix = '_short'

        if tag == 'train':
            namelist = utils.get_jpgs_once(self.opt.train_path)
            shortlist, long8list, long6list, long4list, long2list, gtlist = build_train_file_set(namelist, gt_postfix = gt_postfix)
            
            # Add the sharpened data if it exists
            if hasattr(self.opt, 'train_sharp_path'):
                shortlist, long8list, long6list, long4list, long2list, gtlist = \
                    add_train_sharp_file(shortlist, long8list, long6list, long4list, long2list, gtlist, self.opt.train_path, self.opt.train_sharp_path)
                
            self.in_short_filelist = shortlist
            self.in_long8_filelist = long8list
            self.in_long6_filelist = long6list
            self.in_long4_filelist = long4list
            self.in_long2_filelist = long2list
            self.out_srgb_filelist = gtlist
        elif tag == 'val':
            namelist = utils.get_jpgs_once(self.opt.val_path)
            shortlist, longlist, gtlist = build_file_set(namelist, gt_postfix = gt_postfix)
            
            # Add the sharpened data if it exists
            if hasattr(self.opt, 'val_sharp_path'):
                shortlist, longlist, gtlist = \
                    add_sharp_file(shortlist, longlist, gtlist, self.opt.val_path, self.opt.val_sharp_path)
                
            self.in_short_filelist = shortlist
            self.in_long_filelist = longlist
            self.out_srgb_filelist = gtlist

        # Get the blur patch
        if hasattr(self.opt, 'blur_path'):
            if tag == 'train':
                namelist = utils.get_blur_file_once(self.opt.blur_path.train_path)
                shortlist, long8list, long6list, long4list, long2list, gtlist = build_train_file_set(namelist, gt_postfix = gt_postfix)
                
                if hasattr(self.opt.blur_path, 'train_sharp_path'):
                    shortlist, long8list, long6list, long4list, long2list, gtlist = \
                        add_train_sharp_file(shortlist, long8list, long6list, long4list, long2list, gtlist, self.opt.blur_path.train_path, self.opt.blur_path.train_sharp_path)
                    
                self.blur_short_filelist = shortlist
                self.blur_long8_filelist = long8list
                self.blur_long6_filelist = long6list
                self.blur_long4_filelist = long4list
                self.blur_long2_filelist = long2list
                self.blur_out_filelist = gtlist

            elif tag == 'val':
                namelist = utils.get_blur_file_once(self.opt.blur_path.val_path)
                shortlist, longlist, gtlist = build_file_set(namelist, gt_postfix = gt_postfix)
                
                if hasattr(self.opt.blur_path, 'val_sharp_path'):
                    shortlist, longlist, gtlist = \
                        add_sharp_file(shortlist, longlist, gtlist, self.opt.blur_path.train_path, self.opt.blur_path.train_sharp_path)
                    
                self.blur_short_filelist = shortlist
                self.blur_long_filelist = longlist
                self.blur_out_filelist = gtlist
            
            print('%s === Blur files: %d' % (tag, len(self.blur_short_filelist)))
        
    # Generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def crop_patch(self, in_short_img, in_long_img, out_srgb_img, patch_per_image = 1, crop_size = 312):
        in_short_imgs = []
        in_long_imgs = []
        out_srgb_imgs = []
        gt_long_imgs = []
        h, w = in_short_img.shape[:2]

        if crop_size is None or min(h, w) <= crop_size:
            in_short_imgs.append(in_short_img)
            in_long_imgs.append(in_long_img)
            out_srgb_imgs.append(out_srgb_img)
            gt_long_imgs.append(deepcopy(in_long_img))
        else:
            for i in range(patch_per_image):
                h, w = in_short_img.shape[:2]
                rand_h, rand_w = self.random_crop_start(h, w, crop_size, 2)
                in_short_patch = in_short_img[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size, :]
                in_long_patch = in_long_img[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size, :]
                out_srgb_patch = out_srgb_img[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size, :]
                gt_long_patch = in_long_img[rand_h:rand_h+crop_size, rand_w:rand_w+crop_size, :]

                if self.opt.shot_noise:
                    in_short_patch = DA.add_shot_noise(in_short_patch, iso = self.opt.shot_short_iso, noise = self.opt.shot_noise_mode)
                    in_long_patch = DA.add_shot_noise(in_long_patch, iso = self.opt.shot_long_iso, noise = self.opt.shot_noise_mode)
                
                elif self.opt.noise_aug:
                    a = random.randint(6, 8)
                    noise_short = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6 * a
                    noise_long = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6
                    in_short_patch = in_short_patch + noise_short
                    in_long_patch = in_long_patch + noise_long
                    in_short_patch = np.clip(in_short_patch, 0, 1)
                    in_long_patch = np.clip(in_long_patch, 0, 1)

                in_short_imgs.append(in_short_patch)
                in_long_imgs.append(in_long_patch)
                out_srgb_imgs.append(out_srgb_patch)
                gt_long_imgs.append(gt_long_patch)
        in_short_img = np.array(in_short_imgs)
        in_long_img = np.array(in_long_imgs)
        out_srgb_img = np.array(out_srgb_imgs)
        gt_long_img = np.array(gt_long_imgs)

        return in_short_img, in_long_img, out_srgb_img, gt_long_img

    def get_blur_patch(self, crop_size = 312):
        in_short_blur_patches = []
        in_long_blur_patches = []
        out_blur_patches = []
        gt_long_blur_patches = []
        random_index = np.random.choice(len(self.blur_short_filelist), self.opt.blur_path.patch_num)
        for random_i in random_index:
            in_short_blurpatch_file = self.blur_short_filelist[random_i]
            if self.tag == 'train':
                l = random.randint(1, 4)
                if l == 1:
                    in_long_blurpatch_file = self.blur_long4_filelist[random_i]
                elif l == 2:
                    in_long_blurpatch_file = self.blur_long6_filelist[random_i]
                elif l == 3:
                    in_long_blurpatch_file = self.blur_long8_filelist[random_i]
                elif l == 4:
                    in_long_blurpatch_file = self.blur_long2_filelist[random_i]
            else:
                in_long_blurpatch_file = self.blur_long_filelist[random_i]
            out_blur_patch_file = self.blur_out_filelist[random_i]

            in_short_blurpatch = cv2.imread(in_short_blurpatch_file)
            in_long_blurpatch = cv2.imread(in_long_blurpatch_file)
            out_blur_patch = cv2.imread(out_blur_patch_file)

            in_short_blurpatch = cv2.cvtColor(in_short_blurpatch, cv2.COLOR_BGR2RGB)
            in_long_blurpatch = cv2.cvtColor(in_long_blurpatch, cv2.COLOR_BGR2RGB)
            out_blur_patch = cv2.cvtColor(out_blur_patch, cv2.COLOR_BGR2RGB)

            in_short_blurpatch = in_short_blurpatch.astype(np.float) / 255.0
            in_long_blurpatch = in_long_blurpatch.astype(np.float) / 255.0
            out_blur_patch = out_blur_patch.astype(np.float) / 255.0

            in_short_blurpatch, in_long_blurpatch, out_blur_patch, gt_long_blurpatch = \
                self.crop_patch(in_short_blurpatch, in_long_blurpatch, out_blur_patch, patch_per_image = 1, crop_size = crop_size)

            in_short_blur_patches.append(in_short_blurpatch[0])
            in_long_blur_patches.append(in_long_blurpatch[0])
            out_blur_patches.append(out_blur_patch[0])
            gt_long_blur_patches.append(gt_long_blurpatch[0])

        in_short_blur_patches = np.array(in_short_blur_patches)
        in_long_blur_patches = np.array(in_long_blur_patches)
        out_blur_patches = np.array(out_blur_patches)
        gt_long_blur_patches = np.array(gt_long_blur_patches)

        return in_short_blur_patches, in_long_blur_patches, out_blur_patches, gt_long_blur_patches

    def __getitem__(self, index):

        # Read images
        in_short_path = self.in_short_filelist[index]
        if self.tag == 'train':
            l = random.randint(1, 4)
            if l == 1:
                in_long_path = self.in_long4_filelist[index]
            elif l == 2:
                in_long_path = self.in_long6_filelist[index]
            elif l == 3:
                in_long_path = self.in_long8_filelist[index]
            elif l == 4:
                in_long_path = self.in_long2_filelist[index]
        else:
            in_long_path = self.in_long_filelist[index]
        out_srgb_path = self.out_srgb_filelist[index]

        # if patch_per_image = 0, only using blur patch for training
        if (self.opt.random_crop and hasattr(self.opt, 'patch_per_image') and self.opt.patch_per_image == 0):
            in_short_img, in_long_img, out_srgb_img = None, None, None
        else:
            in_short_img = cv2.imread(in_short_path)
            in_long_img = cv2.imread(in_long_path)
            out_srgb_img = cv2.imread(out_srgb_path)

            try:
                in_short_img = cv2.cvtColor(in_short_img, cv2.COLOR_BGR2RGB)
                in_long_img = cv2.cvtColor(in_long_img, cv2.COLOR_BGR2RGB)
                out_srgb_img = cv2.cvtColor(out_srgb_img, cv2.COLOR_BGR2RGB)
            except Exception:
                print('file %s not found.' % in_short_path)
                return None
            
            # Normalization and add noise
            in_short_img = in_short_img.astype(np.float) / 255.0
            in_long_img = in_long_img.astype(np.float) / 255.0
            out_srgb_img = out_srgb_img.astype(np.float) / 255.0

            if hasattr(self.opt, 'color_adjust') and self.opt.color_adjust:
                in_short_img = DA.color_adjustment(in_short_img, self.opt.color_adjust.prob)

            if hasattr(self.opt, 'illum_adjust') and self.tag == "train":
                if 'day' in in_short_path:
                    darken_images = DA.illum_adjustment(np.array([in_short_img, in_long_img, out_srgb_img]), dark_prob = self.opt.illum_adjust.prob)
                    in_short_img = darken_images[0]
                    in_long_img = darken_images[1]
                    out_srgb_img = darken_images[2]

            if self.opt.random_crop:
                if hasattr(self.opt, 'patch_per_image'):
                    patch_per_image = self.opt.patch_per_image
                else:
                    patch_per_image = 1

                if hasattr(self.opt, 'crop_size'):
                    crop_size = self.opt.crop_size
                else:
                    crop_size = 312

                in_short_img, in_long_img, out_srgb_img, gt_long_img = self.crop_patch(in_short_img, in_long_img, out_srgb_img, patch_per_image, crop_size)
            else:
                gt_long_img = deepcopy(in_long_img)
                if self.opt.shot_noise:
                    in_short_img = DA.add_shot_noise(in_short_img, iso = self.opt.shot_short_iso, noise = self.opt.shot_noise_mode)
                    in_long_img = DA.add_shot_noise(in_long_img, iso = self.opt.shot_long_iso, noise = self.opt.shot_noise_mode)
                elif self.opt.noise_aug:
                    a = random.randint(6, 8)
                    noise_short = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6 * a
                    noise_long = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6
                    in_short_img = in_short_img + noise_short
                    in_long_img = in_long_img + noise_long
                    in_short_img = np.clip(in_short_img, 0, 1)
                    in_long_img = np.clip(in_long_img, 0, 1)

            if hasattr(self.opt, 'cutnoise') and self.tag == "train":
                in_short_img = DA.cutnoise(in_short_img, out_srgb_img, self.opt.cutnoise.size, self.opt.cutnoise.prob)
                if hasattr(self.opt.cutnoise, 'long') and self.opt.cutnoise.long:
                    in_long_img = DA.cutnoise(in_long_img, out_srgb_img, self.opt.cutnoise.size, self.opt.cutnoise.prob)

        # Get blur patches. It is only used when random_crop is True
        if hasattr(self.opt, 'blur_path') and self.opt.random_crop:
            # NHWC
            in_short_blur_patches, in_long_blur_patches, out_blur_patches, gt_long_blur_patches = self.get_blur_patch(crop_size = crop_size)

            # Combine blur patches with cropped patches from the original full-resolution images
            if out_srgb_img is None:
                out_srgb_img = out_blur_patches
                in_short_img = in_short_blur_patches
                in_long_img = in_long_blur_patches
                gt_long_img = gt_long_blur_patches
            else:
                if len(out_srgb_img.shape) == 3:
                    out_srgb_img = np.concatenate((out_srgb_img[np.newaxis, :, :, :], out_blur_patches), axis = 0)
                    in_short_img = np.concatenate((in_short_img[np.newaxis, :, :, :], in_short_blur_patches), axis = 0)
                    in_long_img = np.concatenate((in_long_img[np.newaxis, :, :, :], in_long_blur_patches), axis = 0)
                    gt_long_img = np.concatenate((gt_long_img[np.newaxis, :, :, :], gt_long_blur_patches), axis = 0)
                elif len(out_srgb_img.shape) == 4:
                    out_srgb_img = np.concatenate((out_srgb_img, out_blur_patches), axis = 0)
                    in_short_img = np.concatenate((in_short_img, in_short_blur_patches), axis = 0)
                    in_long_img = np.concatenate((in_long_img, in_long_blur_patches), axis = 0)
                    gt_long_img = np.concatenate((gt_long_img, gt_long_blur_patches), axis = 0)

        # to tensor
        if len(in_short_img.shape) == 3:
            out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(2, 0, 1).contiguous()
            in_short_img = torch.from_numpy(in_short_img).float().permute(2, 0, 1).contiguous()
            in_long_img = torch.from_numpy(in_long_img).float().permute(2, 0, 1).contiguous()
            gt_long_img = torch.from_numpy(gt_long_img).float().permute(2, 0, 1).contiguous()
        elif len(in_short_img.shape) == 4:
            out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(0, 3, 1, 2).contiguous()
            in_short_img = torch.from_numpy(in_short_img).float().permute(0, 3, 1, 2).contiguous()
            in_long_img = torch.from_numpy(in_long_img).float().permute(0, 3, 1, 2).contiguous()
            gt_long_img = torch.from_numpy(gt_long_img).float().permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError('undesired shape in sample.')

        sample = {'short_img': in_short_img,
                  'long_img': in_long_img,
                  'RGBout_img': out_srgb_img,
                  'gt_long_img': gt_long_img,
                  'in_short_path': in_short_path}

        return sample

    def __len__(self):
        return len(self.in_short_filelist)


class SLRGB2RGB_valdataset(data.Dataset):

    def __init__(self, dataset_opt, tag):
        # General
        self.opt = dataset_opt
        if hasattr(self.opt, 'gt_postfix'):
            gt_postfix = self.opt.gt_postfix
        else:
            gt_postfix = '_short'
        if tag == 'train':
            namelist = utils.get_jpgs_once(self.opt.train_path)
            shortlist, longlist, gtlist = build_file_set(namelist, gt_postfix = gt_postfix)
            self.in_short_filelist = shortlist
            self.in_long_filelist = longlist
            self.out_srgb_filelist = gtlist
        if tag == 'val':
            namelist = utils.get_jpgs_once(self.opt.val_path)
            shortlist, longlist, gtlist = build_file_set(namelist, gt_postfix = gt_postfix)
            self.in_short_filelist = shortlist
            self.in_long_filelist = longlist
            self.out_srgb_filelist = gtlist
        
    # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w
    
    def __getitem__(self, index):

        # Read images
        in_short_path = self.in_short_filelist[index]
        in_long_path = self.in_long_filelist[index]
        out_srgb_path = self.out_srgb_filelist[index]
        
        in_short_img = cv2.imread(in_short_path)
        in_long_img = cv2.imread(in_long_path)
        out_srgb_img = cv2.imread(out_srgb_path)
        
        # Normalization and add noise
        in_short_img = in_short_img.astype(np.float) / 255.0
        in_long_img = in_long_img.astype(np.float) / 255.0
        out_srgb_img = out_srgb_img.astype(np.float) / 255.0

        if self.opt.shot_noise:
            in_short_img = DA.add_shot_noise(in_short_img, iso = self.opt.shot_short_iso, noise = self.opt.shot_noise_mode)
            in_long_img = DA.add_shot_noise(in_long_img, iso = self.opt.shot_long_iso, noise = self.opt.shot_noise_mode)
        elif self.opt.noise_aug:
            a = random.randint(6, 8)
            noise_short = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6 * a
            noise_long = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6
            in_short_img = in_short_img + noise_short
            in_long_img = in_long_img + noise_long
            in_short_img = np.clip(in_short_img, 0, 1)
            in_long_img = np.clip(in_long_img, 0, 1)

        # to tensor
        out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(2, 0, 1).contiguous()

        in_short_img = torch.from_numpy(in_short_img).float().permute(2, 0, 1).contiguous()
        in_long_img = torch.from_numpy(in_long_img).float().permute(2, 0, 1).contiguous()

        sample = {'short_img': in_short_img,
                  'long_img': in_long_img,
                  'RGBout_img': out_srgb_img,
                  'in_short_path': in_short_path}

        return sample

    def __len__(self):
        return len(self.in_short_filelist)


def get_dataset_pngs(path):
    # read a folder, return the image name
    ret_long = []
    ret_short = []
    i = 0
    tmp_files = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if 'png' == filespath[-3:]:
                tmp_files.append(os.path.join(root, filespath))
                if len(tmp_files) == 2:
                    tmp_files = sorted(tmp_files)
                    ret_long.append(tmp_files[0])
                    ret_short.append(tmp_files[1])
                    tmp_files = []
    return ret_long, ret_short


def quadra_list(imglist):
    outlist = []
    for i in range(len(imglist)):
        outlist.append(imglist[i])
        outlist.append(imglist[i])
        outlist.append(imglist[i])
        outlist.append(imglist[i])
    return outlist


def read_iso(file):
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith('iso='):
                line = line.strip()
                return float(line.split('iso=')[1])


class SLRGB2RGB_valdataset_singleimage(data.Dataset):
    def __init__(self, src_path):
        # General
        # self.opt = opt
        longlist, shortlist = get_dataset_pngs(src_path)
        self.longlist = longlist
        self.shortlist = shortlist
        
    # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w
    
    def __getitem__(self, index):

        # Read images
        in_short_path = self.shortlist[index]
        in_long_path = self.longlist[index]

        in_short_img = cv2.imread(in_short_path)
        in_long_img = cv2.imread(in_long_path)

        in_short_img = cv2.cvtColor(in_short_img, cv2.COLOR_BGR2RGB)
        in_long_img = cv2.cvtColor(in_long_img, cv2.COLOR_BGR2RGB)
        
        # Normalization
        in_short_img = in_short_img.astype(np.float) / 255.0
        in_long_img = in_long_img.astype(np.float) / 255.0

        # to tensor
        in_short_img = torch.from_numpy(in_short_img).float().permute(2, 0, 1).contiguous()
        in_long_img = torch.from_numpy(in_long_img).float().permute(2, 0, 1).contiguous()

        short_iso = read_iso(in_short_path.replace('png', 'txt'))
        long_iso = read_iso(in_long_path.replace('png', 'txt'))

        sample = {'in_short_img': in_short_img,
                  'in_long_img': in_long_img,
                  'short_iso': short_iso,
                  'long_iso': long_iso}

        return sample

    def __len__(self):
        return len(self.longlist)


class TP_dataset(SLRGB2RGB_dataset):

    def __init__(self, dataset_opt, tag):
        super(TP_dataset, self).__init__(dataset_opt, tag)

    def __len__(self):
        return len(self.in_short_filelist)

    def crop_tensor_patch(self, tensors):
        """
        crop the output of DeblurNet
        tensors: list or Tensor
        """
        if hasattr(self.opt, 'denoise_patch_per_image'):
            patch_per_image = self.opt.denoise_patch_per_image
        else:
            patch_per_image = 1

        if isinstance(tensors, list):
            ret_tensors = [[] for _ in range(len(tensors))]
            N, _, H, W = tensors[0].shape
            for _ in range(patch_per_image):
                ranh_h, rand_w = self.random_crop_start(H, W, self.opt.denoise_crop_size, 2)
                for j, tensor in enumerate(tensors):
                    crop_tensor = tensor[:, :, ranh_h:ranh_h+self.opt.denoise_crop_size, rand_w:rand_w+self.opt.denoise_crop_size]
                    ret_tensors[j].append(crop_tensor)
            
            for i in range(len(ret_tensors)):
                ret_tensors[i] = torch.cat(ret_tensors[i], dim = 0)
            
        else:
            ret_tensors = []
            _, _, h, w = tensors[0].shape
            for _ in range(patch_per_image):
                ranh_h, rand_w = self.random_crop_start(h, w, self.opt.denoise_crop_size, 2)
                crop_tensor = tensors[:, :, ranh_h:ranh_h+self.opt.denoise_crop_size, rand_w:rand_w+self.opt.denoise_crop_size]
                ret_tensors.append(crop_tensor)
            ret_tensors = torch.cat(ret_tensors, dim = 0)
        
        return tuple(ret_tensors)

    def augment_tensor_patch(self, d_images, gt_images):
        """
        apply the augmentation to the output of DeblurNet
        """
        if self.tag == "train":
            if hasattr(self.opt, 'cutnoise'):
                return self.cutnoise_tensor_patch(d_images, gt_images)
            elif hasattr(self.opt, 'cutout'):
                return DA.cutout_tensor(d_images, self.opt.cutout.size, self.opt.cutout.prob)
            else:
                return d_images
        
        return d_images

    def cutnoise_tensor_patch(self, d_images, gt_images):
        """
        apply the CutNoise to the output of DeblurNet
        """
        if self.tag == "train":
            return DA.cutnoise_tensor(d_images, gt_images, self.opt.cutnoise.size, self.opt.cutnoise.prob)
        else:
            return d_images

    def __getitem__(self, index):
        # Read images
        in_short_path = self.in_short_filelist[index]
        if self.tag == 'train':
            l = random.randint(1, 4)
            if l == 1:
                in_long_path = self.in_long4_filelist[index]
            elif l == 2:
                in_long_path = self.in_long6_filelist[index]
            elif l == 3:
                in_long_path = self.in_long8_filelist[index]
            elif l == 4:
                in_long_path = self.in_long2_filelist[index]
        else:
            in_long_path = self.in_long_filelist[index]
        out_srgb_path = self.out_srgb_filelist[index]

        in_short_img = read_img(self.opt, in_short_path)
        in_long_img = read_img(self.opt, in_long_path)
        out_srgb_img = read_img(self.opt, out_srgb_path)

        # Normalization and add noise
        in_short_img = in_short_img.astype(np.float) / 255.0
        in_long_img = in_long_img.astype(np.float) / 255.0
        out_srgb_img = out_srgb_img.astype(np.float) / 255.0

        if hasattr(self.opt, 'color_adjust') and self.opt.color_adjust:
            in_short_img = DA.color_adjustment(in_short_img, self.opt.color_adjust.prob)
        
        if self.opt.random_crop:
            if hasattr(self.opt, 'deblur_patch_per_image'):
                patch_per_image = self.opt.deblur_patch_per_image
            else:
                patch_per_image = 1

            if hasattr(self.opt, 'deblur_crop_size'):
                deblur_crop_size = self.opt.deblur_crop_size
            else:
                deblur_crop_size = 1024

            in_short_img, in_long_img, out_srgb_img, gt_long_img = self.crop_patch(in_short_img, in_long_img, out_srgb_img, patch_per_image, deblur_crop_size)
            
            # Get blur patch
            if hasattr(self.opt, 'blur_path'):
                assert deblur_crop_size <= 1024
                # NHWC
                in_short_blur_patches, in_long_blur_patches, out_blur_patches, gt_long_blur_patches = self.get_blur_patch(crop_size = deblur_crop_size)
                in_short_img = np.concatenate((in_short_img, in_short_blur_patches), axis = 0)
                in_long_img = np.concatenate((in_long_img, in_long_blur_patches), axis = 0)
                out_srgb_img = np.concatenate((out_srgb_img, out_blur_patches), axis = 0)
                gt_long_img = np.concatenate((gt_long_img, gt_long_blur_patches), axis = 0)

        else:
            gt_long_img = deepcopy(in_long_img)
            if self.opt.shot_noise:
                in_short_img = DA.add_shot_noise(in_short_img, iso = self.opt.shot_short_iso, noise = self.opt.shot_noise_mode)
                in_long_img = DA.add_shot_noise(in_long_img, iso = self.opt.shot_long_iso, noise = self.opt.shot_noise_mode)
            elif self.opt.noise_aug:
                a = random.randint(6, 8)
                noise_short = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6 * a
                noise_long = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6
                in_short_img = in_short_img + noise_short
                in_long_img = in_long_img + noise_long
                in_short_img = np.clip(in_short_img, 0, 1)
                in_long_img = np.clip(in_long_img, 0, 1)

        # downsample crop image for deblur
        deblur_size = self.opt.deblur_size
        down_short_img, down_long_img, down_out_img, down_gtlong_img = resize_img([in_short_img, in_long_img, out_srgb_img, gt_long_img], size = deblur_size)

        if hasattr(self.opt, 'cutnoise') and self.tag == "train":
            in_short_img = DA.cutnoise(in_short_img, out_srgb_img, self.opt.cutnoise.size, self.opt.cutnoise.prob)
            # if hasattr(self.opt.cutnoise, 'long') and self.opt.cutnoise.long:
            #     in_long_img = DA.cutnoise(in_long_img, out_srgb_img, self.opt.cutnoise.size, self.opt.cutnoise.prob)

        # to tensor
        if len(in_short_img.shape) == 3:
            out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(2, 0, 1).contiguous()
            in_short_img = torch.from_numpy(in_short_img).float().permute(2, 0, 1).contiguous()
            in_long_img = torch.from_numpy(in_long_img).float().permute(2, 0, 1).contiguous()
            gt_long_img = torch.from_numpy(gt_long_img).float().permute(2, 0, 1).contiguous()
            down_short_img = torch.from_numpy(down_short_img).float().permute(2, 0, 1).contiguous()
            down_long_img = torch.from_numpy(down_long_img).float().permute(2, 0, 1).contiguous()
            down_out_img = torch.from_numpy(down_out_img).float().permute(2, 0, 1).contiguous()
            down_gtlong_img = torch.from_numpy(down_gtlong_img).float().permute(2, 0, 1).contiguous()
        elif len(in_short_img.shape) == 4:
            out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(0, 3, 1, 2).contiguous()
            in_short_img = torch.from_numpy(in_short_img).float().permute(0, 3, 1, 2).contiguous()
            in_long_img = torch.from_numpy(in_long_img).float().permute(0, 3, 1, 2).contiguous()
            gt_long_img = torch.from_numpy(gt_long_img).float().permute(0, 3, 1, 2).contiguous()
            down_short_img = torch.from_numpy(down_short_img).float().permute(0, 3, 1, 2).contiguous()
            down_long_img = torch.from_numpy(down_long_img).float().permute(0, 3, 1, 2).contiguous()
            down_out_img = torch.from_numpy(down_out_img).float().permute(0, 3, 1, 2).contiguous()
            down_gtlong_img = torch.from_numpy(down_gtlong_img).float().permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError('undesired shape in sample.')

        sample = {'short_img': in_short_img,
                  'long_img': in_long_img,
                  'RGBout_img': out_srgb_img,
                  'gt_long_img': gt_long_img,
                  'down_short_img': down_short_img,
                  'down_long_img': down_long_img,
                  'down_out_img': down_out_img,
                  'down_gtlong_img': down_gtlong_img,
                  'in_short_path': in_short_path}

        return sample


class TP_dataset_v1(TP_dataset):

    def __init__(self, dataset_opt, tag):
        super(TP_dataset_v1, self).__init__(dataset_opt, tag)
    
    def downsample_tensors(self, tensors):
        """
        downsampling tensors as the inputs for the DeblurNet
        tensors: list or Tensor
        """
        if isinstance(self.opt.deblur_size, list):
            #deblur_size = random.sample(self.opt.deblur_size, k = 1)[0]
            deblur_size = random.uniform(self.opt.deblur_size[0], self.opt.deblur_size[1])
            deblur_size = int(deblur_size // 16 * 16)
        else:
            deblur_size = self.opt.deblur_size

        if isinstance(tensors, list):
            ret_tensors = [[] for _ in range(len(tensors))]
            for i in range(len(tensors)):
                ret_tensors[i] = F.upsample(tensors[i], size = (deblur_size, deblur_size), mode = 'area')
            return tuple(ret_tensors)
        else:
            ret_tensor = F.upsample(tensors, size = (deblur_size, deblur_size), mode = 'area')
            return ret_tensor
            

    def __getitem__(self, index):
        # Read images
        in_short_path = self.in_short_filelist[index]
        if self.tag == 'train':
            l = random.randint(1, 4)
            if l == 1:
                in_long_path = self.in_long4_filelist[index]
            elif l == 2:
                in_long_path = self.in_long6_filelist[index]
            elif l == 3:
                in_long_path = self.in_long8_filelist[index]
            elif l == 4:
                in_long_path = self.in_long2_filelist[index]
        else:
            in_long_path = self.in_long_filelist[index]
        out_srgb_path = self.out_srgb_filelist[index]

        in_short_img = read_img(self.opt, in_short_path)
        in_long_img = read_img(self.opt, in_long_path)
        out_srgb_img = read_img(self.opt, out_srgb_path)

        # Normalization and add noise
        in_short_img = in_short_img.astype(np.float) / 255.0
        in_long_img = in_long_img.astype(np.float) / 255.0
        out_srgb_img = out_srgb_img.astype(np.float) / 255.0
        
        if hasattr(self.opt, 'color_adjust') and self.opt.color_adjust:
            in_short_img = DA.color_adjustment(in_short_img, self.opt.color_adjust.prob)

        if self.opt.random_crop:
            if hasattr(self.opt, 'deblur_patch_per_image'):
                patch_per_image = self.opt.deblur_patch_per_image
            else:
                patch_per_image = 1

            if hasattr(self.opt, 'deblur_crop_size'):
                deblur_crop_size = self.opt.deblur_crop_size
            else:
                deblur_crop_size = 1024

            in_short_img, in_long_img, out_srgb_img, gt_long_img = self.crop_patch(in_short_img, in_long_img, out_srgb_img, patch_per_image, deblur_crop_size)
            
            # Get blur patch
            if hasattr(self.opt, 'blur_path'):
                assert deblur_crop_size <= 1024
                # NHWC
                in_short_blur_patches, in_long_blur_patches, out_blur_patches, gt_long_blur_patches = self.get_blur_patch(crop_size = deblur_crop_size)
                in_short_img = np.concatenate((in_short_img, in_short_blur_patches), axis = 0)
                in_long_img = np.concatenate((in_long_img, in_long_blur_patches), axis = 0)
                out_srgb_img = np.concatenate((out_srgb_img, out_blur_patches), axis = 0)
                gt_long_img = np.concatenate((gt_long_img, gt_long_blur_patches), axis = 0)

        else:
            gt_long_img = deepcopy(in_long_img)
            if self.opt.shot_noise:
                in_short_img = DA.add_shot_noise(in_short_img, iso = self.opt.shot_short_iso, noise = self.opt.shot_noise_mode)
                in_long_img = DA.add_shot_noise(in_long_img, iso = self.opt.shot_long_iso, noise = self.opt.shot_noise_mode)
            elif self.opt.noise_aug:
                a = random.randint(6, 8)
                noise_short = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6 * a
                noise_long = np.random.normal(loc = 0.0, scale = self.opt.noise_level, size = in_short_img.shape) * 0.6
                in_short_img = in_short_img + noise_short
                in_long_img = in_long_img + noise_long
                in_short_img = np.clip(in_short_img, 0, 1)
                in_long_img = np.clip(in_long_img, 0, 1)

        if hasattr(self.opt, 'cutnoise') and self.tag == "train":
            in_short_img = DA.cutnoise(in_short_img, out_srgb_img, self.opt.cutnoise.size, self.opt.cutnoise.prob)
            # if hasattr(self.opt.cutnoise, 'long') and self.opt.cutnoise.long:
            #     in_long_img = DA.cutnoise(in_long_img, out_srgb_img, self.opt.cutnoise.size, self.opt.cutnoise.prob)

        # to tensor
        if len(in_short_img.shape) == 3:
            out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(2, 0, 1).contiguous()
            in_short_img = torch.from_numpy(in_short_img).float().permute(2, 0, 1).contiguous()
            in_long_img = torch.from_numpy(in_long_img).float().permute(2, 0, 1).contiguous()
            gt_long_img = torch.from_numpy(gt_long_img).float().permute(2, 0, 1).contiguous()
        elif len(in_short_img.shape) == 4:
            out_srgb_img = torch.from_numpy(out_srgb_img).float().permute(0, 3, 1, 2).contiguous()
            in_short_img = torch.from_numpy(in_short_img).float().permute(0, 3, 1, 2).contiguous()
            in_long_img = torch.from_numpy(in_long_img).float().permute(0, 3, 1, 2).contiguous()
            gt_long_img = torch.from_numpy(gt_long_img).float().permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError('undesired shape in sample.')

        sample = {'short_img': in_short_img,
                  'long_img': in_long_img,
                  'RGBout_img': out_srgb_img,
                  'gt_long_img': gt_long_img,
                  'in_short_path': in_short_path}

        return sample


class RAW3RGB_dataset(data.Dataset):
    """RAW 域三帧融合数据集

    加载 3 帧 RAW（短1-长-短2）+ 1 帧 RGB GT。
    RAW 格式：V4L2 SBGGR10P（OV13855, 10-bit packed Bayer, BGGR 排列）。

    目录结构:
        data/raw/train/scene001/short1.raw   # V4L2 SBGGR10P packed
        data/raw/train/scene001/long.raw
        data/raw/train/scene001/short2.raw
        data/raw/train/scene001/gt.png       # sRGB GT（全分辨率）

    分辨率约定:
        crop_size 表示网络处理分辨率（RAW RGGB 平面分辨率）。
        网络输入 [B, 4, crop_size, crop_size]，输出 [B, 3, crop_size, crop_size]。
        GT 在加载后 resize 到 crop_size 以匹配网络输出。
    """

    def __init__(self, dataset_opt, tag):
        self.opt = dataset_opt
        self.tag = tag
        self.crop_size = getattr(self.opt, 'crop_size', 256)    # 网络处理分辨率（RAW RGGB 平面尺寸）
        self.patch_per_image = getattr(self.opt, 'patch_per_image', 4)
        self.random_crop = getattr(self.opt, 'random_crop', True)
        # RAW 图像宽高（从 meta.txt 读取，或使用默认值 1920x1080）
        self.raw_width = getattr(self.opt, 'raw_width', 1920)
        self.raw_height = getattr(self.opt, 'raw_height', 1080)
        self.black_level = getattr(self.opt, 'black_level', 64)

        if tag == 'train':
            base_path = self.opt.train_path
        else:
            base_path = self.opt.val_path

        # 扫描数据目录
        self.file_list = []
        if os.path.exists(base_path):
            for scene_dir in sorted(os.listdir(base_path)):
                scene_path = os.path.join(base_path, scene_dir)
                if os.path.isdir(scene_path):
                    # 检查必要文件是否存在
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
        存储顺序（小端）:
            byte0 = pix0[9:2], byte1 = pix1[9:2],
            byte2 = pix2[9:2], byte3 = pix3[9:2],
            byte4 = pix3[1:0]<<6 | pix2[1:0]<<4 | pix1[1:0]<<2 | pix0[1:0]

        参数:
            raw_bytes: 原始字节 (np.uint8)
            width: 图像宽度（像素数）
            height: 图像高度
            black_level: 黑电平值（OV13855 典型值 64 @ 10bit）
        返回:
            bayer: [height, width] float32, [0, 1]
        """
        total_pixels = width * height
        # 每 5 字节解出 4 个像素
        num_groups = total_pixels // 4
        raw_bytes = raw_bytes[:num_groups * 5]

        # reshape 为 [N, 5]
        data = raw_bytes.reshape(num_groups, 5)

        # 高 8 位
        p0_h = data[:, 0].astype(np.uint16)
        p1_h = data[:, 1].astype(np.uint16)
        p2_h = data[:, 2].astype(np.uint16)
        p3_h = data[:, 3].astype(np.uint16)

        # 低 2 位（从 byte4 解包）
        byte4 = data[:, 4]
        p0_l = (byte4 >> 0) & 0x03
        p1_l = (byte4 >> 2) & 0x03
        p2_l = (byte4 >> 4) & 0x03
        p3_l = (byte4 >> 6) & 0x03

        # 合成 10-bit 值
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
        """从 Bayer mosaic 提取 BGGR 4 平面 → RGGB 通道顺序

        OV13855 BGGR 排列:
            行0: B  Gb B  Gb ...
            行1: Gr R  Gr R  ...

        参数:
            bayer: [H, W] float32 Bayer mosaic
        返回:
            rggb: [4, H/2, W/2] float32, 顺序为 [R, Gr, Gb, B]
        """
        R  = bayer[1::2, 1::2]   # 奇行奇列
        Gr = bayer[1::2, 0::2]   # 奇行偶列
        Gb = bayer[0::2, 1::2]   # 偶行奇列
        B  = bayer[0::2, 0::2]   # 偶行偶列
        rggb = np.stack([R, Gr, Gb, B], axis=0)  # [4, H/2, W/2]
        return rggb

    def random_crop_start(self, h, w, crop_size, min_divide=2):
        """生成随机裁剪坐标，保持 Bayer 2x2 块对齐"""
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def _load_raw_frame(self, raw_path):
        """加载一帧 SBGGR10P RAW → [4, H/2, W/2] RGGB"""
        raw_bytes = np.fromfile(raw_path, dtype=np.uint8)
        bayer = self.unpack_sbggr10p(raw_bytes, self.raw_width, self.raw_height,
                                     black_level=self.black_level)
        rggb = self.bayer_to_rggb(bayer)  # [4, H/2, W/2]
        return rggb

    def __getitem__(self, index):
        if self.use_random:
            # 占位模式：返回随机数据（numpy，与真实路径一致）
            cs = self.crop_size
            short1 = np.random.randn(4, cs, cs).astype(np.float32)
            long = np.random.randn(4, cs, cs).astype(np.float32)
            short2 = np.random.randn(4, cs, cs).astype(np.float32)
            gt = np.random.randn(3, cs * 2, cs * 2).astype(np.float32)
        else:
            scene_path = self.file_list[index]

            # 1. 加载 3 帧 RAW
            short1 = self._load_raw_frame(os.path.join(scene_path, 'short1.raw'))
            long = self._load_raw_frame(os.path.join(scene_path, 'long.raw'))
            short2 = self._load_raw_frame(os.path.join(scene_path, 'short2.raw'))

            # 2. 加载 GT（sRGB），resize 到 RAW RGGB 平面 2 倍分辨率（全分辨率）
            gt = cv2.imread(os.path.join(scene_path, 'gt.png'))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            H, W = short1.shape[1], short1.shape[2]  # RAW RGGB 平面尺寸
            gt = cv2.resize(gt, (W * 2, H * 2), interpolation=cv2.INTER_AREA)
            gt = gt.transpose(2, 0, 1)  # [3, 2H, 2W]

            # 3. 随机裁剪（RAW 在 RGGB 平面裁剪，GT 在 2x 坐标裁剪）
            if self.random_crop:
                rh, rw = self.random_crop_start(H, W, self.crop_size, min_divide=1)
                short1 = short1[:, rh:rh+self.crop_size, rw:rw+self.crop_size]
                long = long[:, rh:rh+self.crop_size, rw:rw+self.crop_size]
                short2 = short2[:, rh:rh+self.crop_size, rw:rw+self.crop_size]
                # GT 裁剪坐标和尺寸是 RAW 的 2 倍
                gt_rh, gt_rw = rh * 2, rw * 2
                gt_crop = self.crop_size * 2
                gt = gt[:, gt_rh:gt_rh+gt_crop, gt_rw:gt_rw+gt_crop]

            # 4. 随机翻转增强（对所有帧一致）
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

        # 与 trainer 兼容的 key 命名
        # short_img: 两短帧拼接 [8, crop_size, crop_size]，由 trainer 拆分给 D2HNet_RK
        short_cat = np.concatenate([short1, short2], axis=0)  # [8, H, W]
        sample = {
            'short_img':  torch.from_numpy(short_cat).float(),  # [8, crop_size, crop_size]
            'long_img':   torch.from_numpy(np.asarray(long)).float(),     # [4, crop_size, crop_size]
            'RGBout_img': torch.from_numpy(np.asarray(gt)).float(),       # [3, crop_size, crop_size]
            'gt_long_img': torch.from_numpy(np.asarray(gt)).float(),      # 与 trainer 兼容
        }
        return sample

    def __len__(self):
        return len(self.file_list)


class TP_valdataset_singleimage(SLRGB2RGB_valdataset_singleimage):
    def __init__(self, src_path):
        super(TP_valdataset_singleimage, self).__init__(src_path = src_path)
    
    def __getitem__(self, index):

        # Read images
        in_short_path = self.shortlist[index]
        in_long_path = self.longlist[index]

        in_short_img = cv2.imread(in_short_path)
        in_long_img = cv2.imread(in_long_path)

        in_short_img = cv2.cvtColor(in_short_img, cv2.COLOR_BGR2RGB)
        in_long_img = cv2.cvtColor(in_long_img, cv2.COLOR_BGR2RGB)

        down_short_img = cv2.resize(in_short_img, (1024, 1024), interpolation = cv2.INTER_AREA)
        down_long_img = cv2.resize(in_long_img, (1024, 1024), interpolation = cv2.INTER_AREA)

        down_short_img = down_short_img.astype(np.float) / 255.
        down_long_img = down_long_img.astype(np.float) / 255.

        down_short_img = torch.from_numpy(down_short_img).float().permute(2, 0, 1)
        down_long_img = torch.from_numpy(down_long_img).float().permute(2, 0, 1)
        
        # Normalization and add noise
        in_short_img = in_short_img.astype(np.float) / 255.0
        in_long_img = in_long_img.astype(np.float) / 255.0

        # to tensor
        in_short_img = torch.from_numpy(in_short_img).float().permute(2, 0, 1).contiguous()
        in_long_img = torch.from_numpy(in_long_img).float().permute(2, 0, 1).contiguous()

        short_iso = read_iso(in_short_path.replace('png', 'txt'))
        long_iso = read_iso(in_long_path.replace('png', 'txt'))

        sample = {'down_short_img': down_short_img,
                  'down_long_img': down_long_img,
                  'in_short_img': in_short_img,
                  'in_long_img': in_long_img,
                  'short_iso': short_iso,
                  'long_iso': long_iso}

        return sample

    def __len__(self):
        return len(self.longlist)
        