import os
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from util import utils
from dataloader import dataset
from models.TrainingModule import TrainingModule, LossManager, pack_network_output, pack_gt_data
from models.utils import create_generator, create_generator_val


class UnifiedTrainer(TrainingModule):
    """单阶段 RAW 域三帧融合训练器"""

    def __init__(self, opt, num_gpus, rank=None, world_size=None):
        super(UnifiedTrainer, self).__init__(opt=opt,
                                             num_gpus=num_gpus,
                                             rank=rank,
                                             world_size=world_size)

        self.Training_config = self.opt.Training_config
        self.optim_config = self.opt.Optimizer

        self.G = create_generator(opt.GNet)
        self.G = self.wrapper(self.G)

        self.LM = LossManager(self.opt.Loss, num_gpus=num_gpus)

        self._init_dataloader()
        self._init_optim()

    def _init_dataloader(self):
        dataset_type = getattr(self.opt.Dataset, 'dataset_type', 'ov13855')
        if dataset_type == 'imx585':
            from dataloader import dataset_imx585
            train_dataset = dataset_imx585.IMX585_RAW3RGB_dataset(self.opt.Dataset, 'train')
            val_dataset = dataset_imx585.IMX585_RAW3RGB_dataset(self.opt.Dataset, 'val')
        else:
            train_dataset = dataset.RAW3RGB_dataset(self.opt.Dataset, 'train')
            val_dataset = dataset.RAW3RGB_dataset(self.opt.Dataset, 'val')
        print('The overall number of training images:', len(train_dataset))
        print('The overall number of validation images:', len(val_dataset))

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.Training_config.train_batch_size,
            shuffle=True,
            num_workers=self.Training_config.num_workers,
            pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.Training_config.val_batch_size,
            shuffle=False,
            num_workers=self.Training_config.num_workers,
            pin_memory=True)

    def _init_optim(self):
        if self.optim_config.name == "Adam":
            self.optim_G = torch.optim.Adam(
                self.G.parameters(),
                lr=self.optim_config.args.lr_g,
                betas=(self.optim_config.args.b1, self.optim_config.args.b2),
                weight_decay=self.optim_config.args.weight_decay)
        elif self.optim_config.name == "SGD":
            self.optim_G = torch.optim.SGD(
                self.G.parameters(),
                lr=self.optim_config.args.lr_g)
        elif self.optim_config.name == "AdamW":
            self.optim_G = torch.optim.AdamW(
                self.G.parameters(),
                lr=self.optim_config.args.lr_g,
                betas=(self.optim_config.args.b1, self.optim_config.args.b2),
                weight_decay=self.optim_config.args.weight_decay)

    def train(self):
        iters_done = 0

        for epoch in range(self.Training_config.start_idx + 1, self.Training_config.epochs):
            print('epoch', epoch)

            for param_group in self.optim_G.param_groups:
                self.add_scalar('lr', param_group['lr'], epoch)

            for i, data in enumerate(self.train_loader):
                print(i, self.device)

                short_cat = data['short_img'].to(self.device)
                long = data['long_img'].to(self.device)
                gt = data['RGBout_img'].to(self.device)

                out = self.G(short_cat, long)

                gt_dict = pack_gt_data([gt, gt])
                outputs = pack_network_output(out, self.opt.GNet.name)

                G_loss, G_loss_info = self.LM(outputs, gt_dict)

                self.optim_G.zero_grad()
                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                self.optim_G.step()

                if iters_done % self.Training_config.show_loss_iter == 0:
                    self.add_scalars(main_tag='G_loss', tag_scalar_dict=G_loss_info, global_step=iters_done)
                if iters_done % self.Training_config.show_img_iter == 0:
                    vis_imgs = [gt, out.clamp(0.0, 1.0)]
                    self.visual_image('train_img', vis_imgs, iters_done)

                self._save_G(self.opt, epoch, iters_done, len(self.train_loader), self.G)
                self._adjust_learning_rate(self.optim_config, (epoch + 1), iters_done, self.optim_G, self.opt.Optimizer.args.lr_g)

                iters_done += 1

            self._validate(epoch)

    def _validate(self, epoch):
        self.G.eval()
        val_PSNR, val_SSIM, num_of_val_image = 0, 0, 0

        for j, data in enumerate(self.val_loader):

            short_cat = data['short_img'].to(self.device)
            long = data['long_img'].to(self.device)
            gt = data['RGBout_img'].to(self.device)

            with torch.no_grad():
                out = self.G(short_cat, long)

            if isinstance(out, list):
                out = out[0]

            out = out.clamp(0.0, 1.0)

            num_of_val_image += out.shape[0]
            val_PSNR += utils.psnr(out, gt, 1) * out.shape[0]
            val_SSIM += utils.ssim(out, gt) * out.shape[0]

            if j % 10 == 0:
                print('val: %d | epoch: %d' % (j, epoch))

        val_PSNR = val_PSNR / num_of_val_image
        val_SSIM = val_SSIM / num_of_val_image

        self.add_scalar('val_PSNR', val_PSNR, global_step=epoch)
        self.add_scalar('val_SSIM', val_SSIM, global_step=epoch)

        self.G.train()

        print('val: epoch: %d, psnr: %.3f, ssim: %.3f' % (epoch, val_PSNR, val_SSIM))
