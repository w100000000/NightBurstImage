import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from . import loss as L
import util.parallel as P
import util.visualboard as V


def pack_network_output(out):
    if isinstance(out, torch.Tensor):
        return {'output': out}
    elif isinstance(out, list) or isinstance(out, tuple):
        return {'output': out[0]}
    else:
        raise ValueError("network output format is illegal.")


def pack_gt_data(data: list):
    gt_dict = {}
    gt_dict['RGBout_img'] = data[0]
    gt_dict['gt_long_img'] = data[1]
    return gt_dict


class LossManager(L.LossBase):

    def __init__(self, loss_conf, num_gpus):
        super(LossManager, self).__init__(num_gpus = num_gpus)
        self.loss_conf = loss_conf
        self.criterions = {}
        for k, v in loss_conf.items():
            print('LossManager:', k, v)
            if hasattr(nn, k):
                if hasattr(v, 'args'):
                    func = getattr(nn, k)(**(v.args))
                else:
                    func = getattr(nn, k)()
                self.criterions[k] = func.to(self.device)
            else:
                if hasattr(v, 'args'):
                    func = getattr(L, k)(**(v.args))
                else:
                    func = getattr(L, k)()
                self.criterions[k] = func.to(self.device)

    def __call__(self, output, gt_dict):
        out = output['output']
        gt_img = gt_dict['RGBout_img']

        loss_info = {}
        for k, v in self.criterions.items():
            loss_info[k] = v(out, gt_img)

        loss_sum = 0.
        for k, v in loss_info.items():
            loss_sum += self.loss_conf[k].weight * v
            loss_info[k] = round(self.loss_conf[k].weight * v.item(), 5)

        return loss_sum, loss_info


class TrainingModule(P.Parallel, V.VisualBoard):

    def __init__(self, opt, num_gpus:int, rank:int = None, world_size:int = None):
        P.Parallel.__init__(self, num_gpus = num_gpus, rank = rank, world_size = world_size)
        V.VisualBoard.__init__(self, log_path = opt.log_path)

        self.opt = opt
        cudnn.benchmark = opt.cudnn_benchmark

        self.save_folder = opt.save_path
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        if not os.path.exists(os.path.join(self.save_folder, "GNet")):
            os.makedirs(os.path.join(self.save_folder, 'GNet'))
        if not os.path.exists(os.path.join(self.save_folder, 'sample')):
            os.makedirs(os.path.join(self.save_folder, 'sample'))

        print("There are %d GPUs used" % self.num_gpus)
        if num_gpus > 0:
            self.opt.Training_config.train_batch_size *= num_gpus
            self.opt.Training_config.val_batch_size *= num_gpus

    def _adjust_learning_rate(self, optim_conf, epoch, iter, optimizer, lr_gd):
        if optim_conf.lr_decrease_mode == 'epoch':
            lr = lr_gd * (optim_conf.lr_decrease_factor ** (epoch // optim_conf.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if optim_conf.lr_decrease_mode == 'iter':
            lr = lr_gd * (optim_conf.lr_decrease_factor ** (iter // optim_conf.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def _save_G(self, opt, epoch, iter, len_dataset, net):
        if opt.save_mode == 'epoch':
            save_path = os.path.join(self.save_folder, 'GNet', 'GNet-epoch-%d.pkl' % epoch)
        elif opt.save_mode == 'iter':
            save_path = os.path.join(self.save_folder, 'GNet', 'GNet-iter-%d.pkl' % iter)

        self._save_model(opt, epoch, iter, len_dataset, net, save_path)

    def _save_model(self, opt, epoch, iter, len_dataset, net, save_path):
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            save_state_dict = net.module.state_dict()
        else:
            save_state_dict = net.state_dict()

        if opt.save_mode == 'epoch':
            if epoch % opt.save_by_epoch == 0 and iter % len_dataset == 0:
                torch.save(save_state_dict, save_path)
                print('The trained model is successfully saved in %s' % (save_path))
        elif opt.save_mode == 'iter':
            if iter % opt.save_by_iter == 0:
                torch.save(save_state_dict, save_path)
                print('The trained model is successfully saved in %s' % (save_path))

    def train(self):
        raise NotImplementedError('Not implemented')

    def finish(self):
        self.close()
