import argparse
import os
import torch
import yaml
from easydict import EasyDict as edict

from trainer import UnifiedTrainer


def attatch_to_config(args, opt):
    opt.num_gpus = args.num_gpus
    opt.save_path = args.save_path
    opt.log_path = args.log_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type = str, default = 'options/nbinet.yaml', help = 'Path to option YAML file.')
    parser.add_argument('--num_gpus', type = int, default = 1, help = 'GPU numbers, 0 means cpu is used.')
    parser.add_argument('--save_path', type = str, default = 'snapshot/nbinet', help = 'Path to save model.')
    parser.add_argument('--log_path', type = str, default = 'log_pt/nbinet', help = 'Path to write log.')
    args = parser.parse_args()

    with open(args.opt, mode = 'r') as f:
        opt = edict(yaml.load(f, Loader=yaml.FullLoader))

    attatch_to_config(args, opt)

    print(opt)

    trainer = UnifiedTrainer(opt = opt, num_gpus = opt.num_gpus)
    trainer.train()
    trainer.finish()
