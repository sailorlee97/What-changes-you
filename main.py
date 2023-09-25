import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse

# torch.cuda.device_count()
# os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % m_gpu
# torch.cuda.set_device(m_gpu)
# torch.cuda.is_available()
# torch.cuda.current_device()
parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 16, type = int)
parser.add_argument('--epoch', default = 50, type = int)
parser.add_argument('--lr', default = 0.001, type = int)
parser.add_argument('--max_size', default = 8000, type = int)
parser.add_argument('--total_cls', default = 9, type = int)
parser.add_argument('--incremental_list', default = [4,1,2,3], type = list)
args = parser.parse_args()


if __name__ == "__main__":
    # showGod()
    trainer = Trainer(args.total_cls,args.incremental_list)
    trainer.train_target_apps(args.batch_size, args.epoch, args.lr, args.max_size)