import torch
from secondtrain import SecondDistributedTrainer
import argparse
import os

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--args', help="priority", type=bool, required=False, default=True)
parser.add_argument('--batch_size', default = 256, type = int)
parser.add_argument('--epoch', default = 60, type = int)
parser.add_argument('--lr', default = 0.0002, type = int)
parser.add_argument('--max_size', default = 8000, type = int)
parser.add_argument('--total_cls', default = 19, type = int)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--incremental_list', default = [16,3], type = list)

args = parser.parse_args()

if __name__ == '__main__':
    trainer = SecondDistributedTrainer( 2,args.total_cls, args.incremental_list, args.batch_size)
    trainer.train(args)