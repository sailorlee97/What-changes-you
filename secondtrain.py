import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch.cuda
from torch.optim.lr_scheduler import StepLR

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from FlowFeatures import flowfeatures
from cnnmodel import ResNet
from dataset import BatchflowData
from exemplar import Exemplar
from model import BiasLayer
from utils import CpuGpuMonitor

class SecondDistributedTrainer:
    def __init__(self, world_size, total_cls, incremental_num_list, batch_size):
        self.world_size = 2
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = flowfeatures()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ResNet(classes=self.total_cls).cuda()
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        self.batch_size = batch_size
        self.bias_layers = [self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4]
        self.sample = incremental_num_list
        # total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Solver total trainable parameters : ", total_params)

    def init_ddp(self):
        # local_rank 当前进程号
        # 总进程数
        # torch.cuda.set_device(local_rank)
        # os.environ['RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12359'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        world_size = torch.cuda.device_count()
        os.environ['WORLD_SIZE'] = str(world_size)

        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        print(f"start rank:{rank}")
        return rank

    def load_model(self, rank):
        device_id = rank % torch.cuda.device_count()
        model = self.model
        # 加载模型检查点
        # checkpoint = torch.load('./model+0.pth', map_location=f'cuda:{device_id}')

        # model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device_id)
        model = DDP(model, device_ids=[device_id], broadcast_buffers=False, find_unused_parameters=True)
        self.bias_layers = [layer.to(device_id) for layer in self.bias_layers]
        self.bias_layers = [DDP(layer, device_ids=[device_id], broadcast_buffers=False, find_unused_parameters=True) for
                            layer in self.bias_layers]
        state_dict = torch.load('./model+0.pth')
        model.load_state_dict(state_dict)
        return model, device_id

    def create_loader(self, x, y):
        traindataset = BatchflowData(x, y)
        train_sampler = torch.utils.data.distributed.DistributedSampler(traindataset)

        trainloader = DataLoader(dataset=traindataset,
                                 pin_memory=True,
                                 shuffle=(train_sampler is None),  # 使用分布式训练 shuffle 应该设置为 False
                                 batch_size=self.batch_size,
                                 num_workers=0,
                                 sampler=train_sampler)
        return trainloader, train_sampler

    def train(self, args):

        rank = self.init_ddp()

        # 自己的数据获取第一轮数据
        total_cls = self.total_cls
        exemplar = Exemplar(8000, total_cls)
        dataset = self.dataset
        test_xs = []
        test_ys = []
        status = dataset.multi_dict
        train, val, test = dataset.getNextClasses(0)
        print('train:', len(train), 'val:', len(val), 'test:', len(test))
        train_x, train_y = zip(*train)
        val_x, val_y = zip(*val)
        test_x, test_y = zip(*test)

        train_xs, train_ys = exemplar.get_exemplar_train()
        train_xs.extend(train_x)
        train_xs.extend(val_x)
        train_ys.extend(train_y)
        train_ys.extend(val_y)
        exemplar.update(self.sample[0], (train_x, train_y), (val_x, val_y))
        # 加载模型
        model, device_id = self.load_model(rank)
        # 这个是第二轮
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4)
        scheduler = StepLR(optimizer, step_size=70, gamma=0.1)
        bias_optimizer = optim.Adam(self.bias_layers[1].parameters(), lr=0.001)

        ins = 1
        train, val, test = dataset.getNextClasses(ins)
        print('train:', len(train), 'val:', len(val), 'test:', len(test))
        train_x, train_y = zip(*train)
        val_x, val_y = zip(*val)

        train_xs_ft = [x for x, y in zip(train_xs, train_ys) if y not in status[ins][2]]
        train_ys_ft = [y for y in train_ys if y not in status[ins][2]]
        train_ys_ft_new, label_mapping_fine_tune = self.automate_label_mapping(train_ys_ft)
        trainftdataset, train_sampler = self.create_loader(train_xs_ft, train_ys_ft_new)

        monitor1 = CpuGpuMonitor()
        monitor1.start()
        start_time = time.time()

        for param in model.parameters():
            param.requires_grad = False
        for param in model.module.fc.parameters():
            param.requires_grad = True
        #     先微调模型
        for epoch in range(int(args.epoch * 0.6)):  # 假设训练5个epoch
            train_sampler.set_epoch(args.epoch)
            if device_id == 0:
                print("---" * 50)
                print("Epoch", epoch)
            # cur_lr = self.get_lr(optimizer)
            # print("Current Learning Rate : ", cur_lr)
            model.train()
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
            self.stage1_fine_tune(device_id, model, trainftdataset, criterion, optimizer, ins, status)
            scheduler.step()

        # 参数解冻
        for param in model.parameters():
            param.requires_grad = True

        end_time = time.time()
        training_time1 = end_time - start_time

        if device_id == 0:
            print('bias- incremental')
        exemplar.update_reduce(self.sample[ins], (train_x, train_y), (val_x, val_y), status[ins][2])
        train_xs, train_ys = exemplar.get_exemplar_train()
        train_xs.extend(train_x)
        train_xs.extend(val_x)
        train_ys.extend(train_y)
        train_ys.extend(val_y)
        train_ys_newlabel, label_mapping = self.automate_label_mapping(train_ys)
        train_data, train_sampler = self.create_loader(train_xs, train_ys_newlabel)

        self.seen_cls = exemplar.get_cur_cls()
        # print("seen cls number : ", self.seen_cls)
        # val_xs, val_ys = exemplar.get_exemplar_val()
        # val_ys_newlabel, label_mapping = self.automate_label_mapping(val_ys)
        # val_bias_data, val_bias_data_sampler= self.create_loader(val_xs, val_ys_newlabel)

        start_time = time.time()
        # test_acc = []
        for epoch in range(args.epoch):
            train_sampler.set_epoch(args.epoch)
            if device_id == 0:
                print("---" * 50)
                print("Epoch", epoch)

            # cur_lr = self.get_lr(optimizer)
            # print("Current Learning Rate : ", cur_lr)
            model.train()
            for _ in range(len(self.bias_layers)):
                self.bias_layers[_].eval()
            self.stage1_distill_status(device_id, model, train_data, optimizer, ins, status, label_mapping)
            scheduler.step()
        end_time = time.time()
        cpu, gpu = monitor1.end()
        training_time2 = end_time - start_time
        print(f"second model time: {training_time1 + training_time2:.2f} s")
        with open('data.txt', 'w') as f:
            f.write(f"second train time：{training_time1 + training_time2:.2f} s")
            f.write(f"Average CPU usage: {cpu}%")
            f.write(f"Average GPU usage: {gpu}%")

    def stage1_distill_status(self, device_id, model, train_data, optimizer, inc, status, label_mapping):
        # if device_id == 0:
        #   print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - self.sample[inc]) / self.seen_cls
        # print("classification proportion 1-alpha = ", 1-alpha)
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda(device_id, non_blocking=True)
            label = label.view(-1).cuda(device_id, non_blocking=True)

            p = model(x)
            p, num = self.bias_forward_new(p, inc, status)
            with torch.no_grad():
                pre_p = model(x)
                pre_p, num_output = self.bias_forward_new(pre_p, inc, status)
                # 老模型中老应用的loss
                pre_p = F.softmax(pre_p[:, :self.seen_cls - self.sample[inc]] / T, dim=1)
            # 新模型中保证老应用的loss
            logp = F.log_softmax(p[:, :self.seen_cls - self.sample[inc]] / T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))

            loss_hard_target = nn.CrossEntropyLoss()(p[:, :self.seen_cls], label)
            loss = loss_soft_target * T * T + (1 - alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        if device_id == 0:
            print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))

    def stage1_fine_tune(self, device_id, model, train_data, criterion, optimizer, inc, status):
        # if device_id == 0:
        #     print("Training ... ")
        losses_fine_tune = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda(device_id, non_blocking=True)
            label = label.view(-1).cuda(device_id, non_blocking=True)
            p = model(x)
            p, num_output = self.bias_forward_new(p, inc, status)
            # a = p[:, :self.seen_cls]
            loss = criterion(p[:, :self.seen_cls - len(status[inc][2])], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_fine_tune.append(loss.item())
        if device_id == 0:
            print("stage fine_tune loss :", np.mean(losses_fine_tune))

    def automate_label_mapping(self, original_labels):
        # 训练前用
        unique_labels = sorted(set(original_labels))
        label_mapping = {label: index for index, label in enumerate(unique_labels)}
        new_labels = [label_mapping[label] for label in original_labels]
        return new_labels, label_mapping

    def inverse_label_mapping(self, new_labels, label_mapping):
        # 测试的时候用
        new_labels_cuda = torch.tensor(new_labels).cuda()
        new_labels_list = new_labels_cuda.cpu().tolist()

        inverse_mapping = {v: k for k, v in label_mapping.items()}
        original_labels = [inverse_mapping[label] for label in new_labels_list]
        return torch.tensor(original_labels).to('cuda')

    def bias_forward_new(self, input, inc, status):
        # 0 1 2
        if inc > 0:
            if status[inc][2]:
                # 老应用个数
                in1 = input[:, :status[inc][3] - len(status[inc][1])]
                in2 = input[:, status[inc][3] - len(status[inc][1]):status[inc][3]]
                out1 = self.bias_layer1(in1)
                out2 = self.bias_layer2(in2)
                if in1.shape[1] + in2.shape[1] == input.shape[1]:
                    return torch.cat([out1, out2], dim=1), 2
                else:
                    in3 = input[:, status[inc][3]:]
                    out3 = self.bias_layer3(in3)
                    return torch.cat([out1, out2, out3], dim=1), 3
            else:
                in1 = input[:, :status[inc][3]]
                in2 = input[:, status[inc][3]:status[inc][3] + len(status[inc][1])]
                out1 = self.bias_layer1(in1)
                out2 = self.bias_layer2(in2)
                if in1.shape[1] + in2.shape[1] == input.shape[1]:
                    return torch.cat([out1, out2], dim=1), 2
                else:
                    in3 = input[:, status[inc][3] + len(status[inc][1]):]
                    out3 = self.bias_layer3(in3)
                    return torch.cat([out1, out2, out3], dim=1), 3
        else:
            in1 = input[:, :self.sample[0]]
            in2 = input[:, self.sample[0]:]
            out1 = self.bias_layer1(in1)
            out2 = self.bias_layer2(in2)
            return torch.cat([out1, out2], dim=1), 2