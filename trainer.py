import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR
import warnings
from cnnmodel import CNN, ResNet

warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from dataset import BatchData, BatchflowData
from model import PreResNet, BiasLayer
from cifar import Cifar100
from FlowFeatures import  flowfeatures
from exemplar import Exemplar
from copy import deepcopy

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Trainer:
    def __init__(self, total_cls):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = flowfeatures()
        self.model = ResNet().cuda()
        print(self.model)
        # self.model = nn.DataParallel(self.model, device_ids=[0,1])
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        # self.bias_layer5 = BiasLayer().cuda()
        self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4]
        self.sample = [4,3,3,3]
        # self.input_transform= Compose([
        #                         transforms.RandomHorizontalFlip(),
        #                         transforms.RandomCrop(32,padding=4),
        #                         ToTensor(),
        #                         Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        # self.input_transform_eval= Compose([
        #                         ToTensor(),
        #                         Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)

    def test(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (x, label) in enumerate(testdata):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p = self.bias_forward(p)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    def eval(self, criterion, evaldata):
        self.model.eval()
        losses = []
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(evaldata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            loss = criterion(p, label)
            losses.append(loss.item())
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        print("Validation Loss: {}".format(np.mean(losses)))
        print("Validation Acc: {}".format(100*correct/(correct+wrong)))
        self.model.train()
        return

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)
        # previous_model = None
        dataset = self.dataset
        test_xs = []
        test_ys = []
        # train_xs = []
        # train_ys = []
        # sample = [4, 3, 3, 3]
        test_accs = []
        for inc_i in range(dataset.batch_num):
            """
            这一段的作用是更新类别
            一共更新4次，每次更新10类，即40/10
            """
            print(f"Incremental num : {inc_i}")
            train, val, test = dataset.getNextClasses(inc_i)
            print(len(train), len(val), len(test))
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            """
            获取训练数据集 训练集由 训练和验证组成
            """
            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_xs.extend(val_x)
            train_ys.extend(train_y)
            train_ys.extend(val_y)


            train_data = DataLoader(BatchflowData(train_xs, train_ys),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            # val_data = DataLoader(BatchflowData(val_x, val_y),
            #             batch_size=batch_size, shuffle=False)
            test_data = DataLoader(BatchflowData(test_xs, test_ys),
                        batch_size=batch_size, shuffle=False)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

            # bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
            bias_optimizer = optim.Adam(self.bias_layers[inc_i].parameters(), lr=0.001)
            # bias_scheduler = StepLR(bias_optimizer, step_size=70, gamma=0.1)

            """
            
            """
            exemplar.update(self.sample[inc_i], (train_x, train_y), (val_x, val_y))
            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)
            val_xs, val_ys = exemplar.get_exemplar_val()
            val_bias_data = DataLoader(BatchflowData(val_xs, val_ys), batch_size=16, shuffle=True, drop_last=False)
            test_acc = []

            for epoch in range(epoches):
                print("---"*50)
                print("Epoch", epoch)
                scheduler.step()
                cur_lr = self.get_lr(optimizer)
                print("Current Learning Rate : ", cur_lr)
                self.model.train()
                for _ in range(len(self.bias_layers)):
                    self.bias_layers[_].eval()
                if inc_i > 0:
                    self.stage1_distill(train_data, optimizer)
                else:
                    self.stage1(train_data, criterion, optimizer)
                acc = self.test(test_data)
            if inc_i > 0:
                for epoch in range(epoches):
                    # bias_scheduler.step()
                    self.model.eval()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].train()
                    self.stage2(val_bias_data, criterion, bias_optimizer)
                    if epoch % 50 == 0:
                        acc = self.test(test_data)
                        test_acc.append(acc)
            for i, layer in enumerate(self.bias_layers):
                layer.printParam(i)

            # 一次增量结束之后，保存本次模型
            self.previous_model = deepcopy(self.model)
            acc = self.test(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)

    def bias_forward(self, input):
        in1 = input[:, :self.sample[0]]
        in2 = input[:, self.sample[0]:self.sample[0]+self.sample[1]]
        in3 = input[:, self.sample[0]+self.sample[1]:self.sample[0]+self.sample[1]+self.sample[2]]
        in4 = input[:, self.sample[0]+self.sample[1]+self.sample[2]:self.sample[0]+self.sample[1]+self.sample[2]+self.sample[3]]
        # in5 = input[:, 80:100]
        # 用于区分第一批分类，标签0-3
        out1 = self.bias_layer1(in1)
        # 区分第二批分类，标签4-6
        out2 = self.bias_layer2(in2)
        # 区分第三批分类，标签7-9
        out3 = self.bias_layer3(in3)
        # 区分第四批分类，标签10-12
        out4 = self.bias_layer4(in4)
        # out5 = self.bias_layer5(in5)
        return torch.cat([out1, out2, out3, out4], dim = 1)

    def stage1_distill(self, train_data, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - 10)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(x)
                pre_p = self.bias_forward(pre_p)
                # 上一次模型的准确率
                pre_p = F.softmax(pre_p[:,:self.seen_cls-10]/T, dim=1)
            # 保证老应用的loss
            logp = F.log_softmax(p[:,:self.seen_cls-10]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage2(self, val_bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        for i, (x, label) in enumerate(tqdm(val_bias_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))
