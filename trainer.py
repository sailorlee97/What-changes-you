import pandas as pd
import torch
import torchvision
from sklearn.metrics import classification_report
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
from plot_ml import plot_conf

warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
from dataset import BatchData, BatchflowData
from model import PreResNet, BiasLayer
from FlowFeatures import  flowfeatures
from exemplar import Exemplar
from copy import deepcopy

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Trainer:
    def __init__(self, total_cls,incremental_num_list):
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = flowfeatures()
        self.model = ResNet(classes=self.total_cls).cuda()
        # print(self.model)
        # self.model = nn.DataParallel(self.model, device_ids=[0,1])
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        # self.bias_layer5 = BiasLayer().cuda()
        self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4]
        self.sample = incremental_num_list
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    def test_fine_tine_data(self, testdata,mappping,inc,status):
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
            p, num_output = self.bias_forward_new(p, inc,status)
            pred = p[:, :self.seen_cls - len(status[inc][2])].argmax(dim=-1)
            pred_leverage = self.inverse_label_mapping(pred ,mappping)
            correct += sum(pred_leverage == label).item()
            wrong += sum(pred_leverage != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    def test_data(self, testdata,mappping,inc,status):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        pred_list = []
        label_list = []

        for i, (x, label) in enumerate(testdata):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, inc,status)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            pred_leverage = self.inverse_label_mapping(pred ,mappping)
            correct += sum(pred_leverage == label).item()
            wrong += sum(pred_leverage != label).item()

            pred_list.append(pred_leverage)
            label_list.append(label)

        pred_py = torch.cat(pred_list, dim=0)
        label_py = torch.cat(label_list, dim=0)

        pred_arr = pred_py.detach().cpu().numpy()
        label_arr = label_py.detach().cpu().numpy()
        print('predict_label : {}'.format(list(set(pred_arr))))
        print('true_label : {}'.format(list(set(label_arr))))
        res_key = list(map(str, mappping.keys()))
        plot_conf(pred_arr, label_arr, res_key, name=inc)
        report = classification_report(label_arr, pred_arr, digits=4, target_names=res_key, output_dict=True)

        print(report)
        df = pd.DataFrame(report).transpose()
        df.to_csv("{}.csv".format(inc), index=True)

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

    def train_target_apps(self,batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)
        # previous_model = None
        dataset = self.dataset
        status = dataset.multi_dict

        test_xs = []
        test_ys = []
        test_accs = []
        finetune_accs = []

        for inc_i in range(dataset.batch_num):
            """
            这一段的作用是更新类别
            """
            print(f"Incremental num : {inc_i}")
            train, val, test = dataset.getNextClasses(inc_i)
            print('train:', len(train),'val:', len(val), 'test:', len(test))
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            # 第一轮是空的数据，第二轮开始获取上一轮存储的网络流量数据，再加入到现在这一轮中
            # train_xs, train_ys = exemplar.get_exemplar_train()
            #
            # train_xs.extend(train_x)
            # train_xs.extend(val_x)
            # train_ys.extend(train_y)
            # train_ys.extend(val_y)
            #
            # train_data = DataLoader(BatchflowData(train_xs, train_ys),
            #                         batch_size=batch_size, shuffle=True, drop_last=True)

            test_data = DataLoader(BatchflowData(test_xs, test_ys),
                                   batch_size=batch_size, shuffle=False)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

            # bias_optimizer = optim.SGD(self.bias_layers[inc_i].parameters(), lr=lr, momentum=0.9)
            # 这里的bias是对增量的元素进行 如果发生增量则定义
            if inc_i>0:
                bias_optimizer = optim.Adam(self.bias_layers[1].parameters(), lr=0.001)
            else:
                bias_optimizer = optim.Adam(self.bias_layers[0].parameters(), lr=0.001)

            """
            从这边需要开始有区分
            """
            if inc_i > 0:
            # fine-tune + 增量
                if status[inc_i][0] == 1:
                    print('开始 fine - tune！')
                    # num_classes = status[inc_i][3] - len(status[inc_i][1])  # 假设有11个新类别
                    # self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
                    # testdata里面淘汰的数据都删了
                    test_xs = [x for x, y in zip(test_xs, test_ys) if y not in status[inc_i][2]]
                    test_ys = [y for y in test_ys if y not in status[inc_i][2]]
                    test_data = DataLoader(BatchflowData(test_xs, test_ys),
                                           batch_size=batch_size, shuffle=False)

                    train_xs_ft = [x for x, y in zip(train_xs, train_ys) if y not in status[inc_i][2] ]
                    train_ys_ft = [y for y in train_ys if y not in status[inc_i][2]]
                    train_ys_ft_new, label_mapping_fine_tune = self.automate_label_mapping(train_ys_ft)
                    train_loader = DataLoader(BatchflowData(train_xs_ft, train_ys_ft_new),
                                            batch_size=batch_size, shuffle=True, drop_last=True)

                    self.model.to(self.device)
                    for param in self.model.parameters():
                        param.requires_grad = False
                    for param in self.model.fc.parameters():
                        param.requires_grad = True
                    #     先微调模型
                    for epoch in range(epoches):  # 假设训练5个epoch
                        scheduler.step()
                        cur_lr = self.get_lr(optimizer)
                        print("Current Learning Rate : ", cur_lr)
                        self.model.train()
                        for _ in range(len(self.bias_layers)):
                            self.bias_layers[_].eval()
                        self.stage1_fine_tune(train_loader,criterion, optimizer, inc_i, status, label_mapping_fine_tune)

                    # 参数解冻
                    for param in self.model.parameters():
                        param.requires_grad = True
                    # for epoch in range(epoches*2):
                    #     # bias_scheduler.step()
                    #     self.model.eval()
                    #     for _ in range(len(self.bias_layers)):
                    #         self.bias_layers[_].train()
                    #     # 这里优化老应用的参数，所以选用bias_layer1
                    #     self.stage2_status_fine_tune(train_loader, criterion, optim.Adam(self.bias_layers[0].parameters(), lr=0.001),inc_i,status)
                    #     if epoch % 50 == 0:
                    #         acc = self.test_fine_tine_data(test_data,label_mapping_fine_tune, inc= inc_i, status = status)
                    #         finetune_accs.append(acc)
                    # for epoch in range(epoches):  # 假设训练5个epoch
                    #     # train_data 需要换了 这里的train data里面的label需要删除
                    #     '''
                    #     标签不一致 需要修改
                    #     '''
                    #     for x, labels in train_loader:
                    #         x = x.type(torch.FloatTensor)
                    #         x = x.cuda()
                    #         labels = labels.view(-1).cuda()
                    #         optimizer.zero_grad()
                    #         outputs = self.model(x)
                    #         loss = criterion(outputs, labels)
                    #         loss.backward()
                    #         optimizer.step()
                    #     print(f'Epoch [{epoch + 1}/6], Loss: {loss.item():.4f}')
                    self.previous_model = deepcopy(self.model)

                    print('再 bias- incremental')
                    # 把val中的0 去掉
                    # val_x_reduce = [x for x, y in zip(val_x, val_y) if y not in status[inc_i][2] and y not in status[inc_i][1]]
                    # val_y_reduce = [y for y in val_y if y not in status[inc_i][2] and y not in status[inc_i][1]]
                    exemplar.update_reduce(self.sample[inc_i], (train_x, train_y), (val_x, val_y), status[inc_i][2])

                    train_xs, train_ys = exemplar.get_exemplar_train()
                    train_xs.extend(train_xs_ft)
                    train_xs.extend(val_x)
                    train_ys.extend(train_ys_ft)
                    train_ys.extend(val_y)
                    train_ys_newlabel,label_mapping = self.automate_label_mapping(train_ys)
                    train_data = DataLoader(BatchflowData(train_xs, train_ys_newlabel),
                                            batch_size=batch_size, shuffle=True, drop_last=True)

                    self.seen_cls = exemplar.get_cur_cls()
                    print("seen cls number : ", self.seen_cls)
                    val_xs, val_ys = exemplar.get_exemplar_val()
                    val_ys_newlabel, label_mapping = self.automate_label_mapping(val_ys)
                    val_bias_data = DataLoader(BatchflowData(val_xs, val_ys_newlabel), batch_size=16, shuffle=True,
                                               drop_last=False)
                    test_acc = []
                    for epoch in range(epoches):
                        print("---" * 50)
                        print("Epoch", epoch)
                        scheduler.step()
                        cur_lr = self.get_lr(optimizer)
                        print("Current Learning Rate : ", cur_lr)
                        self.model.train()
                        for _ in range(len(self.bias_layers)):
                            self.bias_layers[_].eval()
                        '''
                        防止灾难遗忘
                        '''
                        self.stage1_distill_status(train_data, optimizer, inc_i,status,label_mapping)
                    #     偏执层修改
                    for epoch in range(epoches):
                        # bias_scheduler.step()
                        self.model.eval()
                        for _ in range(len(self.bias_layers)):
                            self.bias_layers[_].train()
                        self.stage2_status(val_bias_data, criterion, bias_optimizer,inc_i,status)
                        # if epoch % 50 == 0:
                        #     acc = self.test_data(test_data,label_mapping, inc= inc_i, status = status)
                        #     test_acc.append(acc)

                    self.previous_model = deepcopy(self.model)
                    acc = self.test_data(test_data, label_mapping, inc=inc_i, status=status)
                    test_acc.append(acc)
                    test_accs.append(max(test_acc))
                    print(test_accs)

                elif status[inc_i][0] == 2:
                    exemplar.update(self.sample[inc_i], (train_x, train_y), (val_x, val_y))
                    train_xs, train_ys = exemplar.get_exemplar_train()
                    train_xs.extend(train_x)
                    train_xs.extend(val_x)
                    train_ys.extend(train_y)
                    train_ys.extend(val_y)
                    train_ys_newlabel, label_mapping = self.automate_label_mapping(train_ys)
                    train_data = DataLoader(BatchflowData(train_xs, train_ys_newlabel),
                                            batch_size=batch_size, shuffle=True, drop_last=True)
                    self.seen_cls = exemplar.get_cur_cls()
                    print("seen cls number : ", self.seen_cls)
                    val_xs, val_ys = exemplar.get_exemplar_val()
                    val_ys_newlabel, label_mapping = self.automate_label_mapping(val_ys)
                    val_bias_data = DataLoader(BatchflowData(val_xs, val_ys_newlabel), batch_size=16, shuffle=True,
                                               drop_last=False)
                    test_acc = []
                    for epoch in range(epoches):
                        print("---" * 50)
                        print("Epoch", epoch)
                        scheduler.step()
                        cur_lr = self.get_lr(optimizer)
                        print("Current Learning Rate : ", cur_lr)
                        self.model.train()
                        for _ in range(len(self.bias_layers)):
                            self.bias_layers[_].eval()
                        '''
                        防止灾难遗忘
                        '''
                        self.stage1_distill_status(train_data, optimizer, inc_i, status, label_mapping)
                    #     偏执层修改
                    for epoch in range(epoches):
                        # bias_scheduler.step()
                        self.model.eval()
                        for _ in range(len(self.bias_layers)):
                            self.bias_layers[_].train()
                        self.stage2_status(val_bias_data, criterion, bias_optimizer, inc_i, status)
                        # if epoch % 50 == 0:
                        #     acc = self.test_data(test_data,label_mapping, inc= inc_i, status = status)
                        #     test_acc.append(acc)
                    self.previous_model = deepcopy(self.model)
                    acc =  self.test_data(test_data,label_mapping, inc= inc_i, status = status)
                    test_acc.append(acc)
                    test_accs.append(max(test_acc))
                    print(test_accs)
            else:
            # 进入增量部分
                train_xs, train_ys = exemplar.get_exemplar_train()
                train_xs.extend(train_x)
                train_xs.extend(val_x)
                train_ys.extend(train_y)
                train_ys.extend(val_y)
                train_ys_newlabel, label_mapping = self.automate_label_mapping(train_ys)
                train_data = DataLoader(BatchflowData(train_xs, train_ys_newlabel),
                                        batch_size=batch_size, shuffle=True, drop_last=True)
                exemplar.update(self.sample[inc_i], (train_x, train_y), (val_x, val_y))

                self.seen_cls = exemplar.get_cur_cls()
                print("seen cls number : ", self.seen_cls)
                val_xs, val_ys = exemplar.get_exemplar_val()
                val_bias_data = DataLoader(BatchflowData(val_xs, val_ys), batch_size=16, shuffle=True, drop_last=False)
                test_acc = []

                for epoch in range(epoches):
                    print("---" * 50)
                    print("Epoch", epoch)
                    scheduler.step()
                    cur_lr = self.get_lr(optimizer)
                    print("Current Learning Rate : ", cur_lr)
                    self.model.train()
                    for _ in range(len(self.bias_layers)):
                        self.bias_layers[_].eval()
                    self.stage1_status(train_data, criterion, optimizer,inc_i,status)
                    # acc =  self.test_data(test_data,label_mapping, inc= inc_i, status = status)
                # if inc_i > 0:
                #     """
                #     对偏执层训练，这里可修改
                #     """
                #     for epoch in range(epoches):
                #         # bias_scheduler.step()
                #         self.model.eval()
                #         for _ in range(len(self.bias_layers)):
                #             self.bias_layers[_].train()
                #         self.stage2(val_bias_data, criterion, bias_optimizer)
                #         if epoch % 50 == 0:
                #             acc = self.test(test_data)
                #             test_acc.append(acc)
                for i, layer in enumerate(self.bias_layers):
                    layer.printParam(i)

                # 一次增量结束之后，保存本次模型
                self.previous_model = deepcopy(self.model)
                acc =  self.test_data(test_data,label_mapping, inc= inc_i, status = status)
                test_acc.append(acc)
                test_accs.append(max(test_acc))
                print('test_accs:',test_accs)
                # print('finetune_accs',finetune_accs)

    def train(self, batch_size, epoches, lr, max_size):
        # 对应的批次增加的数量
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
            # 这个是获取上一轮存储的网络流量数据
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
            将增加的类别放进去
            """
            exemplar.update(self.sample[inc_i], (train_x, train_y), (val_x, val_y))
            # self.seen_cls 本轮训练的类别数量
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
                    self.stage1_distill(train_data, optimizer,inc_i)
                else:
                    self.stage1(train_data, criterion, optimizer)
                acc = self.test(test_data)
            if inc_i > 0:
                """
                对偏执层训练，这里可修改
                """
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

    def bias_forward_new(self, input, inc,status):
        # 0 1 2
        if inc > 0:
            if status[inc][2]:
                # 老应用个数
                in1 = input[:,:status[inc][3]-len(status[inc][1])]
                in2 = input[:,status[inc][3]-len(status[inc][1]):status[inc][3]]
                out1 = self.bias_layer1(in1)
                out2 = self.bias_layer2(in2)
                if in1.shape[1] + in2.shape[1] == input.shape[1]:
                    return torch.cat([out1, out2], dim=1), 2
                else:
                    in3 = input[:,status[inc][3]:]
                    out3 = self.bias_layer3(in3)
                    return torch.cat([out1, out2, out3], dim=1),3
            else:
                in1 = input[:, :status[inc][3]]
                in2 = input[:, status[inc][3]:status[inc][3]+len(status[inc][1])]
                out1 = self.bias_layer1(in1)
                out2 = self.bias_layer2(in2)
                if in1.shape[1] + in2.shape[1] == input.shape[1]:
                    return torch.cat([out1, out2], dim=1), 2
                else:
                    in3 = input[:,status[inc][3]+len(status[inc][1]):]
                    out3 = self.bias_layer3(in3)
                    return torch.cat([out1, out2, out3], dim=1), 3
        else:
            in1 = input[:, :self.sample[0]]
            in2 = input[:, self.sample[0]:]
            out1 = self.bias_layer1(in1)
            out2 = self.bias_layer2(in2)
            return torch.cat([out1, out2], dim=1), 2

    def bias_forward(self, input):
        # 这里没有问题，初始模型分类数
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

    def stage1_fine_tune(self, train_data, criterion, optimizer,inc, status,label_mapping):
        print("Training ... ")
        losses_fine_tune = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, inc, status)
            # a = p[:, :self.seen_cls]
            loss = criterion(p[:,:self.seen_cls - len(status[inc][2])], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_fine_tune.append(loss.item())
        print("stage fine_tune loss :", np.mean(losses_fine_tune))

    def stage1_distill_status(self, train_data, optimizer,inc, status,label_mapping):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - self.sample[inc])/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()

            p = self.model(x)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(x)
                pre_p, num_output = self.bias_forward_new(pre_p, inc, status)
                # 老模型中老应用的loss
                pre_p = F.softmax(pre_p[:,:self.seen_cls-self.sample[inc]]/T, dim=1)
            # 新模型中保证老应用的loss
            logp = F.log_softmax(p[:,:self.seen_cls-self.sample[inc]]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))

            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * T * T + (1-alpha) * loss_hard_target
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))

    def stage1_status(self, train_data, criterion, optimizer,inc, status):
        print("Training ... ")
        losses = []
        for i, (x, label) in enumerate(tqdm(train_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, inc, status)
            # a = p[:, :self.seen_cls]
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage2_status_fine_tune(self, val_bias_data, criterion, optimizer, inc, status):
        print("Evaluating ... ")
        losses = []
        for i, (x, label) in enumerate(tqdm(val_bias_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p, num_output = self.bias_forward_new(p, inc, status)
            loss = criterion(p[:, :self.seen_cls - len(status[inc][2])], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))


    def stage2_status(self, val_bias_data, criterion, optimizer, inc, status):
        print("Evaluating ... ")
        losses = []
        for i, (x, label) in enumerate(tqdm(val_bias_data)):
            x = x.type(torch.FloatTensor)
            x = x.cuda()
            label = label.view(-1).cuda()
            p = self.model(x)
            p,num_output = self.bias_forward_new(p,inc, status)
            # print(p[:,:self.seen_cls - len(status[inc][2])])
            # print(label)
            # print(self.seen_cls)
            # print(len(status[inc][2]))
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))

    def automate_label_mapping(self,original_labels):
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

    def stage1_distill(self, train_data, optimizer,inc):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - self.sample[inc])/ self.seen_cls
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
                pre_p = F.softmax(pre_p[:,:self.seen_cls-self.sample[inc]]/T, dim=1)
            # 保证老应用的loss
            logp = F.log_softmax(p[:,:self.seen_cls-self.sample[inc]]/T, dim=1)
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
            # a = p[:, :self.seen_cls]
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