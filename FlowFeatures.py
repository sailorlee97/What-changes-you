"""
@Time    : 2023/9/4 17:16
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: FlowFeatures.py
@Software: PyCharm
"""
#coding:utf-8
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from collections import defaultdict

class flowfeatures():

    def __init__(self):
        self.train_data,self.test_data,self.val_data,self.train_labels,self.test_labels,self.val_labels = self.processdata()
        self.train_groups, self.val_groups, self.test_groups, self.multi_dict = self.evolveinitialize()
        # 持续的次数
        self.batch_num = 4

    def _process_index_label(self,labels):

        le = LabelEncoder()
        labels_en = le.fit_transform(labels).astype(np.int64)
        res = {}
        for cl in le.classes_:
            res.update({cl: le.transform([cl])[0]})
        print(res)
        # logging.info(u'bb:%s' % ('%s' % ss).decode('unicode_escape'))
        # file = open("./log/label.txt", "w")
        # file.write(labels_en)
        # file.close()

    def processdata(self):

        df  = pd.read_csv('./data/newdataframe15.csv')
        scaler = StandardScaler()
        numeric_columns = df.columns[:-1]
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        # dataframe0 = df.drop(df.columns[0], axis=1)
        # train, test = train_test_split(df, test_size=0.1)
        train = df.groupby('appname').head(10000)
        val = df.groupby('appname').head(1000)
        test = df.groupby('appname').tail(1000)
        # train, val = train_test_split(train, test_size=0.1)

        val_labels = val.pop('appname')
        train_labels = train.pop('appname')
        print("The train is :{}".format(list(set(train_labels))))
        test_labels = test.pop('appname')
        print("The test is :{}".format(list(set(test_labels))))
        return train.values, test.values, val.values, train_labels, test_labels, val_labels

    def initialize(self, first_num=4, second_num=7, third_num=10, total_num=13):
        train_groups = [[], [], [], []]
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # print(train_data.shape)
            # train_data_r = train_data[:1024].reshape(32, 32)
            # train_data_g = train_data[1024:2048].reshape(32, 32)
            # train_data_b = train_data[2048:].reshape(32, 32)
            # train_data = np.dstack((train_data_r, train_data_g, train_data_b))
            if train_label < first_num:
                train_groups[0].append((train_data, train_label))
            elif first_num <= train_label < second_num:
                train_groups[1].append((train_data, train_label))
            elif second_num <= train_label < third_num:
                train_groups[2].append((train_data, train_label))
            elif third_num <= train_label < total_num:
                train_groups[3].append((train_data, train_label))
            # elif 80 <= train_label < 100:
            #     train_groups[4].append((train_data,train_label))
        # assert len(train_groups[0]) == 10000, len(train_groups[0])
        # assert len(train_groups[1]) == 10000, len(train_groups[1])
        # assert len(train_groups[2]) == 10000, len(train_groups[2])
        # assert len(train_groups[3]) == 10000, len(train_groups[3])
        # assert len(train_groups[4]) == 10000, len(train_groups[4])

        val_groups = [[], [], [], []]
        for val_data, val_label in zip(self.val_data, self.val_labels):
            # print(train_data.shape)
            # train_data_r = train_data[:1024].reshape(32, 32)
            # train_data_g = train_data[1024:2048].reshape(32, 32)
            # train_data_b = train_data[2048:].reshape(32, 32)
            # train_data = np.dstack((train_data_r, train_data_g, train_data_b))
            if val_label < first_num:
                val_groups[0].append((val_data, val_label))
            elif first_num <= val_label < second_num:
                val_groups[1].append((val_data, val_label))
            elif second_num <= val_label < third_num:
                val_groups[2].append((val_data, val_label))
            elif third_num <= val_label < total_num:
                val_groups[3].append((val_data, val_label))
        # for i, train_group in enumerate(train_groups):
        #
        #     val_groups[i] = train_groups[i][80000:]
        #     train_groups[i] = train_groups[i][:80000]
        # assert len(train_groups[0]) == 9000
        # assert len(train_groups[1]) == 9000
        # assert len(train_groups[2]) == 9000
        # assert len(train_groups[3]) == 9000
        # assert len(train_groups[4]) == 9000
        # assert len(val_groups[0]) == 1000
        # assert len(val_groups[1]) == 1000
        # assert len(val_groups[2]) == 1000
        # assert len(val_groups[3]) == 1000
        # assert len(val_groups[4]) == 1000

        test_groups = [[], [], [], [], []]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            # test_data_r = test_data[:1024].reshape(32, 32)
            # test_data_g = test_data[1024:2048].reshape(32, 32)
            # test_data_b = test_data[2048:].reshape(32, 32)
            # test_data = np.dstack((test_data_r, test_data_g, test_data_b))
            if test_label < first_num:
                test_groups[0].append((test_data, test_label))
            elif first_num <= test_label < second_num:
                test_groups[1].append((test_data, test_label))
            elif second_num <= test_label < third_num:
                test_groups[2].append((test_data, test_label))
            elif third_num <= test_label < total_num:
                test_groups[3].append((test_data, test_label))
            # elif 80 <= test_label < 100:
            #     test_groups[4].append((test_data,test_label))
        # assert len(test_groups[0]) == 2000
        # assert len(test_groups[1]) == 2000
        # assert len(test_groups[2]) == 2000
        # assert len(test_groups[3]) == 2000
        # assert len(test_groups[4]) == 2000
        return train_groups, val_groups, test_groups

    def newinitialize(self, first_num=None, second_num = None, third_num = None, total_num = None):
        if first_num is None:
            first_num = {0, 1, 2, 3, 4, 5}
        if second_num is None:
            second_num = {0, 1, 2, 3, 4, 5, 6, 7, 8}
        if third_num is None:
            third_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if total_num is None:
            total_num = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        train_groups = [[],[],[],[]]
        for train_data, train_label in zip(self.train_data, self.train_labels):

            if train_label in list(first_num):
                train_groups[0].append((train_data,train_label))
            if train_label in list(second_num - first_num):
                train_groups[1].append((train_data,train_label))
            if train_label in list(third_num - second_num):
                train_groups[2].append((train_data,train_label))
            if train_label in list(total_num - third_num):
                train_groups[3].append((train_data,train_label))

        val_groups = [[],[],[],[]]
        for val_data, val_label in zip(self.val_data, self.val_labels):

            if val_label in list(first_num):
                val_groups[0].append((val_data,val_label))
            if val_label in list(second_num - first_num):
                val_groups[1].append((val_data,val_label))
            if val_label in list(third_num - second_num):
                val_groups[2].append((val_data,val_label))
            if val_label in list(total_num - third_num):
                val_groups[3].append((val_data,val_label))

        test_groups = [[],[],[],[],[]]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            if test_label in list(first_num):
                test_groups[0].append((test_data,test_label))
            if test_label in list(second_num - first_num):
                test_groups[1].append((test_data,test_label))
            if test_label in list(third_num - second_num):
                test_groups[2].append((test_data,test_label))
            if test_label in list(total_num - third_num):
                test_groups[3].append((test_data,test_label))

        return train_groups, val_groups, test_groups

    def get_status(self, set1, set2):
        '''

        :param set1: 上一次的应用列表  app list in pre set
        :param set2:  app list in set
        :return:
        '''

        incremental_elements = set2 - set1
        # 查看有没有减少的元素
        Reduce_elements = set1 - set2
        if len(Reduce_elements) > 0 and len(incremental_elements) == 0:
            status = 0
            # print('fine-tune')

        elif len(Reduce_elements) > 0 and len(incremental_elements) > 0:
            status = 1
            # print('先 fine-tune')
            # print('再 bias- incremental')

        elif len(incremental_elements) > 0 and len(Reduce_elements) == 0:
            status = 2
            # print('bias incremental')
        else:
            status = 3
            # print('no change!')

        return status, incremental_elements, Reduce_elements

    def evolveinitialize(self,first_num=None, second_num = None, third_num = None, forth_num = None):
        if first_num is None:
            first_num = {0, 1, 2, 3}
        if second_num is None:
            second_num = {0, 1, 2, 3, 4}
        if third_num is None:
            third_num = {1, 2, 3, 4, 5, 6 }
        if forth_num is None:
            forth_num = {1, 2, 3, 4, 5, 6, 7, 8, 9}

        a = [second_num,third_num,forth_num]
        """
        先判断状态
        """
        multi_dict = defaultdict(list)
        lista = [first_num,second_num,third_num, forth_num]
        for i in range(len(lista)-1):
            status, incremental_elements, Reduce_elements = self.get_status(lista[i],lista[i+1])
            multi_dict[i+1] .extend([status, incremental_elements, Reduce_elements,len(a[i])])
            # statuslist.append(status)
            # incremental_sets.append(incremental_elements)
            # Reduce_sets.append(Reduce_elements)

        train_groups = [[] for _ in range(len(lista))]
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # 第一次需要训练的元素放入
            if train_label in list(first_num):
                train_groups[0].append((train_data, train_label))
            for key, values in multi_dict.items():
                # 依次把标签为4 为5 6 为 7 8 9的放进label中
                if values[1] and train_label in list(values[1]):
                    train_groups[key].append((train_data, train_label))

        val_groups = [[] for _ in range(len(lista))]
        for val_data, val_label in zip(self.val_data, self.val_labels):
            if val_label in list(first_num):
                val_groups[0].append((val_data, val_label))
            for key, values in multi_dict.items():
                # 依次把标签为4 为5 6 为 7 8 9的放进label中
                if values[1] and val_label in list(values[1]):
                    val_groups[key].append((val_data, val_label))

        test_groups = [[] for _ in range(len(lista))]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            if test_label in list(first_num):
                test_groups[0].append((test_data, test_label))
            for key, values in multi_dict.items():
                # 依次把标签为4 为5 6 为 7 8 9的放进label中
                if values[1] and test_label in list(values[1]):
                    test_groups[key].append((test_data, test_label))

        return train_groups, val_groups, test_groups, multi_dict


    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

def process_csv():

    df = pd.read_csv('./data/dataframe15.csv')
    # class_counts = df['appname'].value_counts()
    del_list = ['QQ音乐','爱奇艺','百度贴吧','金铲铲之战']
    for i in del_list:
        df = df[df['appname'] != i]

    # class_counts = df['appname'].value_counts()
    # print(class_counts)
    df.drop(df.columns[[0]], axis=1, inplace=True)

    # propress labels
    labels = df.pop('appname')
    le = preprocessing.LabelEncoder()
    numlabel = le.fit_transform(labels)
    df['appname'] = numlabel
    res = {}
    for cl in le.classes_:
        res.update({cl: le.transform([cl])[0]})
    print(res)
    WorldNet = open("./log/label.txt", "w",encoding="utf-8")
    WorldNet.write(str(res))
    WorldNet.close()
    result_df = df.groupby('appname').head(11000)
    
    result_df.to_csv('./data/newdataframe15.csv',index=False)



if __name__ == '__main__':

    # process_csv()
    # df  = pd.read_csv('./data/dataframe24.csv')
    # class_counts = df['appname'].value_counts()
    # print(class_counts)
    # ff = flowfeatures()
    # train_groups, val_groups, test_groups, multi_dict = ff.evolveinitialize()
    # print(multi_dict)
    # train_groups, val_groups, test_groups = ff.newinitialize()
    # train_groups, val_groups, test_groups = ff.getNextClasses(2)
    # print(len(train_groups))
    # dataframe = df.copy()
    # labels = dataframe.pop('appname')
    # dataframe.drop(dataframe.columns[[0]], axis=1, inplace=True)
    # # propress labels
    # le = preprocessing.LabelEncoder()
    # numlabel = le.fit_transform(labels)
    # dataframe['appname'] = numlabel
    #
    # result_df = dataframe.groupby('appname').head(10000)
    # result_df.to_csv('./data/newdataframe40.csv',index=False)
    # print(numlabel)
    process_csv()