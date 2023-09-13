"""
@Time    : 2023/9/4 17:16
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : sailorlee31@gmail.com
-------------------------------------------------
@FileName: FlowFeatures.py
@Software: PyCharm
"""
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

class flowfeatures():

    def __init__(self):
        self.train_data,self.test_data,self.val_data,self.train_labels,self.test_labels,self.val_labels = self.processdata()
        self.train_groups, self.val_groups, self.test_groups = self.initialize()
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

    def _propress_label(self, labels, sorted_labels):
        """Encode target labels with value between 0 and n_classes-1.

        This transformer should be used to encode target values, *i.e.* `y`, and
        not the input `X`.

        :param data:
        :return: numic
        """
        # le = LabelEncoder()
        # newlabels = le.fit_transform(labels)
        #
        # res = {}
        # for cl in le.classes_:
        #     res.update({cl: le.transform([cl])[0]})
        # print(res)
        reskey = {}
        for i in sorted_labels:
            reskey.update({i: sorted_labels.index(i)})
        print(reskey)
        # map映射
        labels = labels.map(reskey).values

        print(labels)
        return labels, reskey

    def processdata(self):

        df  = pd.read_csv('./data/newdataframe13.csv')
        scaler = StandardScaler()
        numeric_columns = df.columns[:-1]
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        # dataframe0 = df.drop(df.columns[0], axis=1)
        # train, test = train_test_split(df, test_size=0.1)
        train = df.groupby('appname').head(6000)
        val = df.groupby('appname').head(1000)
        test = df.groupby('appname').tail(1000)
        # train, val = train_test_split(train, test_size=0.1)
        val_labels = val.pop('appname')
        train_labels = train.pop('appname')
        test_labels = test.pop('appname')
        return train.values, test.values, val.values, train_labels, test_labels, val_labels

    def initialize(self, first_num= 4, second_num = 7, third_num = 10, total_num = 13):
        train_groups = [[],[],[],[]]
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # print(train_data.shape)
            # train_data_r = train_data[:1024].reshape(32, 32)
            # train_data_g = train_data[1024:2048].reshape(32, 32)
            # train_data_b = train_data[2048:].reshape(32, 32)
            # train_data = np.dstack((train_data_r, train_data_g, train_data_b))
            if train_label < first_num:
                train_groups[0].append((train_data,train_label))
            elif first_num <= train_label < second_num:
                train_groups[1].append((train_data,train_label))
            elif second_num <= train_label < third_num:
                train_groups[2].append((train_data,train_label))
            elif third_num <= train_label < total_num:
                train_groups[3].append((train_data,train_label))
            # elif 80 <= train_label < 100:
            #     train_groups[4].append((train_data,train_label))
        # assert len(train_groups[0]) == 10000, len(train_groups[0])
        # assert len(train_groups[1]) == 10000, len(train_groups[1])
        # assert len(train_groups[2]) == 10000, len(train_groups[2])
        # assert len(train_groups[3]) == 10000, len(train_groups[3])
        # assert len(train_groups[4]) == 10000, len(train_groups[4])

        val_groups = [[],[],[],[]]
        for val_data, val_label in zip(self.val_data, self.val_labels):
            # print(train_data.shape)
            # train_data_r = train_data[:1024].reshape(32, 32)
            # train_data_g = train_data[1024:2048].reshape(32, 32)
            # train_data_b = train_data[2048:].reshape(32, 32)
            # train_data = np.dstack((train_data_r, train_data_g, train_data_b))
            if val_label < first_num:
                val_groups[0].append((val_data,val_label))
            elif first_num <= val_label < second_num:
                val_groups[1].append((val_data,val_label))
            elif second_num <= val_label < third_num:
                val_groups[2].append((val_data,val_label))
            elif third_num <= val_label < total_num:
                val_groups[3].append((val_data,val_label))
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

        test_groups = [[],[],[],[],[]]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            # test_data_r = test_data[:1024].reshape(32, 32)
            # test_data_g = test_data[1024:2048].reshape(32, 32)
            # test_data_b = test_data[2048:].reshape(32, 32)
            # test_data = np.dstack((test_data_r, test_data_g, test_data_b))
            if test_label < first_num:
                test_groups[0].append((test_data,test_label))
            elif first_num <= test_label < second_num:
                test_groups[1].append((test_data,test_label))
            elif second_num <= test_label < third_num:
                test_groups[2].append((test_data,test_label))
            elif third_num <= test_label < total_num:
                test_groups[3].append((test_data,test_label))
            # elif 80 <= test_label < 100:
            #     test_groups[4].append((test_data,test_label))
        # assert len(test_groups[0]) == 2000
        # assert len(test_groups[1]) == 2000
        # assert len(test_groups[2]) == 2000
        # assert len(test_groups[3]) == 2000
        # assert len(test_groups[4]) == 2000
        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

def process_csv():

    df = pd.read_csv('./data/dataframe14.csv')
    # class_counts = df['appname'].value_counts()
    df = df[df['appname'] != '蛋仔派对']
    # class_counts = df['appname'].value_counts()
    df.drop(df.columns[[0]], axis=1, inplace=True)
    # propress labels
    labels = df.pop('appname')
    le = preprocessing.LabelEncoder()
    numlabel = le.fit_transform(labels)
    df['appname'] = numlabel

    result_df = df.groupby('appname').head(7000)
    result_df.to_csv('./data/newdataframe13.csv',index=False)



if __name__ == '__main__':

    # process_csv()
    df  = pd.read_csv('./data/dataframe24.csv')
    class_counts = df['appname'].value_counts()
    # print(class_counts)
    # ff = flowfeatures()
    # train_groups, val_groups, test_groups = ff.initialize()
    # train_groups, val_groups, test_groups = ff.getNextClasses(0)
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