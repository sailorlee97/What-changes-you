class Exemplar:
    def __init__(self, max_size, total_cls):
        self.val = {}
        self.train = {}
        self.cur_cls = 0
        self.max_size = max_size
        self.total_classes = total_cls

    def update(self, cls_num, train, val):
        train_x, train_y = train
        val_x, val_y = val
        assert self.cur_cls == len(list(self.val.keys()))
        assert self.cur_cls == len(list(self.train.keys()))
        # cur_keys = list(set(val_y))
        self.cur_cls += cls_num
        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else self.max_size
        train_store_num = int(total_store_num * 0.9)
        val_store_num = int(total_store_num * 0.1)
        # 每个类取前50
        """
        对字典self.val中的每一个值，保留前val_store_num个元素，并将其更新‘上一批’数据需要保留的部分。
        """
        for key, value in self.val.items():
            self.val[key] = value[:val_store_num]
        # 每个类取前450
        for key, value in self.train.items():
            self.train[key] = value[:train_store_num]

        for x, y in zip(val_x, val_y):
            # 遇到新的类别就放进去
            if y not in self.val:
                self.val[y] = [x]
            else:
                # ·没有补到val_store_num就继续补充
                if len(self.val[y]) < val_store_num:
                    self.val[y].append(x)

        # print(len(list(self.val.keys())))
        # 查看验证集类别的个数和验证集里面的类别数是不是一致
        assert self.cur_cls == len(list(self.val.keys()))
        for key, value in self.val.items():
            assert len(self.val[key]) == val_store_num
        """
        这里是把新增的类别和之前老的类别一起合并 所有的数据都需要保存
        将训练数据按照对应的标签进行分类存储在 self.train 字典中，同时控制每个类别所保存的数据数量不超过train_store_num
        """
        for x, y in zip(train_x, train_y):
            if y not in self.train:
                self.train[y] = [x]
            else:
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)
        assert self.cur_cls == len(list(self.train.keys()))
        for key, value in self.train.items():
            assert len(self.train[key]) == train_store_num

    def update_reduce(self, cls_num, train, val, setl):
        train_x, train_y = train
        val_x, val_y = val
        assert self.cur_cls == len(list(self.val.keys()))
        assert self.cur_cls == len(list(self.train.keys()))
        # cur_keys = list(set(val_y))
        self.cur_cls += cls_num
        self.cur_cls = self.cur_cls - len(setl)
        total_store_num = self.max_size / self.cur_cls if self.cur_cls != 0 else self.max_size
        train_store_num = int(total_store_num * 0.9)
        val_store_num = int(total_store_num * 0.1)
        # 每个类取前50
        """
        即上一次的前 50%
        对字典self.val中的每一个值，保留前val_store_num个元素，并将其更新‘上一批’数据需要保留的部分。
        """
        for key, value in self.val.items():
            self.val[key] = value[:val_store_num]
        # 每个类取前450
        for key, value in self.train.items():
            self.train[key] = value[:train_store_num]

        #  train 里删除set里面的元素
        self.train = {k: v for k, v in self.train.items() if k not in setl}
        self.val = {k: v for k, v in self.val.items() if k not in setl}


        for x, y in zip(val_x, val_y):
            # 遇到新的类别就放进去
            if y not in self.val:
                self.val[y] = [x]
            else:
                # ·没有补到val_store_num就继续补充
                if len(self.val[y]) < val_store_num:
                    self.val[y].append(x)

        print(len(list(self.val.keys())))
        # 查看验证集类别的个数和验证集里面的类别数是不是一致
        assert self.cur_cls == len(list(self.val.keys()))
        for key, value in self.val.items():
            assert len(self.val[key]) == val_store_num
        """
        这里是把新增的类别和之前老的类别一起合并 所有的数据都需要保存
        将训练数据按照对应的标签进行分类存储在 self.train 字典中，同时控制每个类别所保存的数据数量不超过train_store_num
        """
        for x, y in zip(train_x, train_y):
            if y not in self.train:
                self.train[y] = [x]
            else:
                if len(self.train[y]) < train_store_num:
                    self.train[y].append(x)
        assert self.cur_cls == len(list(self.train.keys()))
        for key, value in self.train.items():
            assert len(self.train[key]) == train_store_num





    def get_exemplar_train(self):
        exemplar_train_x = []
        exemplar_train_y = []
        for key, value in self.train.items():
            for train_x in value:
                exemplar_train_x.append(train_x)
                exemplar_train_y.append(key)
        return exemplar_train_x, exemplar_train_y

    def get_exemplar_val(self):
        exemplar_val_x = []
        exemplar_val_y = []
        for key, value in self.val.items():
            for val_x in value:
                exemplar_val_x.append(val_x)
                exemplar_val_y.append(key)
        return exemplar_val_x, exemplar_val_y

    def get_cur_cls(self):
        return self.cur_cls
