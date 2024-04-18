import copy
import os.path

import numpy as np

import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from torch.utils.data import Dataset
from sklearn import preprocessing
from scipy.stats import dirichlet
from abc import ABC, abstractmethod


# 是否需要标注是师兄的方法

class DataManager(ABC):
    def __init__(self, name=None):
        self.task = None
        self.name = name
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_parts = None
        self.y_train_parts = None

    @abstractmethod
    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        pass

    def __str__(self):
        return self.name

    def train_test_split(self, test_ratio, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_ratio, random_state=random_state)
        return

    # 按比例分配参与方，这个写得好
    def ratio_split(self, ratios):
        # assert sum(ratios) == 1
        lo_ratio = 0
        n = len(self.X_train)
        self.X_train_parts = []
        self.y_train_parts = []
        for ratio in ratios:
            lo, hi = int(lo_ratio * n), int((lo_ratio + ratio) * n)
            self.X_train_parts.append(self.X_train[lo:hi])
            self.y_train_parts.append(self.y_train[lo:hi])
            lo_ratio += ratio
        # print([len(self.X_train_parts[i]) for i in range(8)])
        return self.X_train_parts, self.y_train_parts

    # 均分
    def uniform_split(self, num_parts):
        # 不妨化归为ratio_split问题
        ratios = []
        for i in range(num_parts):
            ratios.append((i + 1) / float(num_parts) - i / float(num_parts))
        return self.ratio_split(ratios)

    # 只shuffle y
    def low_quality_data(self, parts:set, random_seed, ratio):
        np.random.seed(random_seed)
        for client in parts:
            y_list = np.random.choice(range(len(self.y_train_parts[client])), round(ratio*len(self.y_train_parts[client])), replace=False)
            y_shuffle_list = copy.deepcopy(y_list)
            np.random.shuffle(y_shuffle_list)
            self.y_train_parts[client][y_list] = self.y_train_parts[client][y_shuffle_list]
        return

    # 恶意参与方：数据复制
    def data_copy(self, clients: set, seed, ratio):
        np.random.seed(seed)
        for client in clients:
            n = len(self.y_train_parts[client])
            replicate_indices = np.random.choice(list(range(n)), round(ratio * n)).tolist()
            self.X_train_parts[client] = torch.cat([self.X_train_parts[client], self.X_train_parts[client][replicate_indices]])
            self.y_train_parts[client] = torch.cat([self.y_train_parts[client], self.y_train_parts[client][replicate_indices]])
        return

    def _data_to_cuda(self):
        self.X_train = self.X_train.cuda()
        self.y_train = self.y_train.cuda()
        self.X_test = self.X_test.cuda()
        self.y_test = self.y_test.cuda()
        return


def read_classification_dataset():
    imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
    data_manager = torch.utils.data.DataLoader(imagenet_data,
                                                batch_size=4,
                                                shuffle=False)
# def
# print(
#     f'Training data size is {len(self.data_tr[0])} with avg. positive rate {self.data_tr[1].float().mean():.2f}, '
#     f'test data size is {len(self.data_te[0])} with avg. positive rate {self.data_te[1].float().mean():.2f} ')
