import os

import sklearn.utils
import torch.utils.data
import torch
import torchvision
from sklearn.model_selection import train_test_split

from module.data_manager.manager import DataManager


# https://blog.csdn.net/sxf1061700625/article/details/105870851?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166763065216800180665746%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166763065216800180665746&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105870851-null-null.142^v63^pc_rank_34_queryrelevant25,201^v3^control_2,213^v1^control&utm_term=mnist&spm=1018.2226.3001.4187

class GraphicalClassificationManager(DataManager):
    pass

class MNIST(GraphicalClassificationManager):
    def __init__(self):
        super().__init__()
        return

    def read(self, test_ratio, shuffle_seed, nrows=None):
        torch.manual_seed(shuffle_seed)
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_dataset = torchvision.datasets.MNIST(os.path.join(project_path, 'data/'),
                                                   train=True, download=True,
                                                   transform=torchvision.transforms.Compose([
                                                       torchvision.transforms.ToTensor(),
                                                       torchvision.transforms.Normalize(
                                                           (0.1307,), (0.3081,))
                                                   ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=len(train_dataset), shuffle=True)
        test_dataset = torchvision.datasets.MNIST(os.path.join(project_path, 'data/'),
                                                  train=False, download=True,
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))
                                                  ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=True)
        X_train, y_train = train_loader.__iter__().next()
        X_test, y_test = test_loader.__iter__().next()
        X, y = torch.concat([X_train, X_test]), torch.concat([y_train, y_test])
        sklearn.utils.shuffle(X, random_state=shuffle_seed)
        sklearn.utils.shuffle(y, random_state=shuffle_seed)

        if nrows is not None:
            X, y = X[:nrows], y[:nrows]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=shuffle_seed)

        # self.X_train = self.X_train.reshape(len(self.X_train), 28 * 28)
        # self.X_test = self.X_test.reshape(len(self.X_test), 28 * 28)
        # self._data_to_cuda()

        return




# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
class CIFAR10(GraphicalClassificationManager):
    def __init__(self):
        super().__init__()
        return

    def read(self, test_ratio, shuffle_seed, nrows=None):
        torch.manual_seed(shuffle_seed)
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, 'data/'),
                                                     train=True, download=True,
                                                     transform=torchvision.transforms.Compose([
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(
                                                             (0.48836562, 0.48134598, 0.4451678),
                                                             (0.24833508, 0.24547848, 0.26617324))
                                                     ]))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=len(train_dataset), shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(os.path.join(project_path, 'data/'),
                                                    train=False, download=True,
                                                    transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(
                                                            (0.47375134, 0.47303376, 0.42989072),
                                                            (0.25467148, 0.25240466, 0.26900575))
                                                    ]))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=len(test_dataset), shuffle=True)

        self.X_train, self.y_train = train_loader.__iter__().next()
        self.X_test, self.y_test = test_loader.__iter__().next()
