import copy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from model.net import *
# from xgboost import XGBClassifier,XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
# import torchvision.ops.
# from model.net import CNNMNIST, CNNCIFAR10
from sklearn.metrics import f1_score
import numpy as np


def return_model(mode, seed, **kwargs):
    if mode == 'SVC':
        model = SVC(gamma=0.001)
    # elif mode == 'Tree':
    #     model = DecisionTreeClassifier(random_state=666)
    # elif mode == 'MLPMNIST':
    #     ic = kwargs.get('input_channels', 784)
    #     oc = kwargs.get('output_channels', 10)
    #     hls = kwargs.get('hidden_layer_size', 128)
    #     tol = kwargs.get('tol', 5e-3)
    #     lr = kwargs.get('lr', 0.005)
    #     elst = kwargs.get('elst', True)
    #     max_iter = kwargs.get('max_iter', 1000)
    #     model = MLP(ic, oc, hidden_layer_size=hls, tol=tol, lr=lr, elst=elst, max_iter=max_iter)
    #     # model = MLPClassifier(max_iter=10000)
    elif mode == 'CNNMNIST':
        model = CNNMNIST(seed=seed)
    # elif mode == 'XGBClassifier':
    #     model = XGBClassifier(random_state=666)
    # elif mode == 'XGBRegressor':
    #     model = XGBRegressor(random_state=666)
    elif mode == 'AdultMLP':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = AdultMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                         batch_size=batch_size)
    elif mode == 'BankMLP':
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = BankMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                        batch_size=batch_size)
    elif mode == "Dota2MLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = Dota2MLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                         batch_size=batch_size)
    elif mode == "TicTacToeMLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = TicTacToeMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                             batch_size=batch_size)
    elif mode == "CreditCardMLP":
        lr = kwargs.get("lr")
        num_epoch = kwargs.get("num_epoch")
        device = kwargs.get("device")
        hidden_layer_size = kwargs.get("hidden_layer_size")
        batch_size = kwargs.get("batch_size")
        model = CreditCardMLP(seed=seed, lr=lr, num_epoch=num_epoch, device=device, hidden_layer_size=hidden_layer_size,
                             batch_size=batch_size)
    elif mode == 'logistic regression':
        tol = kwargs.get('tol', 1e-3)
        # scoring = kwargs.get('scoring')
        model = LogisticRegression(max_iter=50000, tol=tol, solver='liblinear', multi_class="auto", C=1.0, penalty="l2")
    elif mode == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif mode == 'linear regression':
        model = LinearRegression()
    elif mode == "sgd cls log":
        model = SGDClassifier(loss="log_loss", random_state=666)
    elif mode == "sgd reg sqr l2":
        model = SGDRegressor(loss="squared_loss", penalty="l2", random_state=666)
    # elif mode == 'CNNMNIST':
    #     model = CNNMNIST()
    # elif mode == 'CNNCIFAR10':
    #     model = CNNCIFAR10()
    else:
        model = None
    return model
