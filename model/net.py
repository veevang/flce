import copy

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from typing import Union
import torch.utils.data
from sklearn.metrics import accuracy_score, f1_score


# https://blog.csdn.net/sxf1061700625/article/details/105870851?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166763065216800180665746%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166763065216800180665746&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105870851-null-null.142^v63^pc_rank_34_queryrelevant25,201^v3^control_2,213^v1^control&utm_term=mnist&spm=1018.2226.3001.4187

# ref pytorch
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

class Net(nn.Module):
    def __init__(self, seed):
        self.seed = seed
        self.incremental_seed = seed
        torch.manual_seed(self.seed)
        super().__init__()
        return

    # 要保证每次fit前初始化一次
    def _fit(self,
             X_train,
             y_train,
             num_epochs,
             loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()],
             lr,
             incremental: bool,
             batch_size
             ):

        # 初始化一遍，防止上次的训练对这次产生影响
        if not incremental:
            self.load_state_dict(self.initial_state_dict)
            self.incremental_seed = self.seed
            torch.manual_seed(self.seed)
        else:
            self.incremental_seed += 17
            torch.manual_seed(self.incremental_seed)

        loss_fun = loss_fun.to(self.device)
        # print(f"lenX = {len(X_train)}")
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        # first, process data. put into dataloader.
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 进入训练模式
        self.train(True)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            running_loss = 0.0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.unsqueeze(1).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self(inputs)

                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # print(running_loss)
        # 退出训练模式
        self.train(False)
        return

    def _fit_and_score(self, X_train, y_train, X_test, y_test, value_functions, num_epochs,
                       # incremental=False,
                       loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()],
                       lr,
                       test_interval,
                       batch_size
                       ):
        torch.manual_seed(self.seed)

        loss_fun = loss_fun.to(self.device)

        # print(f"lenX = {len(X_train)}")
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        # first, process data. put into dataloader.
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 初始化一遍，防止上次的训练对这次产生影响
        self.load_state_dict(self.initial_state_dict)

        # 进入训练模式
        optimizer = optim.Adam(self.parameters(), lr=lr)

        val_list = np.zeros(len(value_functions))

        for epoch in range(num_epochs):
            self.train(True)
            running_loss = 0.0

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                labels = labels.unsqueeze(1).float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self(inputs)

                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # validate
            if epoch % test_interval == 0 and epoch != 0:
                y_pred = self._predict(X_test, batch_size)
                for idx_val, temp_value_function in enumerate(value_functions):
                    if temp_value_function == "accuracy":
                        val_list[idx_val] = max(val_list[idx_val], accuracy_score(y_true=y_test, y_pred=y_pred))
                    elif temp_value_function == "f1":
                        val_list[idx_val] = max(val_list[idx_val], f1_score(y_true=y_test, y_pred=y_pred))
                    elif temp_value_function == "f1_macro":
                        val_list[idx_val] = max(val_list[idx_val],
                                                f1_score(y_true=y_test, y_pred=y_pred, average="macro"))
                    elif temp_value_function == "f1_micro":
                        val_list[idx_val] = max(val_list[idx_val],
                                                f1_score(y_true=y_test, y_pred=y_pred, average="micro"))

            # print(running_loss)
        # 退出训练模式
        self.train(False)
        return val_list

    def _predict(self,
                 X_test: torch.tensor,
                 batch_size,
                 ):
        torch.manual_seed(self.seed)
        self.eval()
        predicted_labels = []

        test_dataset = torch.utils.data.TensorDataset(X_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Iterate over the test dataset
        for data in test_dataloader:
            # print(type(data))
            (inputs,) = data
            inputs = inputs.to(self.device)

            # Forward pass through the model
            outputs = self(inputs)
            outputs = outputs.to("cpu")

            # Get the predicted labels
            predicted = (outputs >= 0.5).squeeze().long()
            predicted_labels.extend(predicted.tolist())

        # Convert the lists to NumPy arrays
        predicted_labels = np.array(predicted_labels)

        return predicted_labels


class CNNMNIST(Net):
    def __init__(self, seed, device):
        super().__init__(seed=seed, device=device)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(7 * 7 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.initial_state_dict = copy.deepcopy(self.state_dict())
        return

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 32)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    # 假设训练两个epoch？
    def fit(self, X_train, y_train):
        return self._fit(X_train, y_train, num_epochs=10, loss_fun=nn.CrossEntropyLoss(), lr=0.01)

    def predict(self, X_test):
        return self._predict(X_test)

    def _predict(self,
                 X_test: torch.tensor,
                 # loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss()] = nn.BCEWithLogitsLoss(),
                 # test_losses=None
                 ):
        torch.manual_seed(self.seed)
        self.eval()
        predicted_labels = []

        test_dataset = torch.utils.data.TensorDataset(X_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Iterate over the test dataset
        for data in test_dataloader:
            inputs = data

            # Forward pass through the model
            outputs = self(inputs)

            # Get the predicted labels
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.tolist())

        # Convert the lists to NumPy arrays
        predicted_labels = np.array(predicted_labels)

        return predicted_labels


class AdultMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(AdultMLP, self).__init__(seed)
        self.name = "AdultMLP"
        input_size = 105
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class BankMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(BankMLP, self).__init__(seed)
        self.name = "BankMLP"
        input_size = 51
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.9082544457223746
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class TicTacToeMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(TicTacToeMLP, self).__init__(seed)
        self.name = "TicTacToeMLP"
        input_size = 27
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class Dota2MLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(Dota2MLP, self).__init__(seed)
        self.name = "Dota2MLP"
        input_size = 172
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)


class CreditCardMLP(Net):
    def __init__(self, seed, lr, num_epoch, hidden_layer_size, device, batch_size):
        super(CreditCardMLP, self).__init__(seed)
        self.name = "CreditCardMLP"
        input_size = 29
        output_size = 1
        self.lr = lr
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.initial_state_dict = copy.deepcopy(self.state_dict())

        self.to(device)
        self.device = device

    def forward(self, x: torch.Tensor):
        if type(x) == list:
            x = x[0]
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
        return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
                                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)

    # num epoch:40, lr: 0.001, accu:0.854793793926535
    def fit(self, X_train, y_train, incremental=False, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epoch
        return self._fit(X_train, y_train, num_epochs=num_epochs, loss_fun=nn.BCELoss(), lr=self.lr,
                         incremental=incremental, batch_size=self.batch_size)

    def predict(self, X_test):
        return self._predict(X_test, batch_size=self.batch_size)

# torch.org
# class CNNCIFAR10(Net):
#     def __init__(self):
#         super().__init__()
#         self.seq = nn.Sequential(
#             nn.Conv2d(3, 6, (5, 5)),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(6, 16, (5, 5)),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Flatten(1),  # flatten all dimensions except batch
#             nn.Linear(16 * 5 * 5, 120),
#             nn.ReLU(),
#             nn.Linear(120, 84),
#             nn.ReLU(),
#             nn.Linear(84, 10)
#         )
#
#     def forward(self, x):
#         out = self.seq(x)
#         return out
#
#     # 假设训练两个epoch？
#     def fit(self, X_train, y_train):
#         return self._fit(X_train, y_train, num_epochs=5, incremental=False, loss_fun=F.cross_entropy, lr=0.001,
#                          momentum=0.9)
#
#     def score(self, X_test, y_test):
#         return self._score(X_test, y_test, loss_fun=F.cross_entropy)


# class MLP(Net):
#     def __init__(self, input_channels=784, output_channels=10, hidden_layer_size=128, tol=1e-4, lr=0.001, elst=False,
#                  max_iter=50):
#         super(MLP, self).__init__()
#         self.linear_relu_stack = nn.Sequential(nn.Linear(int(input_channels), int(hidden_layer_size)),
#                                                nn.ReLU(),
#                                                nn.Linear(hidden_layer_size, hidden_layer_size),
#                                                nn.ReLU(),
#                                                nn.Linear(hidden_layer_size, output_channels)
#                                                )
#         self.initial_state_dict = copy.deepcopy(self.state_dict())
#         self.tol = tol
#         self.lr = lr
#         self.elst = elst
#         self.max_iter = max_iter
#         return
#
#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits
#
#     def fit(self, X_train, y_train):
#         return self._fit(X_train, y_train, max_iters=self.max_iter, incremental=False, lr=self.lr,
#                          loss_fun=nn.CrossEntropyLoss(), tol=self.tol)
#
#     # def score(self, X_test, y_test):
#     #     return self._score(X_test, y_test, loss_fun=nn.BCEWithLogitsLoss())
#
#     def predict(self, X):
#         with torch.no_grad():
#             predicted = []
#             for image in X:
#                 image = image.reshape(1, 28 * 28)
#                 output = self(image)
#                 _, this_predicted = torch.max(output.data, 1)
#                 # print(this_predicted)
#                 predicted.append(this_predicted)
#             predicted = torch.tensor(predicted)
#             return predicted


# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# early stopper只根据训练的train_loss是否改变来改变
class EarlyStopper:
    def __init__(self, n_iter_no_change=10, tol=1e-4):
        self.best_model_state = None
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self._no_improvement_count = 0
        self.best_loss = np.inf

    def early_stop(self, train_loss, model):
        # print(f"best loss is {self.best_loss}")
        # print(f"train loss is {train_loss}")
        if train_loss > self.best_loss - self.tol:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        if train_loss < self.best_loss:
            self.best_loss = train_loss
            # 必须deepcopy，字典是个随时改变的量，记录的是当前状态，而非最好状态。
            self.best_model_state = copy.deepcopy(model.state_dict())

        if self._no_improvement_count >= self.n_iter_no_change:
            return True, self.best_model_state
        else:
            return False, None
