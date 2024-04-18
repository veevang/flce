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

    def _fed_train(self,
                   X_train_parts,
                   y_train_parts,
                   num_global_rounds,
                   num_local_rounds,
                   loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss(), nn.BCELoss()],
                   lr,
                   score: bool,
                   batch_size,
                   **kwargs
                   ):

        self.load_state_dict(self.initial_state_dict)
        self.incremental_seed = self.seed
        torch.manual_seed(self.seed)

        if score:
            test_interval = kwargs.get("test_interval")
            value_functions = kwargs.get("value_functions")
            X_test, y_test = kwargs.get("X_test"), kwargs.get("y_test")
            val_list = np.zeros(len(value_functions))

        num_clients = len(X_train_parts)

        self.load_state_dict(self.initial_state_dict)
        device = self.device

        m = np.zeros(num_clients)
        for i in range(num_clients):
            m[i] = X_train_parts[i].size(0)

        # start training
        for t in range(num_global_rounds):
            # distribute model, make sure the Adam is initialized locally
            backup_model = copy.deepcopy(self)
            # it is the same as the following:
            # backup_model = self.__class__(seed=self.seed, lr=self.lr, num_epoch=self.num_epoch, hidden_layer_size=self.hidden_layer_size, device=self.device, batch_size=self.batch_size)
            # backup_model.load_state_dict(self.state_dict())
            models = [copy.deepcopy(backup_model) for _ in range(num_clients)]
            deltas = []

            # client update
            for i in range(num_clients):
                # models[i] = copy.deepcopy(backup_model)
                models[i] = self.client_update(X_train_parts[i], y_train_parts[i], models[i], num_local_rounds)
                deltas.append(self.compute_grad_update(old_model=backup_model, new_model=models[i], device=device))

            # -------- run on server side ---------
            # FedAvg
            weights = m / np.sum(m)
            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in self.parameters()]
            for delta, weight in zip(deltas, weights):
                self.add_gradient_updates(aggregated_gradient, delta, weight=weight)
            self = self.add_update_to_model(self, aggregated_gradient)

            if score:
                if t % test_interval == 0 and t != 0:
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
        return val_list

    @staticmethod
    def client_update(X_train_client, y_train_client, model_last_round, num_local_epochs):
        new_model = copy.deepcopy(model_last_round)
        new_model.fit(X_train_client, y_train_client, incremental=True, num_epochs=num_local_epochs)
        return new_model

    @staticmethod
    def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
        assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"

        for param_1, param_2 in zip(grad_update_1, grad_update_2):
            param_1.data += param_2.data * weight

    @staticmethod
    def add_update_to_model(model, update, weight=1.0, device=None):
        if not update:
            return model
        if device:
            model = model.to(device)
            update = [param.to(device) for param in update]

        for param_model, param_update in zip(model.parameters(), update):
            param_model.data += weight * param_update.data
        return model

    @staticmethod
    def compute_grad_update(old_model, new_model, device=None):
        # maybe later to implement on selected layers/parameters
        if device:
            old_model, new_model = old_model.to(device), new_model.to(device)
        return [(new_param.data - old_param.data) for old_param, new_param in
                zip(old_model.parameters(), new_model.parameters())]

    @staticmethod
    def flatten(grad_update):
        return torch.cat([update.data.view(-1) for update in grad_update])

    @staticmethod
    def unflatten(flattened, normal_shape):
        grad_update = []
        for param in normal_shape:
            n_params = len(param.view(-1))
            grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
            flattened = flattened[n_params:]
        return grad_update

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
        # print(optimizer.state_dict())
        # optimizer = optim.SGD(self.parameters(), lr=lr)

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


# class CNNMNIST(Net):
#     def __init__(self, seed, device):
#         super().__init__(seed=seed, device=device)
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(7 * 7 * 32, 128)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#         self.initial_state_dict = copy.deepcopy(self.state_dict())
#         return
#
#     def forward(self, x):
#         x = self.pool1(self.relu1(self.conv1(x)))
#         x = self.pool2(self.relu2(self.conv2(x)))
#         x = x.view(-1, 7 * 7 * 32)
#         x = self.relu3(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#     # 假设训练两个epoch？
#     def fit(self, X_train, y_train):
#         return self._fit(X_train, y_train, num_epochs=10, loss_fun=nn.CrossEntropyLoss(), lr=0.01)
#
#     def predict(self, X_test):
#         return self._predict(X_test)
#
#     def _predict(self,
#                  X_test: torch.tensor,
#                  # loss_fun: Union[nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss(), nn.NLLLoss()] = nn.BCEWithLogitsLoss(),
#                  # test_losses=None
#                  ):
#         torch.manual_seed(self.seed)
#         self.eval()
#         predicted_labels = []
#
#         test_dataset = torch.utils.data.TensorDataset(X_test)
#         test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#         # Iterate over the test dataset
#         for data in test_dataloader:
#             inputs = data
#
#             # Forward pass through the model
#             outputs = self(inputs)
#
#             # Get the predicted labels
#             # _, predicted = torch.max(outputs.data, 1)
#             _, predicted = torch.max(outputs.data, 1)
#             predicted_labels.extend(predicted.tolist())
#
#         # Convert the lists to NumPy arrays
#         predicted_labels = np.array(predicted_labels)
#
#         return predicted_labels


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

    # def fit_and_score(self, X_train, y_train, X_test, y_test, value_functions):
    #     return self._fit_and_score(X_train, y_train, X_test, y_test, value_functions, num_epochs=self.num_epoch,
    #                                loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size)
    # num epoch:40, lr: 0.001, accu:0.854793793926535

    # 改的新的，试试看！2024年3月25日06:14:05
    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

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

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

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

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

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

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

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

    def fed_train_and_score(self, X_train_parts, y_train_parts, X_test, y_test, value_functions):
        return self._fed_train(X_train_parts, y_train_parts, num_global_rounds=self.num_epoch, num_local_rounds=1,
                   loss_fun=nn.BCELoss(), lr=self.lr, test_interval=1, batch_size=self.batch_size, score=True,
                               X_test=X_test, y_test=y_test, value_functions=value_functions)

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
