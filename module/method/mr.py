# Efficient and Fair Data Valuation for Horizontal Federated Learning
import copy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from module.method.measure import Measure
from itertools import combinations as cbs
from math import comb
import time


class Multi_Rounds(Measure):
    name = "MultiRounds"

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions)
        self.num_parts = len(loader.X_train_parts)
        self.name = self.__class__.name

        # global model
        self.model = copy.deepcopy(model)
        self.model.load_state_dict(self.model.initial_state_dict)

        # seed = self.model.seed
        # lr = self.model.lr
        # num_epoch = self.model.num_epoch
        # hidden_layer_size = self.model.hidden_layer_size
        # device = self.model.device
        # batch_size = self.model.batch_size

        return

    # 考虑cache
    # #rounds: num_rounds or convergence criteria
    def get_contributions(self, seed, decfac, num_local_epochs, C=None, **kwargs):
        """
        decfac : lambda
        """

        device = self.model.device
        # start timing!
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if "cuda" in str(device):
            torch.cuda.synchronize()
            start.record()
        t0 = time.process_time()

        num_epochs = self.model.num_epoch
        num_parts = self.num_parts
        if C is None:
            C = 1 / num_parts
        # N = {i for i in range(num_parts)}
        # d = self.X_train_parts[0].shape(1)
        m = np.zeros(num_parts)
        for i in range(num_parts):
            m[i] = self.X_train_parts[i].size(0)

        # init global model
        self.model.load_state_dict(self.model.initial_state_dict)

        round_cons = np.zeros([len(self.value_functions), num_epochs, num_parts])

        device = self.model.device

        '''start training! '''
        for t in range(num_epochs):
            '''calculate global model'''
            backup_model = copy.deepcopy(self.model)

            models = [copy.deepcopy(self.model) for _ in range(num_parts)]
            model_tilde_dict = dict()
            deltas = []
            utility_dict = {str(set()): [0, 0, 0, 0]}

            for i in range(num_parts):
                models[i] = copy.deepcopy(backup_model)
                models[i] = self.client_update(i, models[i], num_local_epochs)
                deltas.append(self.compute_grad_update(old_model=backup_model, new_model=models[i], device=device))

            weights = m / np.sum(m)
            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in self.model.parameters()]
            for delta, weight in zip(deltas, weights):
                self.add_gradient_updates(aggregated_gradient, delta, weight=weight)
            self.model = self.add_update_to_model(self.model, aggregated_gradient)

            # 现在 self.model是更新过的model, 即model(t+1)

            for S_size in range(1, self.num_parts + 1):
                for cb in cbs(list(range(self.num_parts)), S_size):
                    sorted_cb = set(sorted(set(cb)))

                    cb_client_sizes = torch.tensor([m[i] for i in sorted_cb])
                    cb_client_weights: Tensor = torch.div(cb_client_sizes, torch.sum(cb_client_sizes))
                    cb_client_deltas = [deltas[i] for i in sorted_cb]
                    cb_aggregated_delta = [torch.zeros(param.shape).to(device) for param in self.model.parameters()]
                    for cb_client_delta, cb_weight in zip(cb_client_deltas, cb_client_weights):
                        self.add_gradient_updates(cb_aggregated_delta, cb_client_delta, weight=cb_weight)

                    model_tilde_dict[str(sorted_cb)] = copy.deepcopy(backup_model)
                    model_tilde_dict[str(sorted_cb)] = self.add_update_to_model(model_tilde_dict[str(sorted_cb)],
                                                                                cb_aggregated_delta)

            # calculate the round-CIs
            # shapley of a round!
            for S_size in range(1, self.num_parts + 1):
                for cb in cbs(list(range(self.num_parts)), S_size):
                    sorted_cb = set(sorted(set(cb)))

                    utility_dict[str(sorted_cb)] = [self.measure_value(model_tilde_dict[str(sorted_cb)], value_function) for value_function in self.value_functions]

            for i in range(num_parts):
                for val_index, value_function in enumerate(self.value_functions):
                    # 贡献值
                    value = 0.
                    all_set = set(range(self.num_parts))
                    all_set.remove(i)
                    for S_size in range(len(all_set) + 1):
                        weight = 1. / self.num_parts / comb(self.num_parts - 1, S_size)
                        for cb in cbs(all_set, S_size):
                            set_with_i = set(cb)
                            set_with_i.add(i)
                            set_with_i = set(sorted(set_with_i))

                            set_without_i = set(cb)
                            set_without_i = set(sorted(set_without_i))

                            value += (utility_dict[str(set_with_i)][val_index] - utility_dict[str(set_without_i)][val_index]) * weight

                    round_cons[val_index][t][i] = C * value

        '''calculate the contributions'''
        # line 18
        final_cons = np.zeros([len(self.value_functions), num_parts])
        for val_index, value_function in enumerate(self.value_functions):
            for i in range(num_parts):
                for t in range(1, num_epochs + 1):
                    denominator = sum([round_cons[val_index][t - 1][i] for i in range(self.num_parts)])
                    final_cons[val_index][i] += (decfac ** t) * round_cons[val_index][t - 1][i] / denominator

        self.contributions = final_cons

        if "cuda" in str(device):
            end.record()
            torch.cuda.synchronize()
            t_cal = (time.process_time() - t0) + start.elapsed_time(end) * 1e-3
        else:
            t_cal = time.process_time() - t0

        return final_cons, t_cal

    # Client do ...
    def client_update(self, i, model_last_round, num_local_epochs):
        new_model = copy.deepcopy(model_last_round)
        new_model.fit(self.X_train_parts[i], self.y_train_parts[i], incremental=True, num_epochs=num_local_epochs)
        return new_model

    def measure_value(self, model, value_function):
        y_pred = model.predict(self.X_test)
        if value_function == "accuracy":
            metric = accuracy_score(y_true=self.y_test, y_pred=y_pred)
        elif value_function == "f1":
            metric = f1_score(y_true=self.y_test, y_pred=y_pred)
        elif value_function == "f1_macro":
            metric = f1_score(y_true=self.y_test, y_pred=y_pred, average="macro")
        elif value_function == "f1_micro":
            metric = f1_score(y_true=self.y_test, y_pred=y_pred, average="micro")
        else:
            raise ValueError("model performance value function does not exist! ")
        return metric
