# http://github.com/cyyever/torch_algorithm
# GTG-Shapley
from __future__ import annotations
import copy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor
from module.method.measure import Measure
import time
import math


class GTG_Shapley(Measure):
    name = "GTG-Shapley"

    def __init__(self, loader, model, cache, value_functions):
        self.pi_collection = None
        self.m = None

        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name

        self.eps = 1e-6
        self.round_trunc_threshold = 1e-6

        self.num_parts = len(loader.X_train_parts)

        self.model = copy.deepcopy(model)
        self.model.load_state_dict(self.model.initial_state_dict)

        # seed = self.model.seed
        # lr = self.model.lr
        # num_epoch = self.model.num_epoch
        # hidden_layer_size = self.model.hidden_layer_size
        # device = self.model.device
        # batch_size = self.model.batch_size

        return

    def get_contributions(self, seed, num_local_epochs, num_samples, **kwargs):
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
        cons = np.zeros([len(self.value_functions), num_parts])

        self.model.load_state_dict(self.model.initial_state_dict)
        device = self.model.device

        self.m = np.zeros(num_parts)
        for i in range(num_parts):
            self.m[i] = self.X_train_parts[i].size(0)

        # d = self.X_train_parts[0].shape(1)
        # N = set([i for i in range(num_parts)])

        for t in range(num_epochs):
            backup_model = copy.deepcopy(self.model)
            models = [copy.deepcopy(self.model) for _ in range(num_parts)]
            deltas = []

            for i in range(num_parts):
                # models[i] = copy.deepcopy(backup_model)
                models[i] = self.client_update(i, models[i], num_local_epochs)
                deltas.append(self.compute_grad_update(old_model=backup_model, new_model=models[i], device=device))

            # -------- run on server side ---------
            # FedAvg
            weights = self.m / np.sum(self.m)
            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in self.model.parameters()]
            for delta, weight in zip(deltas, weights):
                self.add_gradient_updates(aggregated_gradient, delta, weight=weight)
            self.model = self.add_update_to_model(self.model, aggregated_gradient)

            # GTG-shapley
            round_cons = self.gtg_shapley(backup_model, self.model, self.value_functions, deltas, seed + t, num_samples)

            cons = np.add(cons, round_cons)

        self.contributions = cons

        # timing end!
        if "cuda" in str(device):
            end.record()
            torch.cuda.synchronize()
            t_cal = (time.process_time() - t0) + start.elapsed_time(end)
        else:
            t_cal = time.process_time() - t0

        return cons, t_cal

    def gtg_shapley(self, model_init, model_fin, value_functions, deltas, round_sample_seed, num_samples):
        num_parts = self.num_parts
        round_cons = np.zeros([len(self.value_functions), num_parts])

        init_global_vals = np.zeros([len(value_functions)])
        fin_global_vals = np.zeros([len(value_functions)])
        for val_index, value_function in enumerate(value_functions):
            init_global_vals[val_index] = self.measure_value(model_init, value_function)
            fin_global_vals[val_index] = self.measure_value(model_fin, value_function)

        round_value_dict = {str(set()): init_global_vals, str(set(sorted(range(num_parts)))): fin_global_vals}


        if num_samples is None:
            num_samples = round(min(2 ** self.num_parts, math.log(self.num_parts) * self.num_parts ** 2))
        num_permutations = round(num_samples / self.num_parts)
        pi_collection = self.generate_pi(round_sample_seed, num_parts, num_permutations)


        if any(abs(init_global_val - fin_global_val) > self.round_trunc_threshold for init_global_val, fin_global_val in
               zip(init_global_vals, fin_global_vals)):
            for k, pi in enumerate(pi_collection):
                values = init_global_vals
                for j in range(self.num_parts):
                    prefix_values = values
                    # truncation
                    if any(abs(fin_global_val - prefix_value) for fin_global_val, prefix_value in
                           zip(fin_global_vals, prefix_values)) >= self.eps:
                        value_set = set(sorted(pi[:j + 1].tolist()))
                        # use cache
                        if round_value_dict.get(str(value_set)) is None:

                            values = np.zeros([len(self.value_functions)])
                            model_cb = copy.deepcopy(model_init)

                            # FedAvg
                            cb_client_sizes = torch.tensor([self.m[i] for i in value_set])
                            cb_client_weights: Tensor = torch.div(cb_client_sizes, torch.sum(cb_client_sizes))
                            cb_client_deltas = [deltas[i] for i in value_set]
                            cb_aggregated_delta = [torch.zeros(param.shape).to(self.model.device) for param in
                                                   self.model.parameters()]
                            for cb_client_delta, cb_weight in zip(cb_client_deltas, cb_client_weights):
                                self.add_gradient_updates(cb_aggregated_delta, cb_client_delta, weight=cb_weight)

                            model_cb = self.add_update_to_model(model_cb, cb_aggregated_delta)

                            for val_index, value_function in enumerate(self.value_functions):
                                values[val_index] = self.measure_value(model_cb, value_function)

                            round_value_dict[str(value_set)] = copy.deepcopy(values)
                        else:
                            # in cache
                            values = round_value_dict[str(value_set)]
                    else:
                        # truncate!
                        values = prefix_values

                    for val_index, value_function in enumerate(self.value_functions):
                        # print(f"pi = {pi}, round_cons={round_cons}, ")
                        round_cons[val_index][pi[j]] = k / (k + 1) * round_cons[val_index][pi[j]] + 1 / (k + 1) * (
                                values[val_index] - prefix_values[val_index])
        return round_cons

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

    def generate_pi(self, seed, num_parts, num_permutations):
        np.random.seed(seed)

        if num_permutations % num_parts != 0:
            raise Exception("number of permutations not multiple of num_parts!")
        else:
            repeated_times = round(num_permutations / num_parts)

        self.pi_collection = np.zeros([num_permutations, num_parts])

        for t in range(repeated_times):
            for i in range(num_parts):
                self.pi_collection[t * num_parts + i] = np.array([i for i in range(num_parts)])
                self.pi_collection[t * num_parts + i][i], self.pi_collection[t * num_parts + i][0] = \
                    self.pi_collection[t * num_parts + i][0], \
                    self.pi_collection[t * num_parts + i][i]
                self.pi_collection[t * num_parts + i][1:] = np.random.permutation(
                    self.pi_collection[t * num_parts + i][1:])

        self.pi_collection = self.pi_collection.astype(int)
        return self.pi_collection

    # def compute(self):
    #     assert self.metric_fun is not None
    #     self.round_number += 1
    #     this_round_metric = self.metric_fun(
    #         self.round_number, set(range(self.worker_number))
    #     )
    #     if this_round_metric is None:
    #         get_logger().warning("force stop")
    #         return
    #     if (
    #         abs(self.last_round_metric - this_round_metric)
    #         <= self.round_trunc_threshold
    #     ):
    #         self.shapley_values[self.round_number] = {
    #             i: 0 for i in range(self.worker_number)
    #         }
    #         self.shapley_values_S[self.round_number] = {
    #             i: 0 for i in range(self.worker_number)
    #         }
    #         if self.save_fun is not None:
    #             self.save_fun(
    #                 self.round_number,
    #                 self.shapley_values[self.round_number],
    #                 self.shapley_values_S[self.round_number],
    #             )
    #         get_logger().info(
    #             "skip round %s, this_round_metric %s last_round_metric %s round_trunc_threshold %s",
    #             self.round_number,
    #             this_round_metric,
    #             self.last_round_metric,
    #             self.round_trunc_threshold,
    #         )
    #         self.last_round_metric = this_round_metric
    #         return
    #     metrics = dict()

    #     # for best_S
    #     perm_records = dict()

    #     index = 0
    #     contribution_records: list = []
    #     while self.not_convergent(index, contribution_records):
    #         for worker_id in range(self.worker_number):
    #             index += 1
    #             v: list = [0] * (self.worker_number + 1)
    #             v[0] = self.last_round_metric
    #             marginal_contribution = [0 for i in range(self.worker_number)]
    #             perturbed_indices = np.concatenate(
    #                 (
    #                     np.array([worker_id]),
    #                     np.random.permutation(
    #                         [i for i in range(self.worker_number) if i != worker_id]
    #                     ),
    #                 )
    #             ).astype(int)

    #             for j in range(1, self.worker_number + 1):
    #                 subset = tuple(sorted(perturbed_indices[:j].tolist()))
    #                 # truncation
    #                 if abs(this_round_metric - v[j - 1]) >= self.eps:
    #                     if subset not in metrics:
    #                         if not subset:
    #                             metric = self.last_round_metric
    #                         else:
    #                             metric = self.metric_fun(self.round_number, subset)
    #                             if metric is None:
    #                                 get_logger().warning("force stop")
    #                                 return
    #                         get_logger().info(
    #                             "round %s subset %s metric %s",
    #                             self.round_number,
    #                             subset,
    #                             metric,
    #                         )
    #                         metrics[subset] = metric
    #                     v[j] = metrics[subset]
    #                 else:
    #                     v[j] = v[j - 1]

    #                 # update SV
    #                 marginal_contribution[perturbed_indices[j - 1]] = v[j] - v[j - 1]
    #             contribution_records.append(marginal_contribution)
    #             # for best_S
    #             perm_records[tuple(perturbed_indices.tolist())] = marginal_contribution

    #     # for best_S
    #     subset_rank = sorted(
    #         metrics.items(), key=lambda x: (x[1], -len(x[0])), reverse=True
    #     )
    #     best_S: tuple = None
    #     if subset_rank[0][0]:
    #         best_S = subset_rank[0][0]
    #     else:
    #         best_S = subset_rank[1][0]

    #     contrib_S = [
    #         v for k, v in perm_records.items() if set(k[: len(best_S)]) == set(best_S)
    #     ]
    #     SV_calc_temp = np.sum(contrib_S, 0) / len(contrib_S)
    #     round_marginal_gain_S = metrics[best_S] - self.last_round_metric
    #     round_SV_S: dict = dict()
    #     for client_id in best_S:
    #         round_SV_S[client_id] = float(SV_calc_temp[client_id])

    #     self.shapley_values_S[
    #         self.round_number
    #     ] = ShapleyValue.normalize_shapley_values(round_SV_S, round_marginal_gain_S)

    #     # calculating fullset SV
    #     # shapley value calculation
    #     if set(best_S) == set(range(self.worker_number)):
    #         self.shapley_values[self.round_number] = copy.deepcopy(
    #             self.shapley_values_S[self.round_number]
    #         )
    #     else:
    #         round_shapley_values = np.sum(contribution_records, 0) / len(
    #             contribution_records
    #         )
    #         assert len(round_shapley_values) == self.worker_number

    #         round_marginal_gain = this_round_metric - self.last_round_metric
    #         round_shapley_value_dict = dict()
    #         for idx, value in enumerate(round_shapley_values):
    #             round_shapley_value_dict[idx] = float(value)

    #         self.shapley_values[
    #             self.round_number
    #         ] = ShapleyValue.normalize_shapley_values(
    #             round_shapley_value_dict, round_marginal_gain
    #         )

    #     if self.save_fun is not None:
    #         self.save_fun(
    #             self.round_number,
    #             self.shapley_values[self.round_number],
    #             self.shapley_values_S[self.round_number],
    #         )
    #     get_logger().info("shapley_value %s", self.shapley_values[self.round_number])
    #     get_logger().info(
    #         "shapley_value_S %s", self.shapley_values_S[self.round_number]
    #     )
    #     self.last_round_metric = this_round_metric

    # def not_convergent(self, index, contribution_records):
    #     if index >= self.max_number:
    #         get_logger().info("convergent for max_number %s", self.max_number)
    #         return False
    #     if index <= self.converge_min:
    #         return True
    #     all_vals = (
    #         np.cumsum(contribution_records, 0)
    #         / np.reshape(np.arange(1, len(contribution_records) + 1), (-1, 1))
    #     )[-self.last_k:]
    #     errors = np.mean(
    #         np.abs(all_vals[-self.last_k:] - all_vals[-1:])
    #         / (np.abs(all_vals[-1:]) + 1e-12),
    #         -1,
    #     )
    #     if np.max(errors) > self.converge_criteria:
    #         return True
    #     get_logger().info(
    #         "convergent for index %s and converge_min %s max error %s converge_criteria %s",
    #         index,
    #         self.converge_min,
    #         np.max(errors),
    #         self.converge_criteria,
    #     )
    #     return False
