# from method.tmcsv import TMC_ShapleyValue
import copy

import torch

from module.method.measure import Measure
import numpy as np
import math
import time

class MC_StructuredSampling_Shapley(Measure):
    name = 'MC-StructuredSampling-Shapley'
    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in range(len(value_functions))]
        return

    def get_contributions(self, seed, num_samples, **kwargs):

        device = self.model.device
        # start timing!
        t0 = time.time()

        num_parts = len(self.X_train_parts)

        if num_samples is None:
            num_samples = round(min(2 ** self.num_parts, math.log(self.num_parts) * self.num_parts ** 2))

        t = round(num_samples / (self.num_parts ** 2))
        # 本行是错的：t = round(num_samples / (self.num_parts ** 2) / 2)
        r = t * self.num_parts

        self.contributions = np.zeros((len(self.value_functions), num_parts))

        np.random.seed(seed)

        # next time remember to avoid the calculation of factorial to obtain distinct random orderings.
        factorial = math.factorial(num_parts)

        rng = np.random.default_rng(seed=seed)
        selected_indices = rng.choice(factorial, size=r, replace=False)

        selected_orderings = np.array([self.find_permutation(num_parts, i) for i in selected_indices])

        divided_orderings = np.zeros([num_parts, t, num_parts])
        for j in range(num_parts):
            divided_orderings[j] = selected_orderings[j * t:(j + 1) * t]
        # divided_orderings = divided_orderings.astype(int)
        # 以上对了！

        # [j][t_temp][n]
        for idx_val in range(len(self.value_functions)):
            for i in range(num_parts):
                pi_r_apostrophe = np.zeros([num_parts, t, num_parts])
                pi_r_apostrophe = pi_r_apostrophe.astype(int)
                marginal_values = []
                for j in range(num_parts):
                    for t_temp in range(t):
                        pi_r_apostrophe[j][t_temp] = copy.deepcopy(divided_orderings[j][t_temp])
                        player_i_index = np.where(divided_orderings[j][t_temp] == i)[0][0]
                        # print(pi_r_apostrophe[j][t_temp])
                        pi_r_apostrophe[j][t_temp][player_i_index], pi_r_apostrophe[j][t_temp][j] = pi_r_apostrophe[j][t_temp][j], pi_r_apostrophe[j][t_temp][player_i_index]
                        pi_r_apostrophe[j][t_temp] = pi_r_apostrophe[j][t_temp].astype(int)
                        # print(pi_r_apostrophe[j][t_temp])
                        marginal_values.append(self.evaluate_subset(set(sorted(pi_r_apostrophe[j][t_temp][:j + 1])), self.value_functions[idx_val]) - self.evaluate_subset(set(sorted(pi_r_apostrophe[j][t_temp][:j])), self.value_functions[idx_val]))
                self.contributions[idx_val][i] = np.average(marginal_values)

            # allset_value = self.evaluate_subset({i for i in range(num_parts)}, self.value_functions[idx_val])
            # pi[t][j] 其中t为迭代数 j为参与方的标号
            # pi = np.zeros((num_samples, num_parts), int)
            # v[t][j] 其中t为迭代数 j为参与方的标号 需要注意：v[t][num_parts] == 空集的价值

            # previous_contributions = np.zeros((num_samples, num_parts))

            # 只用迭代次数来控制。
            # for t in range(num_samples):
            #     v = np.zeros(num_parts + 1)
            #     v[num_parts] = self.evaluate_subset(set(), self.value_functions[idx_val])
            #     for j in range(num_parts):
            #         if abs(allset_value - v[j - 1]) < tolerance:
            #             v[j] = v[j - 1]
            #         else:
            #             v[j] = self.evaluate_subset({pi[t][k] for k in range(j + 1)}, self.value_functions[idx_val])
            #         # 这个公式可以再仔细看一下，应该是对的
            #         # print("precon", self.contributions[idx_val][pi[j]])
            #         # print("previous", t / (t + 1) * self.contributions[idx_val][pi[j]])
            #         self.contributions[idx_val][pi[t][j]] = t / (t + 1) * self.contributions[idx_val][pi[t][j]] + 1 / (
            #                     t + 1) * (v[j] - v[j - 1])
            #     # 留档
            #     # previous_contributions[t] = self.contributions[idx_val]
        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0
        return self.contributions.tolist(), t_cal

    # 历史遗留问题，应该删除！
    # 是对的吗？？
    def find_permutation(self, n, i):
        elements = list(range(n))
        permutation = []
        fact = math.factorial(n)
        for x in range(n, 0, -1):
            fact = fact // x
            index = i // fact
            permutation.append(elements.pop(index))
            i %= fact
        return tuple(permutation)