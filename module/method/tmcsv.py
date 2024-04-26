import torch

from module.method.measure import Measure
import numpy as np
import time
import math


# 该方法的代码暂时被搁置
# Data Shapley: Equitable Valuation of Data for Machine Learning
# https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/TMC_Shapley.py

class TMC_ShapleyValue(Measure):
    name = 'TMC-Shapley'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in
                                  range(len(value_functions))]  # cache for computed v(S) results where S = set(...)
        return

    # 要复写
    def generate_pi(self, seed, num_parts, num_permutations):
        np.random.seed(seed)
        self.pi = np.zeros([num_permutations, num_parts])
        for t in range(num_permutations):
            self.pi[t] = np.array([i for i in range(num_parts)]).astype(int)
            np.random.shuffle(self.pi[t])
        self.pi = self.pi.astype(int)
        return self.pi

    def get_contributions(self, seed, num_samples, tolerance=1e-6, **kwargs):
        """
        Use the TMC-Shapley method to score each data source.
        tolerance指的是truncate的指标
        """
        device = self.model.device
        # start timing!
        t0 = time.time()

        num_parts = len(self.X_train_parts)

        if num_samples is None:
            num_samples = round(min(2 ** self.num_parts, math.log(self.num_parts) * self.num_parts ** 2))
        num_permutations = round(num_samples / self.num_parts)

        pi = self.generate_pi(seed, num_parts=num_parts, num_permutations=num_permutations)

        self.contributions = np.zeros((len(self.value_functions), num_parts))

        for idx_val in range(len(self.value_functions)):
            allset_value = self.evaluate_subset({i for i in range(num_parts)}, self.value_functions[idx_val])
            # pi[t][j] 其中t为迭代数 j为参与方的标号
            # pi = np.zeros((num_samples, num_parts), int)
            # v[t][j] 其中t为迭代数 j为参与方的标号 需要注意：v[t][num_parts] == 空集的价值

            # previous_contributions = np.zeros((num_samples, num_parts))

            # 只用迭代次数来控制。
            for t in range(num_permutations):
                v = np.zeros(num_parts + 1)
                v[num_parts] = self.evaluate_subset(set(), self.value_functions[idx_val])
                for j in range(num_parts):
                    if abs(allset_value - v[j - 1]) < tolerance:
                        v[j] = v[j - 1]
                    else:
                        # 一定要包含j
                        v[j] = self.evaluate_subset({pi[t][k] for k in range(j + 1)}, self.value_functions[idx_val])
                    # 这个公式可以再仔细看一下，应该是对的
                    # print("precon", self.contributions[idx_val][pi[j]])
                    # print("previous", t / (t + 1) * self.contributions[idx_val][pi[j]])
                    self.contributions[idx_val][pi[t][j]] = t / (t + 1) * self.contributions[idx_val][pi[t][j]] + 1 / (
                            t + 1) * (v[j] - v[j - 1])
                # 留档
                # previous_contributions[t] = self.contributions[idx_val]

        # if seed == 6690:
        #     pprint(self.cache, width=1)
        #     pprint(self.pi)
        #     pprint(self.contributions[0][0])

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0

        return self.contributions.tolist(), t_cal

