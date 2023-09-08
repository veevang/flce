import time
import torch

from measure.measure import Measure
import numpy as np
import copy
import scipy


# https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/LOO.py

class RV_LeaveOneOut(Measure):
    name = 'robust volume leave one out'

    def __init__(self, loader, model, cache):
        super().__init__(loader, model, cache)
        self.name = self.__class__.name
        return

    def get_contributions(self, *args, **kwargs):
        value_function = "robust volume"
        t0 = time.process_time()
        # 全数据训练
        baseline_value = self.evaluate_subset(set(range(self.num_parts)), value_function, omega=0.1)

        for i in range(self.num_parts):  # i in [0, number of parties) i是每个参与方
            subset = set(range(self.num_parts))
            subset.discard(i)
            removed_value = self.evaluate_subset(subset, value_function, omega=0.1)
            # 计算贡献的公式
            self.contributions[i] = baseline_value - removed_value
        t = time.process_time() - t0
        return self.contributions.tolist(), t
