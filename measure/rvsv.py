# https://github.com/ZhaoxuanWu/VolumeBased-DataValuation
from itertools import permutations
from math import floor

import scipy
import torch
from collections import defaultdict, Counter
from measure.measure import Measure
import numpy as np
from itertools import combinations as cbs
from math import comb
import time

# 解决了，evaluate subset <- compute robust volumes
# 必须要注意，这里的value_scoring并不作数，因为它采用的是自己的独特的衡量volume的方法（RV）。

"""
function: compute_robust_volumes
Discretized Robust Volume-based Shapley values
"""


class RV_ShapleyValue(Measure):
    name = 'robust volume shapley value'

    def __init__(self, loader, model, cache, omega=0.1):
        super().__init__(loader, model, cache)
        self.name = 'robust volume shapley value'
        self.cache[str(set())] = 0  # cache for computed v(S) results where S = set(...)
        self.omega = omega
        return

    # 每一个参与方的贡献
    # 和sv一样
    def phi(self, i, value_function, **kwargs):
        # 贡献值
        accu = 0.
        # 所有参与方全集
        all_set = set(range(self.num_parts))
        # all set指的是移除了第i个参与方后的剩下参与方
        all_set.remove(i)
        # S_size in [0, num_part)
        for S_size in range(len(all_set) + 1):  # 1/N is the position prob, C(N-1,l) is #combs of l parts
            # 考虑大小为S_size的子集，计算其权重
            weight = 1. / self.num_parts / comb(self.num_parts - 1, S_size)
            for cb in cbs(all_set, S_size):
                set_with_i = set(cb)
                set_with_i.add(i)
                set_without_i = set(cb)
                accu += (self.evaluate_subset(set_with_i, value_function, omega=self.omega) - self.evaluate_subset(
                    set_without_i, value_function, omega=self.omega)) * weight  # （(v(S U {i}) - v(S))*weight）再求和
        return accu

    def get_contributions(self, *args, **kwargs):
        value_function = "robust volume"
        t0 = time.process_time()
        # 得到每一个参与方的分数
        for i in range(self.num_parts):
            self.contributions[i] = self.phi(i, value_function)
        t = time.process_time() - t0
        return self.contributions.tolist(), t

