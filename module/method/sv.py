import time

from module.method.measure import Measure
from itertools import combinations as cbs
from math import comb
import torch


class ShapleyValue(Measure):
    name = 'ShapleyValue'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in
                                  range(len(value_functions))]  # cache for computed v(S) results where S = set(...)
        # 根据game的定义，v(空集)=0
        return

    # 每一个参与方的贡献
    def phi(self, i, value_function):
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
                accu += (self.evaluate_subset(set_with_i, value_function) - self.evaluate_subset(set_without_i,
                                                                                                 value_function)) * weight  # （(v(S U {i}) - v(S))*weight）再求和
        return accu

    def get_contributions(self, **kwargs):
        device = self.model.device
        # start timing!
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if "cuda" in str(device):
            torch.cuda.synchronize()
            start.record()
        t0 = time.process_time()

        for idx_val in range(len(self.value_functions)):
            # 得到每一个参与方的分数
            for i in range(self.num_parts):
                self.contributions[idx_val][i] = self.phi(i, self.value_functions[idx_val])

        if "cuda" in str(device):
            end.record()
            torch.cuda.synchronize()
            t_cal = (time.process_time() - t0) + start.elapsed_time(end)
        else:
            t_cal = time.process_time() - t0
        return self.contributions.tolist(), t_cal
