import time
from module.method.measure import *
import numpy as np
from scipy import optimize as opt
from itertools import combinations as cbs


class LeastCore(Measure):
    name = 'LeastCore'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in range(len(value_functions))]  # cache for computed v(S) results where S = set(...)
        return

    def compute_ub(self, value_function):
        assert self.num_parts >= 2  # 此时才需要A_ub
        self.A_ub = None
        self.b_ub = np.empty(0)
        # 根据定义，包含空集和全集情况。
        for S_size in range(0, self.num_parts + 1):
            for cb in cbs(set(range(self.num_parts)), S_size):
                row = np.zeros(self.num_parts)
                # 选中的系数为-1
                for index in cb:
                    row[index] = -1
                # e的系数为-1
                row = np.append(row, -1)

                # A添加一行
                if self.A_ub is None:
                    self.A_ub = np.array([row])
                else:
                    self.A_ub = np.concatenate((self.A_ub, np.array([row])), axis=0)
                # b添加一项
                self.b_ub = np.append(self.b_ub, -self.evaluate_subset(set(cb), value_function))
        return

    def compute_eq(self, value_function):
        # phi == 1
        self.A_eq = np.ones(self.num_parts)
        # e == 0
        self.A_eq = np.append(self.A_eq, 0)
        self.A_eq = np.array([self.A_eq])
        self.b_eq = np.array([self.evaluate_subset(set(range(self.num_parts)), value_function)])
        return

    def get_contributions(self, **kwargs):
        device = self.model.device
        # start timing!
        t0 = time.time()

        assert self.num_parts >= 2

        for idx_val in range(len(self.value_functions)):
            # 构造linprog的参数
            # 不等式参数<=
            self.compute_ub(self.value_functions[idx_val])
            # 等式参数
            self.compute_eq(self.value_functions[idx_val])

            # 构造c
            c = np.zeros(self.num_parts)
            c = np.append(c, 1)

            # 构造bounds
            bound = (-self.num_parts, self.num_parts)
            self.bounds = [bound for _ in range(self.num_parts)]
            # 认为e的值域取[v(空集), inf)，即考虑了LC定义中S=空集的情况所对应的不等式
            self.bounds.append((0, +np.inf))

            # 做线性回归，在此之前需要保证A_ub, b_ub等被计算出了，需要注意bound的设置不能是默认的
            ans = opt.linprog(c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, method='highs', bounds=self.bounds)

            if ans["success"] == False:
                raise ValueError(ans["message"])

            # 判断用该采样计算出的分数是否在我们认为正常的范围内，如果不在则报错，认为采样不合理。
            for i in range(self.num_parts):
                if ans['x'][i] == bound[0] or ans['x'][i] == bound[1]:
                    raise ValueError("The linear program is unbounded.")

            # 得到每一个参与方的分数
            for i in range(self.num_parts):
                self.contributions[idx_val][i] = copy.deepcopy(ans['x'][i])

            print(ans['x'][self.num_parts])

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0

        return self.contributions.tolist(), t_cal
