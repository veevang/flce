import time
import torch

from module.method.measure import Measure


# https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/LOO.py

class LeaveOneOut(Measure):
    name = 'LeaveOneOut'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        return

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
            # 全数据训练
            baseline_value = self.evaluate_subset(set(range(self.num_parts)), self.value_functions[idx_val])

            for i in range(self.num_parts):  # i in [0, number of parties) i是每个参与方
                subset = set(range(self.num_parts))
                subset.discard(i)
                removed_value = self.evaluate_subset(subset, self.value_functions[idx_val])
                # 计算贡献的公式
                self.contributions[idx_val][i] = baseline_value - removed_value

        if "cuda" in str(device):
            end.record()
            torch.cuda.synchronize()
            t_cal = (time.process_time() - t0) + start.elapsed_time(end)
        else:
            t_cal = time.process_time() - t0

        # print(self.cache)
        return self.contributions.tolist(), t_cal
