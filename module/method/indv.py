import torch

from module.method.measure import Measure
import time


class Individual(Measure):
    name = 'Individual'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        return

    def get_contributions(self, **kwargs):

        device = self.model.device
        # start timing!
        t0 = time.time()


        for idx_val in range(len(self.value_functions)):
            for i in range(self.num_parts):  # i in [0, number of parties)
                self.contributions[idx_val][i] = self.evaluate_subset({i}, value_function=self.value_functions[idx_val])
                # print(f"{i} value_function={self.value_functions[idx_val]}, con = {self.contributions[idx_val][i]}")

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0

        return self.contributions.tolist(), t_cal
