import random

from module.method.measure import Measure
import time


class RandomMethod(Measure):
    name = 'Random'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        return

    def get_contributions(self, seed, **kwargs):
        t0 = time.time()

        random.seed(seed)
        for idx_val in range(len(self.value_functions)):
            for i in range(self.num_parts):
                self.contributions[idx_val][i] = random.uniform(0, 1)

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0

        return self.contributions.tolist(), t_cal