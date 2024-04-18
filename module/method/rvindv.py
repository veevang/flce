from module.method.measure import Measure
import time


class RV_Individual(Measure):
    name = 'robust volume individual'

    def __init__(self, loader, model, cache):
        super().__init__(loader, model, cache)
        self.name = self.__class__.name
        return

    def get_contributions(self, *args, **kwargs):
        value_function = "robust volume"
        t0 = time.process_time()

        for i in range(self.num_parts):  # i in [0, number of parties)
            self.contributions[i] = self.evaluate_subset({i}, value_scoring=value_function, omega=0.1)

        t = time.process_time() - t0
        return self.contributions.tolist(), t
