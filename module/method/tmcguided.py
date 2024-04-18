from module.method.tmcsv import TMC_ShapleyValue
import numpy as np


class TMC_GuidedSampling_Shapley(TMC_ShapleyValue):
    name = 'TMC-GuidedSampling-Shapley'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        # self.cache[str(set())] = [0 for _ in range(len(value_functions))]
        return

    def generate_pi(self, seed, num_parts, num_permutations):
        np.random.seed(seed)

        if num_permutations % num_parts != 0:
            raise Exception("number of permutations not multiple of num_parts!")
        else:
            repeated_times = round(num_permutations / num_parts)

        self.pi = np.zeros([num_permutations, num_parts])

        for t in range(repeated_times):
            for i in range(num_parts):
                self.pi[t * num_parts + i] = np.array([i for i in range(num_parts)])
                self.pi[t * num_parts + i][i], self.pi[t * num_parts + i][0] = self.pi[t * num_parts + i][0], \
                                                                               self.pi[t * num_parts + i][i]
                self.pi[t * num_parts + i][1:] = np.random.permutation(self.pi[t * num_parts + i][1:])

        self.pi = self.pi.astype(int)
        return self.pi
