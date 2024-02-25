import scipy.stats
import numpy as np
# alpha_list = [0.625 for _ in range(14)]
alpha_list = [0.3 for _ in range(8)]
ratios = scipy.stats.dirichlet.rvs(alpha_list, size=1000, random_state=666)
col_mean = np.mean([sorted(ratios[i]) for i in range(len(ratios))], axis=0)
print(col_mean)
print(col_mean*500000)
