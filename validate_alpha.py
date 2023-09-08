import scipy.stats
import numpy as np
alpha_list = [0.8 for _ in range(8)]
ratios = scipy.stats.dirichlet.rvs(alpha_list, size=1000, random_state=666)
col_mean = np.mean([sorted(ratios[i]) for i in range(len(ratios))], axis=0)
print(col_mean)
