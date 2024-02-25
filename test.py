# import time
# from measure.measure import *
# import numpy as np
# from scipy import optimize as opt
# from itertools import combinations as cbs
# class LeastCore(Measure):
#     name = 'LeastCore'
#
#     def __init__(self, loader: DataLoader, model, cache, value_functions):
#         super().__init__()
#         self.name = self.__class__.name
#         self.cache[str(set())] = [0 for _ in range(len(value_functions))]  # cache for computed v(S) results where S = set(...)
#         return
#
#     def compute_ub(self, value_function):
#         assert self.num_parts >= 2  # 此时才需要A_ub
#         self.A_ub = None
#         self.b_ub = np.empty(0)
#         # 根据定义，包含空集和全集情况。
#         for S_size in range(0, self.num_parts + 1):
#             for cb in cbs(set(range(self.num_parts)), S_size):
#                 row = np.zeros(self.num_parts)
#                 # 选中的系数为-1
#                 for index in cb:
#                     row[index] = -1
#                 # e的系数为-1
#                 row = np.append(row, -1)
#
#                 # A添加一行
#                 if self.A_ub is None:
#                     self.A_ub = np.array([row])
#                 else:
#                     self.A_ub = np.concatenate((self.A_ub, np.array([row])), axis=0)
#                 # b添加一项
#                 self.b_ub = np.append(self.b_ub, -self.evaluate_subset(set(cb), value_function))
#         return
#
#     def compute_eq(self, value_function):
#         # phi == 1
#         self.A_eq = np.ones(self.num_parts)
#         # e == 0
#         self.A_eq = np.append(self.A_eq, 0)
#         self.A_eq = np.array([self.A_eq])
#         self.b_eq = np.array([self.evaluate_subset(set(range(self.num_parts)), value_function)])
#         return
#
#     def get_contributions(self, **kwargs):
#         device = self.model.device
#         # start timing!
#         start = torch.cuda.Event(enable_timing=True)
#         end = torch.cuda.Event(enable_timing=True)
#         if "cuda" in str(device):
#             torch.cuda.synchronize()
#             start.record()
#         t0 = time.process_time()
#
#         assert self.num_parts >= 2
#
#         for idx_val in range(len(self.value_functions)):
#             # 构造linprog的参数
#             # 不等式参数<=
#             self.compute_ub(self.value_functions[idx_val])
#             # 等式参数
#             self.compute_eq(self.value_functions[idx_val])
#
#             # 构造c
#             c = np.zeros(self.num_parts)
#             c = np.append(c, 1)
#
#             # 构造bounds
#             bound = (-self.num_parts, self.num_parts)
#             self.bounds = [bound for _ in range(self.num_parts)]
#             # 认为e的值域取[v(空集), inf)，即考虑了LC定义中S=空集的情况所对应的不等式
#             self.bounds.append((0, +np.inf))
#
#             # 做线性回归，在此之前需要保证A_ub, b_ub等被计算出了，需要注意bound的设置不能是默认的
#             ans = opt.linprog(c, self.A_ub, self.b_ub, self.A_eq, self.b_eq, method='highs', bounds=self.bounds)
#
#             if ans["success"] == False:
#                 raise ValueError(ans["message"])
#
#             # 判断用该采样计算出的分数是否在我们认为正常的范围内，如果不在则报错，认为采样不合理。
#             for i in range(self.num_parts):
#                 if ans['x'][i] == bound[0] or ans['x'][i] == bound[1]:
#                     raise ValueError("The linear program is unbounded.")
#
#             # 得到每一个参与方的分数
#             for i in range(self.num_parts):
#                 self.contributions[idx_val][i] = copy.deepcopy(ans['x'][i])
#
#             print(ans['x'][self.num_parts])
#
#         if "cuda" in str(device):
#             end.record()
#             torch.cuda.synchronize()
#             t_cal = (time.process_time() - t0) + start.elapsed_time(end)
#         else:
#             t_cal = time.process_time() - t0
#
#         return self.contributions.tolist(), t_cal

# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Create synthetic data
# np.random.seed(10)
# data = {
#     'removed client number': np.random.randint(1, 10, 100),
#     'y': np.random.rand(100),
#     'method': np.random.choice(['Method A', 'Method B', 'Method C'], 100),
#     'dataset': np.random.choice(['Dataset 1', 'Dataset 2'], 100),
#     'distribution': np.random.choice(['Dist 1', 'Dist 2'], 100)
# }
# df = pd.DataFrame(data)
#
# # Define markers and colors for the plot
# markers = ['o', 's', '^']
# colors = ['blue', 'orange', 'green']
# hue_order = ['Method A', 'Method B', 'Method C']
# datasets = ['Dataset 1', 'Dataset 2']
#
# # Plot with scale=1 (default size)
# fig1 = sns.catplot(data=df,
#                    x="removed client number",
#                    y='y', hue='method', kind='point',
#                    markers=markers,
#                    facecolor='none',
#                    scale=1,  # Default marker size
#                    ci=None,
#                    hue_order=hue_order,
#                    col="dataset",
#                    row="distribution",
#                    sharey=False,
#                    dodge=True,
#                    palette=colors,
#                    col_order=datasets,
#                    aspect=1.2,
#                    )
# fig1.fig.suptitle('Marker Scale = 1', y=1.03)
#
# # Plot with scale=2 (larger size)
# fig2 = sns.catplot(data=df,
#                    x="removed client number",
#                    y='y', hue='method', kind='point',
#                    markers=markers,
#                    facecolor='none',
#                    scale=2,  # Larger marker size
#                    ci=None,
#                    hue_order=hue_order,
#                    col="dataset",
#                    row="distribution",
#                    sharey=False,
#                    dodge=True,
#                    palette=colors,
#                    col_order=datasets,
#                    aspect=1.2,
#                    )
# fig2.fig.suptitle('Marker Scale = 2', y=1.03)
#
# plt.show()

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Sample data
tips = sns.load_dataset('tips')

# Create a catplot
g = sns.catplot(x='day', y='total_bill', hue='sex', markers=['o', '^'], data=tips, kind='point')

# Adjust point sizes and line widths
for ax in g.axes.flat:
    path_collections = [col for col in ax.collections if isinstance(col, matplotlib.collections.PathCollection)]
    for path_collection in path_collections:
        original_sizes = path_collection.get_sizes()
        # new_sizes = 10*original_sizes
        # path_collection.set_sizes(new_sizes)

    # for line in ax.get_lines():  # Iterate through lines in the plot
    #     # print(line.get_data())
    #     print(line.get_marker())  # Check if the line has markers (points)
    #     # line.set_marker('o')
    #     line.set_markersize(50)  # Adjust point size here
    #     # line.set_linewidth(1)  # Adjust line width here
    #     continue

plt.show()

# import seaborn as sns
# tips = sns.load_dataset("tips")
#
# ax = sns.pointplot(x="tip", y="day", data=tips, join=False)
# points = ax.collections[0]
# print(points)
# print(points.get_sizes().item())
# size = points.get_sizes().item()
# new_sizes = [size * 3 if name.get_text() == "Fri" else size for name in ax.get_yticklabels()]
# points.set_sizes(new_sizes)

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Sample data
# x = np.linspace(0, 10, 30)
# y = np.sin(x)
#
# # Create a line plot
# fig, ax = plt.subplots()
# line, = ax.plot(x, y, marker='o')  # 'o' is the marker style
#
# print(line.get_marker())
# # Set the marker size
# # line.set_markersize(100)  # Change 10 to your desired marker size
#
# plt.show()


# Create synthetic data
# np.random.seed(10)
# data = {
#     'removed client number': np.random.randint(1, 10, 100),
#     'y': np.random.rand(100),
#     'method': np.random.choice(['Method A', 'Method B', 'Method C'], 100),
#     'dataset': np.random.choice(['Dataset 1', 'Dataset 2'], 100),
#     'distribution': np.random.choice(['Dist 1', 'Dist 2'], 100)
# }
# df = pd.DataFrame(data)
#
# # Define colors for the plot
# colors = {'Method A': 'blue', 'Method B': 'orange', 'Method C': 'green'}
#
# # Create catplot
# g = sns.catplot(data=df,
#                 x="removed client number",
#                 y='y', hue='method', kind='point',
#                 ci=None,
#                 hue_order=['Method A', 'Method B', 'Method C'],
#                 col="dataset",
#                 row="distribution",
#                 sharey=False,
#                 dodge=True,
#                 palette=colors,
#                 aspect=1.2,
#                 )

# Iterate over the DataFrame and add scatter plot markers
# for i, row in df.iterrows():
#     plt.scatter(x=row['removed client number'],
#                 y=row['y'],
#                 color=colors[row['method']],
#                 s=100,  # Set marker size here
#                 alpha=0.6)

plt.show()

