import seaborn as sns
import pandas as pd
import pickle
import itertools
from matplotlib import pyplot as plt
import os
import matplotlib.collections

'''
This file is used to plot line figures associated with metric comparison. 
Please note that, the can only process one kind of data (attack or non-attack) each time, 
if you want to process one kind of data, please comment out the rest 
(i.e. the part between "metric plot with attack/non-attack data preprocessing starts" and "ends"). 
'''

def search_files_with_keywords(folder_path, keywords):
    matching_files = []
    for filename in os.listdir(folder_path):
        if all(keyword in filename for keyword in keywords):
            matching_files.append(filename)
    return matching_files

# ----------------- style setting begins ----------------------

# print("start")
# pd.set_option('display.max_columns', None)

colors = sns.color_palette("muted")
# colors = None
markers = ['o', 'v', '^', '<', '>', 's', 'd', 'p', 'h', 'H', '*', '+', 'x', 'D', '|']
style = "white"
sns.set_style(style)
# style=None
# sns.set(font_scale=1.5)
point_size_scale = 4

# ----------------- style setting ends ----------------------

# ----------------- config setting begins ----------------------
parameters_collection = {
    "adult quantity skew": {"num_epoch": 25, "lr": 0.001, "hidden_layer_size": 24, "batch_size": 64, "alpha": 0.365},
    "adult label skew": {"num_epoch": 25, "lr": 0.001, "hidden_layer_size": 24, "batch_size": 64, "alpha": 0.8},
    "bank quantity skew": {"num_epoch": 20, "lr": 0.001, "hidden_layer_size": 8, "batch_size": 64, "alpha": 0.35},
    "bank label skew": {"num_epoch": 20, "lr": 0.001, "hidden_layer_size": 8, "batch_size": 64, "alpha": 0.8},
    "tictactoe quantity skew": {"num_epoch": 80, "lr": 0.008, "hidden_layer_size": 16, "batch_size": 16, "alpha": 0.65},
    "tictactoe label skew": {"num_epoch": 80, "lr": 0.008, "hidden_layer_size": 16, "batch_size": 16, "alpha": 0.8},
    "dota2 quantity skew": {"num_epoch": 5, "lr": 0.001, "hidden_layer_size": 4, "batch_size": 128, "alpha": 0.4},
    "dota2 label skew": {"num_epoch": 5, "lr": 0.001, "hidden_layer_size": 4, "batch_size": 128, "alpha": 0.8},
}

# attack_methods = ["data replication", "random data generation", "low quality data", "label flip"]
attack_methods = ["label flip"]
distributions = ["quantity skew", "label skew"]
datasets = ["tictactoe", "adult", "bank", "dota2"]
models = ["TicTacToeMLP", "AdultMLP", "BankMLP", "Dota2MLP"]
hue_order = ["accuracy", "data_quantity", "gradient_similarity", "robust_volume"]
seeds = list(range(6694, 6704))
num_parts = 8
# accuracy 指标在 metrics 序列中的位置.
accuracy_index = 3
# ----------------- config setting ends ----------------------

# ------------------------- metric plot with non-attack data preprocessing starts --------------------------
plot_path = f"./figure/exp_result/final/metric_remove_client/"
filename = f"metric_remove_{point_size_scale}.pdf"

data = pd.read_csv('./result/exp_result/final/metric_remove.csv', sep=';')

x_lo, x_hi = 0, 0.625
data = data[(data["x"] >= x_lo) & (data["x"] <= x_hi)]
x_li = data["x"].unique()
for x in x_li:
    data.loc[(data["x"] == x), "removed client number"] = round(x * num_parts)
data["removed client number"] = data["removed client number"].astype(int)

data = data[data["remove topic"] == "best"]

for distribution, (dataset, model) in itertools.product(distributions, zip(datasets, models)):

    folder_path = f"./data/utility_cache/--topic effective --dataset {dataset} --model {model} --num_parts 8/"

    parameters = parameters_collection[f"{dataset} {distribution}"]
    for seed in seeds:
        searching_keywords = [distribution, f"--alpha {parameters['alpha']}",
                              f"--batch_size {parameters['batch_size']}", f"--num_epoch {parameters['num_epoch']}",
                              f"--lr {parameters['lr']}", f"--hidden_layer_size {parameters['hidden_layer_size']}",
                              f"--seed {seed}", "['f1', 'f1_macro', 'f1_micro', 'accuracy']", ]
        file_paths = search_files_with_keywords(folder_path, searching_keywords)
        # print(searching_keywords)
        # print(file_paths)
        assert len(file_paths) == 1
        file_path = file_paths[0]
        file_path = folder_path + file_path

        with open(file_path, "rb") as handle:
            cache = pickle.load(handle)

        for metric in hue_order:
            if len(data.loc[(data["distribution"] == distribution) & (data["dataset"] == dataset) & (
                    data["seed"] == seed) & (
                                    data["value function"] == metric) & (data["remove topic"] == "best") & (
                                    data["method"] == "ShapleyValue")]) == 0:
                continue
            else:
                # if distribution dataset metric 有数据，则将y改变成为accuracy
                retained_set = set(range(num_parts))
                for removed_client_number in range(round(x_lo * num_parts), round(x_hi * num_parts) + 1):
                    # 给y重新赋值
                    data.loc[(data["distribution"] == distribution) & (data["dataset"] == dataset) & (
                            data["removed client number"] == removed_client_number) & (data["seed"] == seed) & (
                                     data["value function"] == metric) & (data["remove topic"] == "best") & (
                                     data["method"] == "ShapleyValue"), "y"] = cache[str(retained_set)][accuracy_index]
                    num_removed_client: pd.Series = data.loc[
                        (data["distribution"] == distribution) & (data["dataset"] == dataset) & (
                                data["removed client number"] == removed_client_number) & (data["seed"] == seed) & (
                                data["value function"] == metric) & (data["remove topic"] == "best") & (
                                data["method"] == "ShapleyValue"), "num_removed_client"]
                    assert len(num_removed_client) == 1
                    num_removed_client = num_removed_client.tolist()[0]
                    retained_set.remove(num_removed_client)

# ------------------------- metric plot with non-attack data preprocessing ends --------------------------

# ------------------------- metric plot with attack data preprocessing starts --------------------------
# distribution = "label skew"
# attack_arg = 0.8
# plot_path = f"./figure/exp_result/final/metric_remove_client/"
# num_attack_clients = 2
# filename = f"metric_remove_robust_setting_{num_attack_clients}_{attack_arg}_label_flip_{point_size_scale}.pdf"
#
# data = pd.read_csv(f'result/exp_result/final/metric_remove_client_robust_setting_{num_attack_clients}_{attack_arg}.csv', sep=';')
#
# x_lo, x_hi = 0, 0.625
# data = data[(data["x"] >= x_lo) & (data["x"] <= x_hi)]
# x_li = data["x"].unique()
# for x in x_li:
#     data.loc[(data["x"] == x), "removed client number"] = round(x * num_parts)
# data["removed client number"] = data["removed client number"].astype(int)
#
# data = data[data["remove topic"] == "best"]
# for attack_method, (dataset, model) in itertools.product(attack_methods, zip(datasets, models)):
#
#     folder_path = f"./data/utility_cache/--topic robust --dataset {dataset} --model {model} --num_parts 8/"
#
#     parameters = parameters_collection[f"{dataset} {distribution}"]
#     for seed in seeds:
#         searching_keywords = [distribution, f"--alpha {parameters['alpha']}",
#                               f"--batch_size {parameters['batch_size']}", f"--num_epoch {parameters['num_epoch']}",
#                               f"--lr {parameters['lr']}", f"--hidden_layer_size {parameters['hidden_layer_size']}",
#                               f"--seed {seed}", f"{attack_method} {num_attack_clients}", f"--attack_arg {attack_arg}",
#                               "['f1', 'f1_macro', 'f1_micro', 'accuracy']", ]
#         file_paths = search_files_with_keywords(folder_path, searching_keywords)
#         # print(file_paths)
#         assert len(file_paths) == 1
#         file_path = file_paths[0]
#         file_path = folder_path + file_path
#
#         with open(file_path, "rb") as handle:
#             cache = pickle.load(handle)
#
#         for metric in hue_order:
#             if len(data.loc[(data["distribution"] == distribution) & (data["dataset"] == dataset) & (
#                     data["seed"] == seed) & (
#                     data["value function"] == metric) & (data["remove topic"] == "best") & (
#                     data["method"] == "ShapleyValue") & (data["attack_method"] == attack_method)]) == 0:
#                 continue
#             else:
#                 # if distribution dataset metric 有数据，则将y改变成为accuracy
#                 retained_set = set(range(num_parts))
#                 for removed_client_number in range(round(x_lo * num_parts), round(x_hi * num_parts) + 1):
#                     # 给y重新赋值
#                     data.loc[(data["distribution"] == distribution) & (data["dataset"] == dataset) & (
#                     data["seed"] == seed) & (
#                     data["value function"] == metric) & (data["remove topic"] == "best") & (
#                     data["method"] == "ShapleyValue") & (data["attack_method"] == attack_method) & (data["removed client number"] == removed_client_number), "y"] = cache[str(retained_set)][accuracy_index]
#                     num_removed_client: pd.Series = data.loc[
#                         (data["distribution"] == distribution) & (data["dataset"] == dataset) & (
#                                 data["seed"] == seed) & (
#                                 data["value function"] == metric) & (data["remove topic"] == "best") & (
#                                 data["method"] == "ShapleyValue") & (data["attack_method"] == attack_method)
#                         & (data["removed client number"] == removed_client_number), "num_removed_client"]
#                     # print(f"{distribution}{dataset}{seed}{metric}{attack_method}{num_removed_client}")
#                     assert len(num_removed_client) == 1
#                     num_removed_client = num_removed_client.tolist()[0]
#                     retained_set.remove(num_removed_client)
#
# data = data[data["attack_method"] == "label flip"]

# ------------------------- metric plot with attack data preprocessing ends --------------------------

# ------------------------- plot starts -----------------------

data['value function'] = data['value function'].map(
    {"accuracy": "Accuracy", "data_quantity": "DataQuantity", "gradient_similarity": "CosineGradient",
     "robust_volume": "RobustVolume"})

sns.set(font_scale=3.8)
sns.set_style(style)

fig = sns.catplot(data=data,
                  x="removed client number",
                  y='y', hue='value function',
                  kind='point',
                  markers=markers,
                  scale=1.5,
                  ci=None,
                  hue_order=["Accuracy", "DataQuantity", "CosineGradient", "RobustVolume"],
                  col="dataset",
                  row="distribution",     # non-attack
                  # row="attack_method",      # attack
                  # row_order=attack_methods,
                  sharey=False,
                  dodge=True,
                  palette=colors,
                  col_order=datasets,
                  aspect=1.2,
                  )
axes = fig.axes.flatten()

for ax in axes:
    path_collections = [col for col in ax.collections if isinstance(col, matplotlib.collections.PathCollection)]
    for path_collection in path_collections:
        original_sizes = path_collection.get_sizes()
        new_sizes = point_size_scale * original_sizes
        path_collection.set_sizes(new_sizes)

for axis in axes:
    axis.spines['top'].set_visible(True)
    axis.spines['right'].set_visible(True)
    axis.spines['bottom'].set_visible(True)
    axis.spines['left'].set_visible(True)

# sns.move_legend(fig, "upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, title=None, frameon=True,
#                 borderaxespad=0.)
sns.move_legend(fig, "upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, title=None, frameon=True,
                borderaxespad=0.)
fig.set_titles(row_template="{row_name}", col_template="{col_name}", pad=10)
fig.set(xlabel="removed clients")
fig.set_ylabels("accuracy")
plt.tight_layout(pad=0.2, w_pad=0.5)


fig.savefig(f'{plot_path}{filename}')
fig.figure.show()
plt.close("all")
