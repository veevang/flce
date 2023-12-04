import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import config
import itertools
'''
The file is the definition of class Plotter and contains the implementations of member functions. 
'''
colors = sns.color_palette("muted")
markers = ['o', 'v', '^', '<', '>', 's', 'd', 'p', 'h', 'H', '*', '+', 'x', 'D', '|']
style = "white"
sns.set(font_scale=1.5)


class Plotter:
    def __init__(self, start_date, num_try, is_final):
        # self.start_date = start_date
        # self.num_try = num_try
        if is_final:
            self.log_dir = f"./logs/final/"
            self.plot_dir = f"./figs/final/"
        else:
            self.log_dir = f"./logs/{start_date}/{num_try}/"
            self.plot_dir = f"./figs/{start_date}/{num_try}/"
        return

    # 同一个问题+data划分，不同数据集给一个图
    # 出两张图，第一张是model performance plot
    # 第二张图是volume plot
    def plot_effective(self, num_parts, model, value_function, hue_order, datasets, title):
        # sns.set(font_scale=3.8)
        # sns.set(font_scale=3.6)
        sns.set(font_scale=3.4)
        sns.set_style(style)
        # lim_value_function_list = ["r2 score"]
        x_lo, x_hi = 0, 0.625
        data_path = os.path.join(self.log_dir, "remove_client_data.csv")

        df = pd.read_csv(data_path, sep=';')
        df = df[(df["value function"] == value_function) &
                (df["#parts"] == num_parts) &
                (df['model'].str.contains(model)) &
                (df["x"] >= x_lo) &
                (df["x"] <= x_hi)]

        x_li = df["x"].unique()
        for x in x_li:
            df.loc[(df["x"] == x), "removed client number"] = round(x * config.num_parts)
        df["removed client number"] = df["removed client number"].astype(int)

        # metric2scheme = {
        #     "model performance": ["Individual", "LeaveOneOut", "ShapleyValue",
        #                           "LeastCore", "TMC-Shapley",
        #                           "MC-LeastCore", "Random", "TMC-GuidedSampling-Shapley", "MC-StructuredSampling-Shapley"],
        #     "volume": ["robust volume individual", "robust volume leave one out", "robust volume shapley value",
        #                "robust volume least core"]}

        # for metric in ["model performance", "volume"]:
        plot_path = f"{self.plot_dir}remove_client/"
        filename = f"{title}_{model}_{value_function}.pdf"
        # elif metric == "volume":
        #     filename = f"{metric}.pdf"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # tdf = df[(df["method"].isin(metric2scheme[metric]))]
        # if tdf.empty:
        #     continue

        df = df[df["remove topic"] == "best"]
        df["distribution | dataset"] = df["distribution"].astype(str) + " | " + df["dataset"].astype(str)
        col_order_separate = list(itertools.product(["quantity skew | ", "label skew | "], datasets))
        col_order = ["".join(col_order_separate[i]) for i in range(len(col_order_separate))]

        for full_name in config.method_abbr.keys():
            df.loc[(df["method"] == full_name), "method"] = config.method_abbr[full_name]

        for i, full_name in enumerate(hue_order):
            if full_name in config.method_abbr.keys():
                hue_order[i] = config.method_abbr[full_name]

        fig = sns.catplot(data=df,
                          # x='x',
                          x="removed client number",
                          y='y', hue='method', kind='point',
                          markers=markers,
                          facecolor='none',
                          scale=1.5,
                          # errwidth=1,
                          # capsize=.5,
                          ci=None,
                          hue_order=hue_order,
                          col="dataset",
                          row="distribution",
                          # row="remove topic",
                          sharey=False,
                          dodge=True,
                          palette=colors,
                          col_order=["tictactoe", "adult", "bank", "dota2"],
                          aspect=1.2,
                          )

        axes = fig.axes.flatten()

        # Add a box or frame around the axis
        for axis in axes:
            axis.spines['top'].set_visible(True)
            axis.spines['right'].set_visible(True)
            axis.spines['bottom'].set_visible(True)
            axis.spines['left'].set_visible(True)

        # 6 basic
        # sns.move_legend(fig, "upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, title=None, frameon=True,
        #                 borderaxespad=0.)
        # plt.legend(fontsize='14')
        sns.move_legend(fig, "upper center", bbox_to_anchor=(0.5, 1.1), ncol=5, title=None, frameon=True,
                        borderaxespad=0.)

        # legend = plt.legend()
        # for i in range(5):
        #     legend.get_texts()[i].set_fontsize(14)
        # fig.yaxis.set_major_formatter('{:.2f}'.format)
        fig.set_titles(row_template="{row_name}", col_template="{col_name}", pad=10)
        # fmt = ticker.StrMethodFormatter("{x:.2f}")
        # for i, j in itertools.product(range(1), range(4)):
        #     fig.axes[i][j].yaxis.set_major_formatter(fmt)
        # fig.set(xlabel="removed ratio")
        fig.set(xlabel="removed clients")
        fig.set_ylabels(value_function)
        plt.tight_layout(pad=0.25, w_pad=0.5)
        # if value_function in log_scale_value_function_list:
        #     fig.set(yscale="log")
        fig.savefig(f'{plot_path}{filename}')
        fig.figure.show()
        plt.close("all")
        return

    # attack
    def plot_robust(self, distribution, num_parts, model, value_function, hue_order, attack_client, attack_arg):
        sns.set(font_scale=2)
        cut_top = 0.3
        cut_bottom = -0.3
        aspect = 1.5
        sns.set_style(style)
        con_data_path = f"{self.log_dir}contribution.csv"
        if not os.path.exists(con_data_path):
            raise Exception("Contributions are not generated! ")
        attacked_con_data_path = f"{self.log_dir}attacked_contribution.csv"
        plot_path = f"{self.plot_dir}robust/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"{distribution}_{model}_{value_function}.pdf"

        # not attacked
        df = pd.read_csv(con_data_path, sep=';')
        # attacked
        adf = pd.read_csv(attacked_con_data_path, sep=';')

        # print("df", df)
        # print("adf", adf)

        # print(df.columns)
        # 筛选数据
        df = df[(df["method"] != "random")]
        df = df[(df["distribution"] == distribution) & (df["#parts"] == num_parts) & (
            (df['model'].str.contains(model))) & (df["value function"] == value_function) & (
                            df["client number"] == attack_client)]
        adf = adf[(adf["attack arg"] == attack_arg)]
        adf = adf[(adf["distribution"] == distribution) & (adf["#parts"] == num_parts) & (
            adf['model'].str.contains(model)) & (adf["value function"] == value_function) & (
                              adf["client number"] == attack_client)]

        # 数据处理，对于每个seed而言，他变成差分再除以的形式
        seed_li = adf["seed"].unique()
        method_li = adf["method"].unique()
        attack_method_li = adf["attack method"].unique()
        dataset_li = adf["dataset"].unique()
        # print(seed_li, method_li, attack_method_li, dataset_li)

        ndf = pd.DataFrame(
            # columns=["seed", "method", "contribution relative change", "attack method", "dataset", "method_abbr"],
            columns=["seed", "method", "contribution relative change", "attack method", "dataset"],
        )

        for dataset in dataset_li:
            for attack_method in attack_method_li:
                for method in method_li:
                    for seed in seed_li:
                        # 有时候会出现
                        if adf[(adf["dataset"] == dataset) & (adf["seed"] == seed) & (adf["method"] == method) & (
                                adf["attack method"] == attack_method)].empty or df[(df["dataset"] == dataset) & (
                                df["seed"] == seed) & (df["method"] == method)].empty:
                            continue
                        acon = float(adf.loc[(adf["dataset"] == dataset) & (adf["seed"] == seed) & (
                                adf["method"] == method) & (adf["attack method"] == attack_method), "contribution"])
                        ocon = float(df.loc[(df["dataset"] == dataset) & (df["seed"] == seed) & (
                                df["method"] == method), "contribution"])

                        # 认为是1e-5这样一个小值
                        if ocon == 0.0:
                            relative_change = (acon - ocon) / 1e-5
                        else:
                            relative_change = (acon - ocon) / abs(ocon)

                        # cut
                        if relative_change > cut_top:
                            relative_change = cut_top
                        elif relative_change < cut_bottom:
                            relative_change = cut_bottom

                        ndf = ndf.append(
                            {"seed": seed, "method": method, "attack method": attack_method, "dataset": dataset,
                             "contribution relative change": relative_change},
                            ignore_index=True,
                        )
        # ndf = ndf.reset_index(drop=True)
        # for full_name in config.method_abbr.keys():
        #     ndf.loc[(ndf["method"] == full_name), "method_abbr"] = config.method_abbr[full_name]
        for full_name in config.method_abbr.keys():
            ndf.loc[(ndf["method"] == full_name), "method"] = config.method_abbr[full_name]

        for i, full_name in enumerate(hue_order):
            if full_name in config.method_abbr.keys():
                hue_order[i] = config.method_abbr[full_name]
        # print(ndf)

        # start plotting
        # nrows = len(attack_method_li)
        # ncols = len(dataset_li)
        fig = sns.catplot(kind="box", data=ndf,
                          # y="method",
                          y="attack method",
                          x='contribution relative change',
                          hue="method",
                          hue_order = ["Individual", "LeaveOneOut", "ShapleyValue", "StructuredMC-Shapley", "LeastCore","MC-LeastCore"],
                          # order=hue_order,
                          # row="attack method",
                          col="dataset",
                          # row="attack method",
                          col_order=["tictactoe", "adult", "bank", "dota2"],
                          order=["data replication", "random data generation", "low quality data", "label flip", ],
                          # order=hue_order,
                          # ci=95,
                          sharey=True,
                          # palette=colors,
                          palette=sns.color_palette("muted"),
                          height=10,
                          aspect=3/5,
                          # width=2,
                          # dodge=True,
                          # legend=True,
                          # hue="method",
                          # hue_order=hue_order,
                          )
        # 画0参考线
        # axes = fig.axes
        # axes = axes.flatten()
        # for axis in axes:
        #     axis.axhline(0, c='r', ls='--', lw=1)
        new_labels = ['Repl.', 'Rand. Gen.', 'Low Qual.', 'Lbl. Flip']
        fig.set_yticklabels(new_labels)

        fig.set_titles(row_template="{row_name}",
                       col_template="{col_name}",
                       )
        # fig.set(xlabel="method")
        # fig.set(xscale="symlog")
        fig.set(xlim=(cut_bottom * 1.1, cut_top * 1.1))
        # fig.set(xlabel=f"relative contribution change ({value_function})")
        fig.set(xlabel=f"relative contribution change")
        # fig.set(xlabel=f"relative change")
        # fig.set(xticks=[-1, -0.5, 0, 0.5, 1])
        fig.set(xticks=[cut_bottom, 0, cut_top])

        # reference line
        axes = fig.axes.flatten()
        for i in range(len(axes)):
            axes[i].axvline(0, c='r', ls='--', lw=1)
        sns.move_legend(fig, "upper center", bbox_to_anchor=(.5, 1.08), ncol=6, title=None, frameon=True)
        plt.tight_layout(pad=0.2)
        fig.savefig(f'{plot_path}robust_{filename}')
        fig.figure.show()
        plt.close("all")
        return

    def plot_metric_robust(self, distribution, num_parts, model, attack_client, attack_arg):
        value_functions = ["accuracy", "data_quantity", "gradient_similarity", "robust_volume"]
        sns.set(font_scale=1.7)
        cut_top = 0.3
        cut_bottom = -0.3
        aspect = 5/3
        sns.set_style(style)
        con_data_path = f"{self.log_dir}metric_contribution.csv"
        if not os.path.exists(con_data_path):
            raise Exception("Contributions are not generated! ")
        attacked_con_data_path = f"{self.log_dir}metric_attacked_contribution_1_0.3.csv"
        plot_path = f"{self.plot_dir}robust/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"{distribution}_{model}_metric_1_0.3.pdf"

        # not attacked
        df = pd.read_csv(con_data_path, sep=';')
        # attacked
        adf = pd.read_csv(attacked_con_data_path, sep=';')
        # print(adf["value function"].unique())

        # print("df", df)
        # print("adf", adf)

        # print(df.columns)
        # 筛选数据
        # df = df[(df["method"] == "ShapleyValue")]
        df = df[(df["distribution"] == distribution) & (df["#parts"] == num_parts) & (
            (df['model'].str.contains(model))) & (df["method"] == "ShapleyValue") & (
                            df["client number"] == attack_client)]
        adf = adf[(adf["attack arg"] == attack_arg)]
        adf = adf[(adf["distribution"] == distribution) & (adf["#parts"] == num_parts) & (
            adf['model'].str.contains(model)) & (adf["method"] == "ShapleyValue") & (
                              adf["client number"] == attack_client)]

        # 数据处理，对于每个seed而言，他变成差分再除以的形式
        seed_li = adf["seed"].unique()
        # value_function = adf["value function"].unique()
        attack_method_li = adf["attack method"].unique()
        dataset_li = adf["dataset"].unique()
        # print(adf["value function"].unique())
        # print(seed_li, method_li, attack_method_li, dataset_li)

        ndf = pd.DataFrame(
            # columns=["seed", "method", "contribution relative change", "attack method", "dataset", "method_abbr"],
            columns=["seed", "value function", "contribution relative change", "attack method", "dataset"],
        )

        for dataset in dataset_li:
            for attack_method in attack_method_li:
                for value_function in value_functions:
                    for seed in seed_li:
                        # 有时候会出现
                        if adf[(adf["dataset"] == dataset) & (adf["seed"] == seed)
                               & (adf["attack method"] == attack_method) &
                               (adf["value function"] == value_function)].empty or \
                                df[(df["dataset"] == dataset) & (
                                df["seed"] == seed) & (df["method"] == "ShapleyValue") &
                                   (df["value function"] == value_function)].empty:
                            continue
                        acon = float(adf.loc[(adf["dataset"] == dataset) & (adf["seed"] == seed) & (
                                adf["value function"] == value_function) & (adf["attack method"] == attack_method), "contribution"])
                        ocon = float(df.loc[(df["dataset"] == dataset) & (df["seed"] == seed) & (
                                df["value function"] == value_function), "contribution"])

                        # 认为是1e-5这样一个小值
                        if ocon == 0.0:
                            relative_change = (acon - ocon) / 1e-5
                        else:
                            relative_change = (acon - ocon) / abs(ocon)

                        # cut
                        if relative_change > cut_top:
                            relative_change = cut_top
                        elif relative_change < cut_bottom:
                            relative_change = cut_bottom

                        ndf = ndf.append(
                            {"seed": seed, "value function": value_function, "attack method": attack_method, "dataset": dataset,
                             "contribution relative change": relative_change},
                            ignore_index=True,
                        )
        # ndf = ndf.reset_index(drop=True)
        # for full_name in config.method_abbr.keys():
        #     ndf.loc[(ndf["method"] == full_name), "method_abbr"] = config.method_abbr[full_name]
        # for full_name in config.method_abbr.keys():
        #     ndf.loc[(ndf["method"] == full_name), "method"] = config.method_abbr[full_name]

        # for i, full_name in enumerate(hue_order):
        #     if full_name in config.method_abbr.keys():
        #         hue_order[i] = config.method_abbr[full_name]
        # print(ndf)

        # start plotting
        # nrows = len(attack_method_li)
        # ncols = len(dataset_li)

        ndf['value function'] = ndf['value function'].map(
            {"accuracy": "Accuracy", "data_quantity": "DataQuantity", "gradient_similarity": "CosineGradient",
             "robust_volume": "RobustVolume"})
        ndf['attack method'] = ndf['attack method'].map({"random data generation":"RG", "label flip": "LF"})

        fig = sns.catplot(kind="box", data=ndf,
                          x="attack method",
                          y='contribution relative change',
                          hue="value function",
                          hue_order=["Accuracy", "DataQuantity", "CosineGradient", "RobustVolume"],
                          col="dataset",
                          col_order=["tictactoe", "adult", "bank", "dota2"],
                          # row_order=["data replication", "random data generation", "low quality data", "label flip", ],
                          # row_order=["random data generation", "label flip", ],
                          # ci=95,
                          order=['RG', 'LF'],
                          sharey=True,
                          # palette=colors,
                          palette=sns.color_palette("muted"),
                          width=0.5,
                          # height=10,
                          aspect=0.5,
                          # hue="method",
                          # hue_order=hue_order,
                          )
        # 画0参考线
        # axes = fig.axes
        # axes = axes.flatten()
        # for axis in axes:
        #     axis.axhline(0, c='r', ls='--', lw=1)
        fig.set_titles(row_template="{row_name}",
                       col_template="{col_name}",
                       )
        # fig.set(xlabel="method")
        # fig.set(xscale="symlog")
        fig.set(ylim=(cut_bottom * 1.1, cut_top * 1.1))
        fig.set(yticks=[cut_bottom, 0, cut_top])
        # fig.set(xlabel=f"relative contribution change ({value_function})")
        fig.set(ylabel=f"relative contribution change")
        fig.set(xlabel="attack method")


        # new_labels = ['Rand. Gen.', 'Flip Lbl.']
        # fig.set_xticklabels(new_labels)

        sns.move_legend(fig, "upper center", bbox_to_anchor=(.5, 1.15), ncol=6, title=None, frameon=True)
        plt.tight_layout(pad=0.2)
        # reference line
        axes = fig.axes.flatten()
        for i in range(len(axes)):
            axes[i].axhline(0, c='r', ls='--', lw=1)

        # sns.move_legend(fig, "lower center", bbox_to_anchor=(.45, 1), ncol=10, title=None, frameon=True)
        fig.savefig(f'{plot_path}robust_{filename}')
        fig.figure.show()
        plt.close("all")
        return

    def plot_metric_time(self):
        sns.set(font_scale=2.6)
        # aspect = 1.1
        sns.set_style(style)
        # sns.set(font='DejaVu Sans', font_scale=1.2)
        # sns.set_context(rc={"font.family": "serif", "font.serif": ["cmr10"]})

        plot_path = f"{self.plot_dir}efficiency/"
        data_path = f"{self.log_dir}metric_process_time.csv"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"metric_efficiency.pdf"

        df = pd.read_csv(data_path, sep=';')

        # 筛选一下数据
        # df = df[(df["#parts"] == num_parts) & (df["model"] == model) & (df["value function"] == value_function)]
        # df = df[(df["#parts"] == num_parts)]

        # 处理一下，使得每个individual为单位时间
        # ds_li = df["dataset"].unique()
        # for ds in ds_li:
        #     # 计算每个dataset的individual均值
        #     avg_indiv = np.average(df.loc[(df["dataset"] == ds) & (df["method"] == "individual"), "process time"])
        #     df.loc[df["dataset"] == ds, "process time"] /= avg_indiv
        df = df[(df["method"] != "random")]
        df = df[(df['distribution'] == 'label skew')]

        # for full_name in config.method_abbr.keys():
        #     df.loc[(df["method"] == full_name), "method"] = config.method_abbr[full_name]

        # for i, full_name in enumerate(hue_order):
        #     if full_name in config.method_abbr.keys():
        #         hue_order[i] = config.method_abbr[full_name]

        df['value functions'] = df['value functions'].map({"['accuracy']":"Accuracy", "['data_quantity']":"DataQuantity", "['gradient_similarity']":"CosineGradient", "['robust_volume']":"RobustVolume"})
        # 改一下
        g = sns.catplot(data=df,
                        x="dataset",
                        order=["tictactoe", "adult", "bank", "dota2"],
                        y='process time',
                        hue="value functions",
                        kind="bar",
                        # hue_order=["['accuracy']", "['data_quantity']", "['gradient_similarity']", "['robust_volume']"],
                        hue_order=["Accuracy", "DataQuantity", "CosineGradient", "RobustVolume"],
                        sharey=False,
                        palette=colors,
                        # aspect=aspect,
                        # legend="lower center",
                        )

        ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]

        # iterate through the axes containers
        for c in ax.containers:
            labels = [f'{(v.get_height()):.2E}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge',size="7")

        # plt.legend(loc="lower center", bbox_to_anchor=(.3, 1.15), ncol=10, title=None, frameon=True)
        # sns.move_legend(g, "upper center", ncol=10, title=None, frameon=True, bbox_to_anchor=(.5, 0.8),
        #                 )
        # g.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        g.set_titles(col_template="{col_name}")
        # g.set(ylabel="Process Time (s)")
        g.set(yscale="log")
        g.set_ylabels("Process Time (s)")
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, title=None, frameon=True,
                        borderaxespad=0.)
        # sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.1), ncol=10, title=None, frameon=True,
        #                 borderaxespad=0.)
        plt.tight_layout(pad=0.2)
        # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, .9), ncol=7, title=None, frameon=True)
        # fig.set(ylabel="normalized process time")

        g.savefig(f'{plot_path}{filename}')
        g.figure.show()
        plt.close(g.fig)
        return


    # process time
    # 不管topic
    def plot_time(self, hue_order):
        sns.set(font_scale=2.6)
        # aspect = 1.1
        sns.set_style(style)
        # sns.set(font='DejaVu Sans', font_scale=1.2)
        # sns.set_context(rc={"font.family": "serif", "font.serif": ["cmr10"]})

        plot_path = f"{self.plot_dir}efficiency/"
        data_path = f"{self.log_dir}process_time.csv"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"efficiency.pdf"

        df = pd.read_csv(data_path, sep=';')

        # 筛选一下数据
        # df = df[(df["#parts"] == num_parts) & (df["model"] == model) & (df["value function"] == value_function)]
        # df = df[(df["#parts"] == num_parts)]

        # 处理一下，使得每个individual为单位时间
        # ds_li = df["dataset"].unique()
        # for ds in ds_li:
        #     # 计算每个dataset的individual均值
        #     avg_indiv = np.average(df.loc[(df["dataset"] == ds) & (df["method"] == "individual"), "process time"])
        #     df.loc[df["dataset"] == ds, "process time"] /= avg_indiv
        df = df[(df["method"] != "random")]

        for full_name in config.method_abbr.keys():
            df.loc[(df["method"] == full_name), "method"] = config.method_abbr[full_name]

        for i, full_name in enumerate(hue_order):
            if full_name in config.method_abbr.keys():
                hue_order[i] = config.method_abbr[full_name]

        # 改一下
        g = sns.catplot(data=df,
                        x='distribution',
                        y='process time',
                        hue="method",
                        kind="bar",
                        hue_order=hue_order,
                        col="dataset",
                        col_order=["tictactoe", "adult", "bank", "dota2"],
                        sharey=False,
                        palette=colors,
                        # aspect=aspect,
                        # legend="lower center",
                        )


        # plt.legend(loc="lower center", bbox_to_anchor=(.3, 1.15), ncol=10, title=None, frameon=True)
        # sns.move_legend(g, "upper center", ncol=10, title=None, frameon=True, bbox_to_anchor=(.5, 0.8),
        #                 )
        # g.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        g.set_titles(col_template="{col_name}")
        g.set(ylabel="Process Time (s)")
        g.set(yscale="log")
        g.set_ylabels("Process Time (s)")
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, title=None, frameon=True, borderaxespad=0.)
        plt.tight_layout(pad=0.2)
        # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, .9), ncol=7, title=None, frameon=True)
        # fig.set(ylabel="normalized process time")

        g.savefig(f'{plot_path}{filename}')
        g.figure.show()
        plt.close(g.fig)
        return
