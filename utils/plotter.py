import copy
import os

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import config
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
            self.log_dir = f"./result/exp_result/final/"
            self.plot_dir = f"./figure/exp_result/final/"
        else:
            self.log_dir = f"./result/exp_result/{start_date}/{num_try}/"
            self.plot_dir = f"./figs/exp_results/{start_date}/{num_try}/"
        return

    # 同一个问题+data划分，不同数据集给一个图
    # 出两张图，第一张是model performance plot
    # 第二张图是volume plot
    def plot_effective(self, num_parts, model, value_function, hue_order, datasets, title, point_size_scale):
        # sns.set(font_scale=3.8)
        # sns.set(font_scale=3.6)
        sns.set(font_scale=3.4)
        sns.set_style(style)
        # lim_value_function_list = ["r2 score"]
        x_lo, x_hi = 0, 0.625
        if num_parts == 8:
            data_path = os.path.join(self.log_dir, "remove_client_data_8&14.csv")
        elif num_parts == 14:
            data_path = os.path.join(self.log_dir, os.path.join("supplementary", "remove_client_data_14.csv"))
        else:
            raise ValueError

        df = pd.read_csv(data_path, sep=';')
        df = df[(df["value function"] == value_function) &
                (df["#parts"] == num_parts) &
                (df['model'].str.contains(model)) &
                (df["x"] >= x_lo) &
                (df["x"] <= x_hi)]

        x_li = df["x"].unique()
        for x in x_li:
            df.loc[(df["x"] == x), "removed client number"] = round(x * num_parts)
        df["removed client number"] = df["removed client number"].astype(int)

        # metric2scheme = {
        #     "model performance": ["Individual", "LeaveOneOut", "ShapleyValue",
        #                           "LeastCore", "TMC-Shapley",
        #                           "MC-LeastCore", "Random", "TMC-GuidedSampling-Shapley", "MC-StructuredSampling-Shapley"],
        #     "volume": ["robust volume individual", "robust volume leave one out", "robust volume shapley value",
        #                "robust volume least core"]}

        # for metric in ["model performance", "volume"]:
        plot_path = f"{self.plot_dir}remove_client/"
        filename = f"{title}_{model}_{value_function}_{num_parts}.pdf"
        # elif metric == "volume":
        #     filename = f"{metric}.pdf"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # tdf = df[(df["method"].isin(metric2scheme[metric]))]
        # if tdf.empty:
        #     continue

        df = df[df["remove topic"] == "best"]
        # df["distribution | dataset"] = df["distribution"].astype(str) + " | " + df["dataset"].astype(str)
        # col_order_separate = list(itertools.product(["quantity skew | ", "label skew | "], datasets))
        # col_order = ["".join(col_order_separate[i]) for i in range(len(col_order_separate))]
        # print(col_order)

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
                          col_order=datasets,
                          aspect=1.2,
                          )

        axes = fig.axes.flatten()

        for ax in axes:
            path_collections = [col for col in ax.collections if isinstance(col, matplotlib.collections.PathCollection)]
            for path_collection in path_collections:
                original_sizes = path_collection.get_sizes()
                new_sizes = point_size_scale  * original_sizes
                path_collection.set_sizes(new_sizes)

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

    def plot_effective_all_num_parts(self, model, value_function, hue_order, datasets, title, point_size_scale):
        """
        Very genius code, please be careful if you want to modify it.
        :param model:
        :param value_function:
        :param hue_order:
        :param datasets:
        :param title:
        :param point_size_scale:
        :return:
        """
        # sns.set(font_scale=3.4)
        sns.set(font_scale=3.3)
        sns.set_style(style)
        x_lo, x_hi = 0, 0.5
        data_path = os.path.join(self.log_dir, "remove_client_data_8&14.csv")
        df = pd.read_csv(data_path, sep=';')

        # The dataframe is "incorrectly" changed just to make it easier to draw with catplot,
        # and then the title will be changed back to the correct one!
        # see "change titles back to the correct ones"!
        # df.loc[
        #     (df['#parts'] == 14) & (df['distribution'] == 'label skew') & (df['dataset'] == 'adult'), ['distribution',
        #                                                                                                'dataset']] = [
        #     'quantity skew', 'tictactoe']

        df = df[(df["value function"] == value_function) &
                (df['model'].str.contains(model)) &
                (df["x"] >= x_lo) &
                (df["x"]*df["#parts"] <= df["#parts"]/2 + 1)]

        df['removed client number'] = (df['x'] * df['#parts']).round()
        df["removed client number"] = df["removed client number"].astype(int)

        plot_path = f"{self.plot_dir}remove_client/"
        filename = f"{title}_{model}_{value_function}_8&14.pdf"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        df = df[df["remove topic"] == "best"]
        for full_name in config.method_abbr.keys():
            df.loc[(df["method"] == full_name), "method"] = config.method_abbr[full_name]

        for i, full_name in enumerate(hue_order):
            if full_name in config.method_abbr.keys():
                hue_order[i] = config.method_abbr[full_name]

        plot_label = '999'

        df.loc[
            (df['#parts'] == 8) & (df['distribution'] == 'quantity skew') & (df['dataset'] == 'creditcard'), ['#parts', 'distribution',
                                                                                                       'dataset']] = [
            plot_label, 'plot distribution', 'tictactoe']
        df.loc[
            (df['#parts'] == 8) & (df['distribution'] == 'label skew') & (df['dataset'] == 'creditcard'), ['#parts','distribution',
                                                                                                       'dataset']] = [
            plot_label, 'plot distribution', 'adult']
        df.loc[
            (df['#parts'] == 14) & (df['distribution'] == 'quantity skew') & (df['dataset'] == 'dota2'), ['#parts', 'distribution',
                                                                                                       'dataset']] = [
            plot_label, 'plot distribution', 'bank']
        df.loc[
            (df['#parts'] == 14) & (df['distribution'] == 'label skew') & (df['dataset'] == 'dota2'), ['#parts', 'distribution',
                                                                                                       'dataset']] = [
            plot_label, 'plot distribution', 'dota2']

        df["distribution | #parts"] = df["distribution"].astype(str) + " | " + df["#parts"].astype(int).astype(str)
        # col_order_separate = list(itertools.product(["quantity skew | ", "label skew | "], datasets))
        # col_order = ["".join(col_order_separate[i]) for i in range(len(col_order_separate))]
        # print(col_order)

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
                          row="distribution | #parts",
                          # row="remove topic",
                          sharex=False,
                          sharey=False,
                          dodge=True,
                          palette=colors,
                          # col_order=datasets,
                          col_order=['tictactoe', 'adult', 'bank', 'dota2'],
                          row_order=["quantity skew | 8", "label skew | 8", f"plot distribution | {plot_label}"],
                          aspect=1.2,
                          )



        axes = fig.axes.flatten()

        for ax in axes:
            path_collections = [col for col in ax.collections if isinstance(col, matplotlib.collections.PathCollection)]
            for path_collection in path_collections:
                original_sizes = path_collection.get_sizes()
                new_sizes = point_size_scale  * original_sizes
                path_collection.set_sizes(new_sizes)

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
        sns.move_legend(fig, "upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, title=None, frameon=True,
                        borderaxespad=0.)

        # print(fig.legend.legendHandles)
        for handle in fig.legend.legendHandles:
            original_sizes = handle.get_sizes()
            new_sizes = point_size_scale * original_sizes
            handle.set_sizes(new_sizes)

        # legend = plt.legend()
        # for i in range(5):
        #     legend.get_texts()[i].set_fontsize(14)
        # fig.yaxis.set_major_formatter('{:.2f}'.format)
        title_pad: int = 13
        fig.set_titles(row_template="{row_name}", col_template="{col_name}", pad=title_pad)
        # fig.set_titles(row_template="{row_name}", col_template="{col_name}", pad=30)

        # change titles to the correct ones!!!
        fig.axes[2, 0].set_title('quantity skew | 8 | creditcard', pad=title_pad)
        fig.axes[2, 1].set_title('label skew | 8 | creditcard', pad=title_pad)
        fig.axes[2, 2].set_title('quantity skew | 14 | dota2', pad=title_pad)
        fig.axes[2, 3].set_title('label skew | 14 | dota2', pad=title_pad)

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
        con_data_path = f"{self.log_dir}contribution_8.csv"
        if not os.path.exists(con_data_path):
            raise Exception("Contributions are not generated! ")
        attacked_con_data_path = f"{self.log_dir}attacked_contribution_1_0.3.csv"
        plot_path = f"{self.plot_dir}robust/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"{distribution}_{model}_{value_function}.pdf"

        # not attacked
        df = pd.read_csv(con_data_path, sep=';',engine="python")
        # attacked
        adf = pd.read_csv(attacked_con_data_path, sep=';',engine="python")

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
                          hue_order=["Individual", "LeaveOneOut", "ShapleyValue", "StructuredMC-Shapley", "LeastCore",
                                     "MC-LeastCore"],
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
                          aspect=3 / 5,
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

        axes = fig.axes.flatten()
        for axis in axes:
            axis.spines['top'].set_visible(True)
            axis.spines['right'].set_visible(True)
            axis.spines['bottom'].set_visible(True)
            axis.spines['left'].set_visible(True)

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

    def plot_robust_num_attack_clients(self, distribution, num_parts, model, value_function, hue_order, attack_client_coals,
                                  attack_arg, point_size_scale):
        sns.set(font_scale=3.5)
        cut_top = 0.3
        cut_bottom = -0.3
        aspect = 1.5
        sns.set_style(style)
        con_data_path = f"{self.log_dir}contribution.csv"
        if not os.path.exists(con_data_path):
            raise Exception("Contributions are not generated! ")
        attacked_con_data_paths = []
        attacked_con_data_paths.append(f"{self.log_dir}attacked_contribution.csv")
        attacked_con_data_paths.append(f"{self.log_dir}supplementary/attacked_contribution_2_0.3.csv")
        attacked_con_data_paths.append(f"{self.log_dir}supplementary/attacked_contribution_3_0.3.csv")
        plot_path = f"{self.plot_dir}robust/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"{distribution}_{model}_{value_function}.pdf"

        # not attacked
        df = pd.read_csv(con_data_path, sep=';')
        # attacked
        adfs = []
        for i in range(3):
            adfs.append(pd.read_csv(attacked_con_data_paths[i], sep=';'))
        # print("df", df)
        # print("adf", adf)
        adfs[0]['num attack clients'] = 1
        adf = pd.concat(adfs, axis=0)
        adf = adf.reset_index(drop=True)

        # print(df.columns)
        # 筛选数据
        df = df[(df["method"] != "random")]
        df = df[(df["distribution"] == distribution) & (df["#parts"] == num_parts) & (
            (df['model'].str.contains(model))) & (df["value function"] == value_function)]
        adf = adf[(adf["attack arg"] == attack_arg)]
        adf = adf[(adf["distribution"] == distribution) & (adf["#parts"] == num_parts) & (
            adf['model'].str.contains(model)) & (adf["value function"] == value_function)]

        # 数据处理，对于每个seed而言，他变成差分再除以的形式
        seed_li = adf["seed"].unique()
        # method_li = adf["method"].unique()
        method_li = ["Individual", "LeaveOneOut", "ShapleyValue", "LeastCore", "MC-StructuredSampling-Shapley", "MC-LeastCore"]
        # attack_method_li = adf["attack method"].unique()
        attack_method_li = ["random data generation", "label flip"]
        # dataset_li = adf["dataset"].unique()
        dataset_li = ["tictactoe", "adult", "bank", "dota2"]
        # print(seed_li, method_li, attack_method_li, dataset_li)

        ndf = pd.DataFrame(
            # columns=["seed", "method", "contribution relative change", "attack method", "dataset", "method_abbr"],
            columns=["seed", "method", "contribution relative change", "attack method", "dataset",
                     "num attack clients"],
        )
        for num_attack_clients in range(1, 4):
            for dataset in dataset_li:
                for attack_method in attack_method_li:
                    for method in method_li:
                        for seed in seed_li:
                            relative_changes = np.zeros(num_attack_clients)
                            for num_attack_client in range(num_attack_clients):
                                # 有时候会出现，为什么？？？
                                # if adf[(adf["dataset"] == dataset) & (adf["seed"] == seed) & (adf["method"] == method) & (adf["attack method"] == attack_method) & (adf["num attack clients"] == num_attack_clients) & (adf["client number"] == num_attack_client)].empty or df[(df["dataset"] == dataset) & (df["seed"] == seed) & (df["method"] == method) & (df["client number"] == num_attack_client)].empty:
                                #     continue

                                acon = float(adf.loc[(adf["dataset"] == dataset) & (adf["seed"] == seed) & (
                                        adf["method"] == method) & (adf["attack method"] == attack_method) & (
                                                                 adf["num attack clients"] == num_attack_clients) & (
                                                                 adf[
                                                                     "client number"] == num_attack_client), "contribution"])
                                ocon = float(df.loc[(df["dataset"] == dataset) & (df["seed"] == seed) & (
                                        df["method"] == method) & (df[
                                                                       "client number"] == num_attack_client), "contribution"])

                                # 认为是1e-5这样一个小值
                                if ocon == 0.0:
                                    relative_changes[num_attack_client] = (acon - ocon) / 1e-5
                                else:
                                    relative_changes[num_attack_client] = (acon - ocon) / abs(ocon)

                                # cut
                                if relative_changes[num_attack_client] > cut_top:
                                    relative_changes[num_attack_client] = cut_top
                                elif relative_changes[num_attack_client] < cut_bottom:
                                    relative_changes[num_attack_client] = cut_bottom

                            relative_change = np.average(relative_changes)
                            ndf = ndf.append(
                                {"seed": seed, "method": method, "attack method": attack_method, "dataset": dataset,
                                 "contribution relative change": relative_change,
                                 "num attack clients": num_attack_clients},
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

        attack_method_abbreviation = {"random data generation": "RG", "label flip": "LF"}
        # abbreviation
        for full_name in ["random data generation", "label flip"]:
            ndf.loc[(ndf["attack method"] == full_name), "attack method"] = attack_method_abbreviation[full_name]

        fig = sns.catplot(data=ndf,
                          # x='x',
                          x="num attack clients",
                          y='contribution relative change', hue='method', kind='point',
                          markers=markers,
                          facecolor='none',
                          scale=1.5,
                          # errwidth=1,
                          capsize=.1,
                          # ci=None,
                          hue_order=hue_order,
                          col="dataset",
                          row="attack method",
                          # row="remove topic",
                          sharey=False,
                          dodge=.5,
                          palette=colors,
                          col_order=["tictactoe", "adult", "bank", "dota2"],
                          # row_order=["random data generation", "label flip", ],
                          row_order=["RG", "LF", ],
                          aspect=1.2,
                          )
        axes = fig.axes.flatten()

        for axis in axes:
            axis.spines['top'].set_visible(True)
            axis.spines['right'].set_visible(True)
            axis.spines['bottom'].set_visible(True)
            axis.spines['left'].set_visible(True)

        for ax in axes:
            path_collections = [col for col in ax.collections if isinstance(col, matplotlib.collections.PathCollection)]
            for path_collection in path_collections:
                original_sizes = path_collection.get_sizes()
                new_sizes = point_size_scale * original_sizes
                path_collection.set_sizes(new_sizes)

        for handle in fig.legend.legendHandles:
            original_sizes = handle.get_sizes()
            new_sizes = point_size_scale * original_sizes
            handle.set_sizes(new_sizes)

        # 画0参考线
        # axes = fig.axes
        # axes = axes.flatten()
        # for axis in axes:
        #     axis.axhline(0, c='r', ls='--', lw=1)
        # new_labels = ['Repl.', 'Rand. Gen.', 'Low Qual.', 'Lbl. Flip']
        # fig.set_yticklabels(new_labels)

        fig.set_titles(row_template="{row_name}",
                       col_template="{col_name}",
                       )
        # fig.set(xlabel="method")
        # fig.set(xscale="symlog")
        # fig.set(xlim=(cut_bottom * 1.1, cut_top * 1.1))
        # fig.set(xlabel=f"relative contribution change ({value_function})")
        fig.set_ylabels("RelConCh")
        # fig.set(xlabel=f"relative change")
        # fig.set(xticks=[-1, -0.5, 0, 0.5, 1])
        # fig.set(xticks=[cut_bottom, 0, cut_top])

        # reference line
        axes = fig.axes.flatten()
        for i in range(len(axes)):
            axes[i].axhline(0, c='r', ls='--', lw=1)
        sns.move_legend(fig, "upper center", bbox_to_anchor=(.5, 1.2), ncol=3, title=None, frameon=True)
        plt.tight_layout(pad=0.2)
        fig.savefig(f'{plot_path}supplementary_robust_{filename}')
        fig.figure.show()
        plt.close("all")
        return

    def plot_robust_attack_arg(self, distribution, num_parts, model, value_function, hue_order, num_attack_client,
                                  attack_args, point_size_scale):
        sns.set(font_scale=3.5)
        ###????!!!!!!!!!!!!!!!!!!!!
        cut_top = max(attack_args)
        cut_bottom = -cut_top
        aspect = 1.5
        num_attack_clients = 1
        sns.set_style(style)
        con_data_path = f"{self.log_dir}contribution.csv"
        if not os.path.exists(con_data_path):
            raise Exception("Contributions are not generated! ")
        attacked_con_data_paths = []
        attacked_con_data_paths.append(f"{self.log_dir}supplementary/attacked_contribution_1_{attack_args[0]}.csv")
        attacked_con_data_paths.append(f"{self.log_dir}supplementary/attacked_contribution_1_{attack_args[1]}.csv")
        attacked_con_data_paths.append(f"{self.log_dir}supplementary/attacked_contribution_1_{attack_args[2]}.csv")
        attacked_con_data_paths.append(f"{self.log_dir}supplementary/attacked_contribution_1_{attack_args[3]}.csv")
        plot_path = f"{self.plot_dir}robust/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        filename = f"{distribution}_{model}_{value_function}.pdf"

        # not attacked
        df = pd.read_csv(con_data_path, sep=';')
        # attacked
        adfs = []
        for i in range(4):
            adfs.append(pd.read_csv(attacked_con_data_paths[i], sep=';'))

        adfs[1]['num attack clients'] = 1
        adf = pd.concat(adfs, axis=0)
        adf = adf.reset_index(drop=True)

        # print(df.columns)
        # 筛选数据
        df = df[(df["method"] != "random")]
        df = df[(df["distribution"] == distribution) & (df["#parts"] == num_parts) & (
            (df['model'].str.contains(model))) & (df["value function"] == value_function)]
        adf = adf[(adf["num attack clients"] == num_attack_clients)]
        adf = adf[(adf["distribution"] == distribution) & (adf["#parts"] == num_parts) & (
            adf['model'].str.contains(model)) & (adf["value function"] == value_function)]

        # 数据处理，对于每个seed而言，他变成差分再除以的形式
        seed_li = adf["seed"].unique()
        # method_li = adf["method"].unique()
        method_li = ["Individual", "LeaveOneOut", "ShapleyValue", "LeastCore", "MC-StructuredSampling-Shapley", "MC-LeastCore"]
        # attack_method_li = adf["attack method"].unique()
        attack_method_li = ["random data generation", "label flip"]
        dataset_li = ["tictactoe", "dota2"]
        # dataset_li = adf["dataset"].unique()
        # dataset_li = ["tictactoe", "adult", "bank", "dota2"]
        # print(seed_li, method_li, attack_method_li, dataset_li)

        ndf = pd.DataFrame(
            # columns=["seed", "method", "contribution relative change", "attack method", "dataset", "method_abbr"],
            columns=["seed", "method", "contribution relative change", "attack method", "dataset",
                     "attack arg"],
        )
        for dataset in dataset_li:
            for attack_method in attack_method_li:
                for method in method_li:
                    for seed in seed_li:
                        for attack_arg in attack_args:
                            # 有时候会出现，为什么？？？
                            # if adf[(adf["dataset"] == dataset) & (adf["seed"] == seed) & (adf["method"] == method) & (adf["attack method"] == attack_method) & (adf["num attack clients"] == num_attack_clients) & (adf["client number"] == num_attack_client) & (adf["attack arg"] == attack_arg)].empty or df[(df["dataset"] == dataset) & (df["seed"] == seed) & (df["method"] == method) & (df["client number"] == num_attack_client)].empty:
                            #     continue

                            acon = float(adf.loc[(adf["dataset"] == dataset) & (
                                    adf["seed"] == seed) & (
                                    adf["method"] == method) & (
                                    adf["attack method"] == attack_method) & (
                                    adf["num attack clients"] == num_attack_clients) & (
                                    adf["attack arg"] == attack_arg) & (
                                    adf["client number"] == num_attack_client), "contribution"])
                            ocon = float(df.loc[(df["dataset"] == dataset) & (
                                    df["seed"] == seed) & (
                                    df["method"] == method) & (
                                    df["client number"] == num_attack_client), "contribution"])

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
                             "contribution relative change": relative_change,
                             "attack arg": attack_arg},
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

        attack_method_abbreviation = {"random data generation": "RG", "label flip": "LF"}
        # abbreviation
        for full_name in ["random data generation", "label flip"]:
            ndf.loc[(ndf["attack method"] == full_name), "attack method"] = attack_method_abbreviation[full_name]

        ndf["attack method | dataset"] = ndf["attack method"].astype(str) + " | " + ndf["dataset"].astype(str)
        fig = sns.catplot(data=ndf,
                          # x='x',
                          x="attack arg",
                          y='contribution relative change', hue='method', kind='point',
                          markers=markers,
                          facecolor='none',
                          scale=1.5,
                          # errwidth=1,
                          capsize=.1,
                          # ci=None,
                          hue_order=hue_order,
                          col="attack method | dataset",
                          # col="dataset",
                          # row="attack method",
                          sharey=False,
                          dodge=.65,
                          palette=colors,
                          col_order=["RG | tictactoe", "RG | dota2", "LF | tictactoe", "LF | dota2" ],
                          # col_order=["tictactoe", "dota2"],
                          # row_order=["RG", "LF", ],
                          # row_order=[8],
                          aspect=1.2,
                          )
        axes = fig.axes.flatten()

        for axis in axes:
            axis.spines['top'].set_visible(True)
            axis.spines['right'].set_visible(True)
            axis.spines['bottom'].set_visible(True)
            axis.spines['left'].set_visible(True)

        for ax in axes:
            path_collections = [col for col in ax.collections if isinstance(col, matplotlib.collections.PathCollection)]
            for path_collection in path_collections:
                original_sizes = path_collection.get_sizes()
                new_sizes = point_size_scale * original_sizes
                path_collection.set_sizes(new_sizes)

        # print(fig.legend.legendHandles)
        for handle in fig.legend.legendHandles:
            original_sizes = handle.get_sizes()
            new_sizes = point_size_scale * original_sizes
            handle.set_sizes(new_sizes)

        # 画0参考线
        # axes = fig.axes
        # axes = axes.flatten()
        # for axis in axes:
        #     axis.axhline(0, c='r', ls='--', lw=1)
        # new_labels = ['Repl.', 'Rand. Gen.', 'Low Qual.', 'Lbl. Flip']
        # fig.set_yticklabels(new_labels)

        fig.set_titles(#row_template="{row_name}",
                       col_template="{col_name}",
                       )
        # fig.set(xlabel="method")
        # fig.set(xscale="symlog")
        # fig.set(xlim=(cut_bottom * 1.1, cut_top * 1.1))
        # fig.set(xlabel=f"relative contribution change ({value_function})")
        # fig.set(ylabel=f"RelConCh")
        fig.set_ylabels("RelConCh")
        # fig.set(xlabel=f"relative change")
        # fig.set(xticks=[-1, -0.5, 0, 0.5, 1])
        # fig.set(xticks=[cut_bottom, 0, cut_top])

        # reference line
        axes = fig.axes.flatten()
        for i in range(len(axes)):
            axes[i].axhline(0, c='r', ls='--', lw=1)
        sns.move_legend(fig, "upper center", bbox_to_anchor=(.5, 1.4), ncol=3, title=None, frameon=True)
        plt.tight_layout(pad=0.2)
        fig.savefig(f'{plot_path}robustness_vs_attack_arg_{filename}')
        fig.figure.show()
        plt.close("all")
        return

    def plot_metric_robust(self, distribution, num_parts, model, attack_client, attack_arg):
        value_functions = ["accuracy", "data_quantity", "gradient_similarity", "robust_volume"]
        sns.set(font_scale=1.7)
        cut_top = 0.3
        cut_bottom = -0.3
        aspect = 5 / 3
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
                                adf["value function"] == value_function) & (
                                                         adf["attack method"] == attack_method), "contribution"])
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
                            {"seed": seed, "value function": value_function, "attack method": attack_method,
                             "dataset": dataset,
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
        ndf['attack method'] = ndf['attack method'].map({"random data generation": "RG", "label flip": "LF"})

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

        axes = fig.axes.flatten()

        for axis in axes:
            axis.spines['top'].set_visible(True)
            axis.spines['right'].set_visible(True)
            axis.spines['bottom'].set_visible(True)
            axis.spines['left'].set_visible(True)

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
        df = df[(df["method"] == "ShapleyValue")]
        df = df[(df['distribution'] == 'label skew')]
        # df['process time'] = pd.to_numeric(df['column_name'], errors='raise')

        # for full_name in config.method_abbr.keys():
        #     df.loc[(df["method"] == full_name), "method"] = config.method_abbr[full_name]

        # for i, full_name in enumerate(hue_order):
        #     if full_name in config.method_abbr.keys():
        #         hue_order[i] = config.method_abbr[full_name]

        df['value functions'] = df['value functions'].map(
            {"['accuracy']": "Accuracy", "['data_quantity']": "DataQuantity",
             "['gradient_similarity']": "CosineGradient", "['robust_volume']": "RobustVolume"})
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
            ax.bar_label(c, labels=labels, label_type='edge', size="7")

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
        ### begin 2024-2-24
        # ax = g.facet_axis(0, 3)  # or ax = g.axes.flat[0]
        # for c in ax.containers:
        #     labels = [f'{(v.get_height()):.2E}' for v in c]
        #     ax.bar_label(c, labels=labels, label_type='edge', size="7")
        ### end 2024-2-24

        # plt.legend(loc="lower center", bbox_to_anchor=(.3, 1.15), ncol=10, title=None, frameon=True)
        # sns.move_legend(g, "upper center", ncol=10, title=None, frameon=True, bbox_to_anchor=(.5, 0.8),
        #                 )
        # g.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
        g.set_titles(col_template="{col_name}")
        g.set(ylabel="Process Time (s)")
        g.set(yscale="log")
        g.set_ylabels("Process Time (s)")
        sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, title=None, frameon=True,
                        borderaxespad=0.)
        plt.tight_layout(pad=0.2)
        # sns.move_legend(g, "lower center", bbox_to_anchor=(.5, .9), ncol=7, title=None, frameon=True)
        # fig.set(ylabel="normalized process time")

        g.savefig(f'{plot_path}{filename}')
        g.figure.show()
        plt.close(g.fig)
        return
