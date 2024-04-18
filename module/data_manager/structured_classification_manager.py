import os
import numpy as np
import pandas as pd
import sklearn.utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from sklearn import preprocessing
from module.data_manager.manager import DataManager
from typing import Union, Tuple
import scipy.stats
from abc import abstractmethod


# database citation

class StructuredClassificationManager(DataManager):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.data = None
        self.X_categorical_fields = None
        self.X_categories = None
        self.X_numerical_fields = None
        self.y_field = None

    @abstractmethod
    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        pass

    # 恶意参与方：攻击对抗
    def flip_y_train(self, parts: set, random_seed, ratio=1):
        assert type(parts) is set
        np.random.seed(random_seed)
        # 1->0, 0->1
        for client in parts:
            flip = np.array([False for _ in range(len(self.y_train_parts[client]))])
            flip[:round(ratio * len(self.y_train_parts[client]))] = True
            np.random.shuffle(flip)
            self.y_train_parts[client][flip] = torch.tensor(1) - self.y_train_parts[client][flip]
        return

    # 恶意参与方：随机生成
    # 在一个参与方上生成一些随机数据
    def randomly_generate_data(self, clients, ratio, seed):
        num_categorical_columns = sum(self.num_each_categorical_field())
        num_numerical_fields = len(self.X_numerical_fields)

        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        for client in clients:
            num_rows: int = round(ratio * len(self.y_train_parts[client]))
            X = np.zeros(shape=(num_rows, num_categorical_columns + num_numerical_fields))
            y = np.zeros(shape=num_rows)

            for row in X:
                # categorical是通过OneHotEncoder来产生的，比如说一个有3个元素的枚举类，那么其编码后对应的列为三列，
                # 这个枚举类中的三个元素分别对应这三列取值为[1, 0, 0]，[0, 1, 0]，[0, 0, 1]

                # categorical左侧的界从0开始。
                categorical_left_bound = 0
                # 对于每一个分类categorical
                for num_this_categorical in self.num_each_categorical_field():
                    # 考虑这个categorical的哪一个enum值变为1
                    rand = np.random.randint(0, num_this_categorical)
                    row[rand + categorical_left_bound] = 1
                    # 维护categorical的左界
                    categorical_left_bound += num_this_categorical

                # 连续continuous
                for i in range(num_numerical_fields):
                    row[num_categorical_columns + i] = rng.standard_normal(1)

            for i in range(num_rows):
                y[i] = np.random.randint(0, 2)

            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)

            self.X_train_parts[client] = torch.cat([self.X_train_parts[client], X])
            self.y_train_parts[client] = torch.cat([self.y_train_parts[client], y])

        return

    # 如果dataframe = none，那么就编码对象自己读取的data，并且储存到
    def _encode(self, df: pd.DataFrame = None):
        if df is None:
            df = self.data
            encode_self_dataframe = True
        else:
            # 处理self.data
            encode_self_dataframe = False

        # 编码X
        X_categorical_encoder = OneHotEncoder(categories=self.X_categories)
        X_categorical = X_categorical_encoder.fit_transform(df[self.X_categorical_fields]).toarray()

        # 把numerical的列量化为标准分
        if len(self.X_numerical_fields) != 0:
            X_numerical = preprocessing.scale(df[self.X_numerical_fields].to_numpy(float))
        else:
            X_numerical = np.array(df[self.X_numerical_fields])

        X = np.concatenate((X_categorical, X_numerical), axis=1)

        # 编码y
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(df[self.y_field])

        # 如果编码的是自己的数据，需要shuffle一下
        if encode_self_dataframe:
            np.random.seed(666)
            shuffle_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffle_indices)
            X = X[shuffle_indices]
            y = y[shuffle_indices]

        # 转变为tensor
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # 如果编码的是自己的数据，那么要赋值给自己的X，y变量
        if encode_self_dataframe:
            self.X = X
            self.y = y

        return X, y

    def num_each_categorical_field(self):
        # print(num_each_categorical_field)
        return [len(key) for key in self.X_categories]

    # use dirichlet distribution to create non-iid distributions
    def non_iid_split(self, num_parts, alpha, random_state):
        # 先将全数据集分割为0与1
        labels = set()
        for label in self.y_train:
            if str(label) not in labels:
                labels.add(str(label))
        # 构造一个labels集合

        X_train_label_sorted = dict(list())
        y_train_label_sorted = dict(list())

        # 让label不同的分开
        for i in range(len(self.y_train)):
            if str(self.y_train[i]) not in X_train_label_sorted.keys():
                X_train_label_sorted[str(self.y_train[i])] = []
                y_train_label_sorted[str(self.y_train[i])] = []
            X_train_label_sorted[str(self.y_train[i])].append(self.X_train[i])
            y_train_label_sorted[str(self.y_train[i])].append(self.y_train[i])

        for key in X_train_label_sorted.keys():
            X_train_label_sorted[key] = torch.stack(X_train_label_sorted[key])
            y_train_label_sorted[key] = torch.stack(y_train_label_sorted[key])

        # 每一组数据按照狄利克雷分布分成若干参与方
        list_of_ratios = scipy.stats.dirichlet.rvs(alpha, size=len(X_train_label_sorted), random_state=random_state)
        ratios = dict()
        index = 0
        for key in X_train_label_sorted.keys():
            ratios[key] = list_of_ratios[index]
            index += 1

        self.X_train_parts = []
        self.y_train_parts = []
        lo_ratios = dict()
        for key in X_train_label_sorted.keys():
            lo_ratios[key] = 0

        # 对于每一个参与方而言
        for i in range(num_parts):
            # 划定每个标签的数据里，哪些数据对应的是这个参与方？
            X_train_this_client = []
            y_train_this_client = []
            # 必须保证每个参与方里既有0又有1，这样一些模型才可以正常fit
            for label in X_train_label_sorted.keys():
                n = len(X_train_label_sorted[label])
                X_train_this_client.extend(X_train_label_sorted[label][
                                           int(lo_ratios[label] * n):int((lo_ratios[label] + ratios[label][i]) * n)])
                y_train_this_client.extend(y_train_label_sorted[label][
                                           int(lo_ratios[label] * n):int((lo_ratios[label] + ratios[label][i]) * n)])
                lo_ratios[label] += ratios[label][i]

            X_train_this_client = torch.stack(X_train_this_client)
            y_train_this_client = torch.stack(y_train_this_client)
            self.X_train_parts.append(X_train_this_client)
            self.y_train_parts.append(y_train_this_client)

        return self.X_train_parts, self.y_train_parts


class Adult(StructuredClassificationManager):
    def __init__(self):
        super().__init__(name="Adult")
        self.task = "Classification"
        return

    # 105列，去除无效列后有30162行
    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        fields = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income'
        ]

        self.data = pd.read_csv('data/raw/adult/adult.data',
                                engine='python',
                                names=fields,
                                sep=', ',
                                na_values="?",  # na_values指的是，当碰到"?"时将其替换为nan
                                nrows=nrows)
        self.data = sklearn.utils.shuffle(self.data, random_state=shuffle_seed)

        # 预处理，处理未知值，将上一步得到的含NaN的行都删除
        self.data.dropna(axis=0, how='any', inplace=True)

        # X列，按照X的性质分为两类，分类与连续
        # X为分类列
        self.X_categorical_fields = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country'
        ]
        workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                     'Without-pay', 'Never-worked']
        education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                     '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
        marital_status = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                          'Married-spouse-absent', 'Married-AF-spouse']
        occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                      'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                      'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
        relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
        race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
        sex = ['Female', 'Male']
        native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                          'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                          'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                          'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                          'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                          'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
        self.X_categories = [workclass, education, marital_status, occupation, relationship, race, sex, native_country]

        # X为连续的
        self.X_numerical_fields = [
            'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
        ]

        self.y_field = "income"

        self._encode()

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)

        if cuda:
            self._data_to_cuda()

        return self.X_train, self.y_train, self.X_test, self.y_test


class Bank(StructuredClassificationManager):
    def __init__(self):
        super().__init__(name="Bank")
        self.data = None
        return

    # 51列，去除无效行后有45211行
    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None) -> Union[Tuple, Tuple, Tuple, Tuple]:
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data = pd.read_csv('data/raw/bank/bank-full.csv',
                                engine='python',
                                sep=';',
                                nrows=nrows)
        self.data = sklearn.utils.shuffle(self.data, random_state=shuffle_seed)
        # 预处理，处理未知值，将上一步得到的含NaN的行都删除
        self.data.dropna(axis=0, how='any', inplace=True)

        # X列，按照X的性质分为两类，分类与连续
        # X为分类列
        self.X_categorical_fields = [
            'job', 'marital', 'education', 'default',
            'housing', 'loan', 'contact', 'month', 'poutcome'
        ]

        job = ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
               "blue-collar", "self-employed", "retired", "technician", "services"]
        marital = ["married", "divorced", "single"]
        education = ["unknown", "secondary", "primary", "tertiary"]
        default = ["yes", "no"]
        housing = ["yes", "no"]
        loan = ["yes", "no"]
        contact = ["unknown", "telephone", "cellular"]
        month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        poutcome = ["unknown", "other", "failure", "success"]
        self.X_categories = [job, marital, education, default, housing, loan, contact, month, poutcome]

        # X为连续的
        self.X_numerical_fields = [
            'age', 'balance', 'day', 'duration', 'campaign', 'pdays',
            'previous'
        ]

        self.y_field = "y"

        self._encode()

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)

        return self.X_train, self.y_train, self.X_test, self.y_test


class TicTacToe(StructuredClassificationManager):
    def __init__(self):
        super().__init__(name="TicTacToe")
        self.task = "Classification"
        return

    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        fields = [
            "top-left-square",
            "top-middle-square",
            "top-right-square",
            "middle-left-square",
            "middle-middle-square",
            "middle-right-square",
            "bottom-left-square",
            "bottom-middle-square",
            "bottom-right-square",
            "class",
        ]

        self.data = pd.read_csv('data/raw/tic+tac+toe+endgame/tic-tac-toe.data',
                                names=fields,
                                sep=',',
                                na_values="false",  # na_values指的是，当碰到"?"时将其替换为nan
                                nrows=nrows)
        self.data = sklearn.utils.shuffle(self.data, random_state=shuffle_seed)

        # 预处理，处理未知值，将上一步得到的含NaN的行都删除
        self.data.dropna(axis=0, how='any', inplace=True)

        # X列，按照X的性质分为两类，分类与连续
        # X为分类列
        self.X_categorical_fields = [
            "top-left-square",
            "top-middle-square",
            "top-right-square",
            "middle-left-square",
            "middle-middle-square",
            "middle-right-square",
            "bottom-left-square",
            "bottom-middle-square",
            "bottom-right-square",
        ]

        self.X_categories = [["x", "o", "b"] for _ in range(9)]

        # X为连续的
        self.X_numerical_fields = []

        self.y_field = "class"

        self._encode()

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)

        if cuda:
            self._data_to_cuda()

        return self.X_train, self.y_train, self.X_test, self.y_test


class Dota2(StructuredClassificationManager):
    def __init__(self):
        super().__init__(name="Dota2")
        self.task = "Classification"
        return

    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        fields = [
                     'win', 'cluster', 'mode', 'type',
                 ] + [f"hero{i}" for i in range(1, 114)]

        self.data = pd.read_csv('data/raw/dota2+games+results/dota2.csv',
                                engine='python',
                                sep=',',
                                names=fields,
                                na_values="false",  # na_values指的是，当碰到"?"时将其替换为nan
                                nrows=nrows)
        self.data = sklearn.utils.shuffle(self.data, random_state=shuffle_seed)

        # 预处理，处理未知值，将上一步得到的含NaN的行都删除
        self.data.dropna(axis=0, how='any', inplace=True)

        # X列，按照X的性质分为两类，分类与连续
        # X为分类列
        self.X_categorical_fields = [
            'cluster', 'mode', 'type',
        ]
        cluster = sorted(self.data["cluster"].unique())
        mode = sorted(self.data["mode"].unique())
        type = sorted(self.data["type"].unique())
        self.X_categories = [cluster, mode, type]

        # X为连续的
        self.X_numerical_fields = [f"hero{i}" for i in range(1, 114)]

        self.y_field = "win"

        self._encode()

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)

        if cuda:
            self._data_to_cuda()

        return self.X_train, self.y_train, self.X_test, self.y_test

    def _encode(self, df: pd.DataFrame = None):
        if df is None:
            df = self.data
            encode_self_dataframe = True
        else:
            # 处理self.data
            encode_self_dataframe = False

        # 编码X
        X_categorical_encoder = OneHotEncoder(categories=self.X_categories)
        X_categorical = X_categorical_encoder.fit_transform(df[self.X_categorical_fields]).toarray()

        # 把numerical的列量化为标准分
        if len(self.X_numerical_fields) != 0:
            self.X_numerical_means = df[self.X_numerical_fields].to_numpy(float).mean(axis=0)
            self.X_numerical_stds = df[self.X_numerical_fields].to_numpy(float).std(axis=0)
            X_numerical = preprocessing.scale(df[self.X_numerical_fields].to_numpy(float))
        else:
            X_numerical = np.array(df[self.X_numerical_fields])

        X = np.concatenate((X_categorical, X_numerical), axis=1)

        # 编码y
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(df[self.y_field])

        # 如果编码的是自己的数据，需要shuffle一下
        if encode_self_dataframe:
            np.random.seed(666)
            shuffle_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffle_indices)
            X = X[shuffle_indices]
            y = y[shuffle_indices]

        # 转变为tensor
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        # 如果编码的是自己的数据，那么要赋值给自己的X，y变量
        if encode_self_dataframe:
            self.X = X
            self.y = y

        return X, y

    # dota2的编码需要重写!
    def randomly_generate_data(self, clients, ratio, seed):
        num_categorical_columns = sum(self.num_each_categorical_field())
        num_numerical_fields = len(self.X_numerical_fields)
        np.random.seed(seed)

        for client in clients:
            num_rows: int = round(ratio * len(self.y_train_parts[client]))
            X = np.zeros(shape=(num_rows, num_categorical_columns + num_numerical_fields))
            y = np.zeros(shape=num_rows)

            for row in X:
                # categorical左侧的界从0开始。
                categorical_left_bound = 0
                # 对于每一个分类categorical
                for num_this_categorical in self.num_each_categorical_field():
                    # 考虑这个categorical的哪一个enum值变为1
                    rand = np.random.randint(0, num_this_categorical)
                    row[rand + categorical_left_bound] = 1
                    # 维护categorical的左界
                    categorical_left_bound += num_this_categorical

                # 113个角色中选择5个己方角色，选择5个敌方角色。
                selected_champion_ids = np.random.choice([i for i in range(1, 114)], size=10, replace=False)
                selected_champions = [f"hero{i}" for i in selected_champion_ids]
                for i, selected_champion in enumerate(selected_champions):
                    if self.X_numerical_stds[selected_champion_ids[i] - 1] != 0:
                        if i < 5:
                            row[num_categorical_columns + selected_champion_ids[i] - 1] = (1 - self.X_numerical_means[selected_champion_ids[i] - 1]) / self.X_numerical_stds[selected_champion_ids[i] - 1]
                        else:
                            row[num_categorical_columns + selected_champion_ids[i] - 1] = (-1 - self.X_numerical_means[selected_champion_ids[i] - 1]) / self.X_numerical_stds[selected_champion_ids[i] - 1]
                    else:
                        # std == 0
                        if i < 5:
                            row[num_categorical_columns + selected_champion_ids[i] - 1] = 1
                        else:
                            row[num_categorical_columns + selected_champion_ids[i] - 1] = -1

            for i in range(num_rows):
                y[i] = np.random.randint(0, 2)

            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)

            self.X_train_parts[client] = torch.cat([self.X_train_parts[client], X])
            self.y_train_parts[client] = torch.cat([self.y_train_parts[client], y])

        return


class UrlReputation(StructuredClassificationManager):
    """
    We do not have the information about the attributes, thus we cannot simulate any adverse data on this dataset.
    That is, the dataset is not suitable for robustness testing!!!
    Not implemented!
    """
    pass


class CreditCard(StructuredClassificationManager):
    """
    We do not have the information about the attributes, thus we cannot simulate any adverse data on this dataset.
    That is, the dataset is not suitable for robustness testing!!!
    Not implemented!
    """
    def __init__(self):
        super().__init__(name="CreditCard")
        self.task = "Classification"
        return

    def read(self, test_ratio, shuffle_seed, cuda=False, nrows=None):
        self.data = pd.read_csv('./data/raw/creditcard_2023/creditcard_2023.csv',
                                nrows=nrows)
        self.data = sklearn.utils.shuffle(self.data, random_state=shuffle_seed)

        # 预处理，处理未知值，将上一步得到的含NaN的行都删除
        self.data.dropna(axis=0, how='any', inplace=True)

        X_df = self.data.drop(['id', 'Class'], axis=1)
        X = preprocessing.scale(X_df.to_numpy(float))

        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(self.data.Class)

        np.random.seed(666)
        shuffle_indices = np.arange(X.shape[0])
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        y = y[shuffle_indices]

        # 转变为tensor
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)

        self.X = X
        self.y = y

        self.train_test_split(test_ratio=test_ratio, random_state=shuffle_seed)

        if cuda:
            self._data_to_cuda()

        return self.X_train, self.y_train, self.X_test, self.y_test
