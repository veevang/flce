from loader.regression_data_loader import Diabetes, CaliforniaHousing
import scipy.stats
from loader.classification_data_loader import *
from loader.image_data_loader import *
from config import *
import pickle


def str_to_dataset(dataset):
    if dataset == "adult":
        loader = Adult()
    elif dataset == "bank":
        loader = Bank()
    elif dataset == "diabetes":
        loader = Diabetes()
    elif dataset == "california housing":
        loader = CaliforniaHousing()
    elif dataset == "MNIST":
        loader = MNIST()
    elif dataset == "dota2":
        loader = Dota2()
    elif dataset == "tictactoe":
        loader = TicTacToe()
    else:
        raise ValueError("dataset not exist!")
    return loader


def get_data(seed, dataset, distribution, alpha):
    loader = load_and_partition(seed, dataset, distribution, alpha)
    # with open(file_path, 'wb') as file:
    #     pickle.dump(loader, file)
    for y in loader.y_train_parts:
        if len(y) == 0:
            raise ValueError("the dataset of a client is empty! ")
    return loader


def load_and_partition(seed, dataset, distribution, alpha):
    # first load data
    loader = str_to_dataset(dataset)
    alpha_list = np.array([alpha for _ in range(config.num_parts)])
    loader.read(test_ratio=config.test_ratio, shuffle_seed=seed, nrows=config.num_rows_of_dataset[dataset])

    if distribution == "uniform":
        loader.uniform_split(config.num_parts)
    elif distribution == "quantity skew":
        ratios = scipy.stats.dirichlet.rvs(alpha_list, random_state=seed)
        ratios = np.concatenate(ratios)
        # print(ratios)
        loader.ratio_split(ratios)
    elif distribution == "label skew":
        loader.non_iid_split(config.num_parts, alpha_list, random_state=seed)
    # elif args.topic == "QualitySkew":
    #     loader.uniform_split(config.num_parts)
    #     flip_client = np.zeros(config.num_parts)
    #     flip_client[:round(config.num_parts * config.noise_client_ratio)] = 1
    #     np.random.seed(seed)
    #     np.random.shuffle(flip_client)
    #     flip_set = set()
    #     for i in range(config.num_parts):
    #         if flip_client[i] == 1:
    #             flip_set.add(i)
    #     # print(flip_set)
    #     loader.flip_y_train(flip_set, ratio=config.noise_ratio_each_client, random_seed=seed)
    #     dic["NoiseClientRatio"] = config.noise_client_ratio
    #     dic["NoiseRatioEachClient"] = config.noise_ratio_each_client

    else:
        raise ValueError("separate method does not exist!")
    return loader


def get_attack_data(seed, dataset, distribution, attack_method, alpha, attack_arg, num_attack_clients):
    loader = get_data(seed, dataset, distribution, alpha=alpha)
    attack_clients = set(range(num_attack_clients))
    if attack_method == "data replication":
        loader.data_copy(attack_clients, seed, attack_arg)
    elif attack_method == "random data generation":
        loader.randomly_generate_data(attack_clients, round(attack_arg * len(loader.X_train_parts[0])), seed)
    elif attack_method == "low quality data":
        loader.low_quality_data(attack_clients, random_seed=seed, ratio=attack_arg)
    elif attack_method == "label flip":
        loader.flip_y_train(attack_clients, seed, ratio=attack_arg)
    else:
        raise Exception(f"there is no attack method called {attack_method}!")
    # with open(file_path, 'wb') as file:
    #     pickle.dump(loader, file)

    return loader
