import measure
from config import *
from model.model import return_model


def cal_cache():
    loader = generate_data()
    method_instance = measure.Individual()
    model = return_model(config.model_of_dataset[dataset], seed=seed)

def generate_data():
    alpha = config.alpha_of_dataset_distribution[f"{dataset} {distribution}"]
    loader = get_attack_data(seed=seed, dataset=dataset, distribution=distribution,
                             attack_method=attack_method, rebuild=rebuild, alpha=alpha)

def log_cache(cache):
    data_path = f"./data/utility_cache/"

