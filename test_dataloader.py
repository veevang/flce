from tqdm import tqdm

from datapre import *

# -6705

# original_seed = 6694
# num_try = 20
original_seed = 6694
num_try = 10
distributions_datasets_alphas = [
    # ("quantity skew", "adult", 0.365),
    # ("quantity skew", "bank", 0.35),
    # ("label skew", "adult", 0.8), ("label skew", "bank", 0.8),
    # ("quantity skew", "dota2", 0.4),
    ("quantity skew", "tictactoe", 0.65),
    ("label skew", "tictactoe", 0.8),
]

for seed in tqdm(range(original_seed, original_seed + num_try)):
    for distribution, dataset, alpha in distributions_datasets_alphas:
        loader = load_and_partition(seed, dataset, distribution, alpha)
        # print(loader.X_train.size(1))
        for y in loader.y_train_parts:
            if len(y) == 0:
                raise ValueError(f"{seed}, {distribution}, {dataset}, {alpha}, the dataset of a client is empty! ")
            else:
                # print("pass")
                continue
