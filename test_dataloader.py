from tqdm import tqdm

from datapre import *

# -6705

# original_seed = 6694
# num_try = 20
num_parts = 8
original_seed = 6694
num_try = 10
distributions_datasets_alphas = [
    # for 8 clients
    # ("quantity skew", "adult", 0.365),
    # ("quantity skew", "bank", 0.35),
    # ("label skew", "adult", 0.8),
    # ("label skew", "bank", 0.8),
    # ("quantity skew", "dota2", 0.4),
    # ("quantity skew", "tictactoe", 0.65),
    # ("label skew", "tictactoe", 0.8),
    ("quantity skew", "creditcard", 0.3),
    ("label skew", "creditcard", 0.6),

    # for 14 clients
    # ("quantity skew", "adult", 0.625),
    # ("label skew", "adult", 0.8),
    # ("quantity skew", "dota2", 0.625),
    # ("label skew", "dota2", 0.8),
]

for seed in tqdm(range(original_seed, original_seed + num_try)):
    for distribution, dataset, alpha in distributions_datasets_alphas:
        loader = load_and_partition(seed, dataset, distribution, alpha, num_parts=num_parts)
        # print(f"{len(loader.y_train_parts)}")
        for y in loader.y_train_parts:
            if len(y) == 0:
                raise ValueError(f"{seed}, {distribution}, {dataset}, {alpha}, the dataset of a client is empty! ")
            else:
                continue
print("pass!")

        # for y in loader.y_train_parts:
        #     print(len(y))
        # loader.randomly_generate_data({1,2,3},1.0,1)
        # for y in loader.y_train_parts:
        #     print(len(y))
