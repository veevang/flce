import math

from measure import *

# original_seed = 6696
num_tries_every_day = 100
# attack_distribution = "label skew"

num_local_epochs = 1
dec_fac = 0.95

# 8参与方，重复10次
# num_parts = 8
# num_samples = round(math.log(num_parts)) * (num_parts ** 2)


# num_repeat = 10
# num_repeat = 1

# method_list = [RandomMethod, Individual, LeaveOneOut, ShapleyValue, LeastCore, TMC_ShapleyValue, MC_LeastCore, RV_ShapleyValue, RV_LeastCore]
# method_list = [RV_Individual, RV_LeaveOneOut, RV_ShapleyValue, RV_LeastCore]
# method_list = [Individual]
# method_list = [LeastCore, ShapleyValue]
# method_list = [RandomMethod, Individual, LeaveOneOut, ShapleyValue, LeastCore, TMC_ShapleyValue, MC_LeastCore, ]
# method_list = [LeastCore, MC_LeastCore]

# distribution_list = ["quantity skew", "label skew"]
# distribution_list = ["quantity skew", ]

# dataset_list = ["adult", "bank"]
# dataset_list = ["adult", ]
# dataset_list = ["MNIST"]

method_abbr = {"MC-StructuredSampling-Shapley": "StructuredMC-Shapley",
               "TMC-GuidedSampling-Shapley": "GuidedTMC-Shapley",
               "robust volume individual": "RVINDV",
               "robust volume leave one out": "RVLOO",
               "robust volume shapley value": "RVSV",
               "robust volume least core": "RVLC"}

# model_of_dataset = {"adult": "AdultMLP", "bank": "BankMLP", "MNIST": "CNNMNIST"}
# model_of_dataset = {"adult": "logistic regression", "bank": "BankMLP", "MNIST": "CNNMNIST"}

# model_of_dataset = {"adult": "logistic regression", "bank": "logistic regression", "diabetes": "linear regression",
#                     "california housing": "linear regression"}
# model_of_dataset = {"adult":"sgd cls log", "bank":"sgd cls log", "diabetes":"sgd reg sqr l2", "california housing":"sgd reg sqr l2"}

# value_function_of_dataset = {"adult": "f1_score", "bank": "f1_score", "diabetes": "r2_score",
                             # "california housing": "r2_score", "MNIST": ["f1", "f1_macro", "f1_micro", "accuracy"]}

# value_functions = ["f1", "f1_macro", "f1_micro", "accuracy"]

# alpha_of_dataset = {"adult": 1, "bank": 1, "diabetes": 2, "california housing": 1, "MNIST": 1}
# alpha_of_dataset_distribution = {"adult quantity skew": 0.6, "adult label skew": 1,
#                                  "bank quantity skew": 0.5, "bank label skew": 1,
#                                  # "diabetes": 2, "california housing": 1,
#                                  "MNIST quantity skew": 1, "MNIST label skew": 1}

num_rows_of_dataset = {"adult": None, "bank": None, "diabetes": None, "california housing": None, "MNIST": None, "dota2": None, "tictactoe": None, "creditcard":None}
# num_rows_of_dataset = {"adult": 1000, "bank": 10000, "diabetes": None, "california housing": 10000}
# attack_method_list = ["data replication", "random data generation", "low quality data", "label flip"]

# train test split
test_ratio = 0.1
# test_ratio = 0.25

# noise_client_ratio = 0.4
# noise_ratio_each_client = 0.2

# # 这些参数可以调整！
# # random generation
# randomly_generate_ratio = 0.3
#
# # low quality
# low_quality_shuffle_ratio = 0.3
#
# # replication
# multiplier_replication = 0.3
#
# # flip
# flip_ratio = 0.3
