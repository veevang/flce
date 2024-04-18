num_tries_every_day = 100

num_local_epochs = 1
dec_fac = 0.95

method_abbr = {"MC-StructuredSampling-Shapley": "StructuredMC-Shapley",
               "TMC-GuidedSampling-Shapley": "GuidedTMC-Shapley",
               "robust volume individual": "RVINDV",
               "robust volume leave one out": "RVLOO",
               "robust volume shapley value": "RVSV",
               "robust volume least core": "RVLC"}


num_rows_of_dataset = {"adult": None, "bank": None, "diabetes": None, "california housing": None, "MNIST": None, "dota2": None, "tictactoe": None, "creditcard":None}

# train test split
test_ratio = 0.1
