import datetime

from utils.plotter import Plotter
import argparse
from config import *

'''
This file can plot 5 different plots in the text of the article. 
i.e., the 3 remove client plots (6-schemes, sampling-and-truncation, gradient-reuse), robustness plot and time plot. 
This file only contains the configs, and does not include the implementation. 
For the implementation of plotters, please refer to the "./utils/plotter.py" file. 

For plotting, please first comment out all the parts that you do not need. 
After that, please type "python plotres --metric f1_macro accuracy f1_micro --is_final true" in the command line to\
plot the figures based on the final logs. 
'''

parser = argparse.ArgumentParser()
# parser.add_argument('arg_name', help='Help message for arg_name')
parser.add_argument('--start_date', help='the date of logs')
parser.add_argument('--num_try', help='the number of the running try of the day')
parser.add_argument('--metrics', nargs="+", help='utility metric')
parser.add_argument('--is_final', help="whether to plot final version")
args = parser.parse_args()
start_date = args.start_date
num_try = args.num_try
metrics = args.metrics
is_final = args.is_final
if is_final == "true":
    is_final = True
else:
    is_final = False

if not is_final:
    if start_date is None:
        start_date = datetime.datetime.now().strftime("%Y_%m_%d")
    if num_try is None:
        for i in range(config.num_tries_every_day, -1, -1):
            if os.path.exists(f"./logs/{start_date}/{str(i).rjust(2, '0')}"):
                num_try = i
                num_try = str(num_try).rjust(2, '0')
                break
            else:
                continue
        else:
            raise Exception("log file doesn't exist!")

datasets = ["tictactoe", "adult", "bank", "dota2"]
model_of_dataset = {"adult": "AdultMLP", "bank": "BankMLP", "dota2": "Dota2MLP", "tictactoe": "TicTacToeMLP"}
models = []
for ds in datasets:
    if "MLP" in model_of_dataset[ds]:
        model = "MLP"
    else:
        model = model_of_dataset[ds]
    models.append(model)

models = list(set(models))

attack_distribution = "label skew"
attack_client = 0
attack_arg = 0.3

# the following are configurations for robust and time plot.
attack_hue_order = ["Individual",
                    "LeaveOneOut",
                    "ShapleyValue",
                    "MC-StructuredSampling-Shapley",
                    "LeastCore",
                    "MC-LeastCore",
                    # "Random",
                    ]

time_hue_order = [
    # "MultiRounds",
    "Individual",
    "LeaveOneOut",
    "GTG-Shapley",
    "MC-StructuredSampling-Shapley",
    "ShapleyValue",
    "MC-LeastCore",
    "LeastCore",
]

"""
The following are configurations for 3 effectiveness line plots, 
which covers the 6 basic schemes, sampling-and-truncation, gradient reuse.
Only one can be selected each time, and the others should be commented out. 
"""

# 6 basic schemes.
# effective_title = "6-basic"
# effective_hue_order = ["Individual",
#                        "LeaveOneOut",
#                        "MC-StructuredSampling-Shapley",
#                        "MC-LeastCore",
#                        "ShapleyValue",
#                        "LeastCore",
#                        "Random",
#                        ]

# shapley sampling-and-truncation optimization techniques.
# effective_title = "sampling-and-truncation"
# effective_hue_order = [
#     "TMC-Shapley",
#     "MC-StructuredSampling-Shapley",
#     "TMC-GuidedSampling-Shapley",
#     "ShapleyValue",
#     "Random", ]

# gradient reuse optimization techniques.
# gtg mr structured shapley random
# effective_title = "gradient-reuse"
# effective_hue_order = [
#     "GTG-Shapley",
#     "MultiRounds",
#     "MC-StructuredSampling-Shapley",
#     "ShapleyValue",
#     "Random",
# ]

# init plotter.
p = Plotter(start_date=start_date, num_try=num_try, is_final=is_final)

# plot time.
# p.plot_time(time_hue_order)

# plot metric time.
# p.plot_metric_time()

# plot remove client best.
# for model in models:
#     # for vf in config.value_functions:
#     for metric in metrics:
#         p.plot_effective(num_parts=config.num_parts, model=model, value_function=metric, hue_order=effective_hue_order,datasets=datasets, title=effective_title)

# plot robustness.
# for model in models:
#     for metric in metrics:
#         p.plot_robust(
#             distribution=attack_distribution,
#             num_parts=config.num_parts,
#             model=model,
#             value_function=metric,
#             hue_order=attack_hue_order,
#             attack_client=attack_client,
#             attack_arg=attack_arg,
#         )

# plot metric robustness plot.
p.plot_metric_robust(
    distribution=attack_distribution,
    num_parts=config.num_parts,
    model=model,
    attack_client=attack_client,
    attack_arg=attack_arg,
)

