from datapre import *
from model.model import *
from utils.logger import Logger
import datetime
import argparse
import config


# cache里面是list
def str_to_bool(string):
    if string.lower() == 'true':
        flag = True
    else:
        flag = False
    return flag


def log_contributions(seed, contribution_list, method, distribution, dataset, model, value_function,
                      attack_method, logging_path):
    if attack_method is None:
        l_con = Logger(logging_path, "contribution.csv")
    else:
        l_con = Logger(logging_path, "attacked_contribution.csv")
    d_con = dict()
    d_con["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    d_con["dataset"] = dataset
    d_con["alpha"] = alpha
    d_con["distribution"] = distribution
    d_con["seed"] = seed
    d_con["#parts"] = config.num_parts
    d_con["model"] = model
    d_con["lr"] = lr
    d_con["num_epoch"] = num_epoch
    d_con["hidden_layer_size"] = hidden_layer_size
    d_con["batch_size"] = batch_size
    d_con["value function"] = value_function
    d_con["method"] = method
    if attack_method is not None:
        d_con["attack method"] = attack_method
        d_con["attack arg"] = attack_arg
        d_con["num attack clients"] = num_attack_clients
    for number, contribution in enumerate(contribution_list):
        d_con["client number"] = number
        d_con["contribution"] = contribution
        l_con.log(d_con)


def log_process_time(seed, method, process_time, distribution, dataset, logging_path, model_name):
    l_t = Logger(logging_path, "process_time.csv")
    d_t = dict()
    d_t["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    d_t["dataset"] = dataset
    d_t["distribution"] = distribution
    d_t["seed"] = seed
    d_t["#parts"] = config.num_parts
    d_t["model"] = model_name
    d_t["lr"] = lr
    d_t["num_epoch"] = num_epoch
    d_t["hidden_layer_size"] = hidden_layer_size
    d_t["batch_size"] = batch_size
    d_t["value functions"] = value_functions
    d_t["method"] = method
    d_t["process time"] = process_time
    l_t.log(d_t)


def log_remove_client(seed, distribution, dataset, method_name, x, y_remove_best, y_remove_worst, alpha, logging_path,
                      value_function, model_name, num_removed_client_best, num_removed_client_worst, attack_method,
                      attack_arg):
    logger = Logger(logging_path, "remove_client_data.csv")
    dic = dict()
    dic["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dic["distribution"] = distribution
    dic["dataset"] = dataset
    dic["seed"] = seed
    dic["#parts"] = config.num_parts
    dic["model"] = model_name
    dic["lr"] = lr
    dic["num_epoch"] = num_epoch
    dic["hidden_layer_size"] = hidden_layer_size
    dic["batch_size"] = batch_size
    dic["value function"] = value_function
    dic["method"] = method_name
    dic["alpha"] = alpha
    if attack_method:
        dic["attack_method"] = attack_method
        dic["attack_arg"] = attack_arg
        dic["num_attack_clients"] = num_attack_clients
    # log每一个数据点
    for remove_topic, y_list, num_removed_client_list in zip(["best", "worst"], [y_remove_best, y_remove_worst],
                                                             [num_removed_client_best, num_removed_client_worst]):
        for j in range(len(x)):
            dic["x"] = x[j]
            dic["y"] = y_list[j]
            dic["remove topic"] = remove_topic
            dic["num_removed_client"] = num_removed_client_list[j]
            logger.log(dic)
    return


def cache_path(seed):
    folder = f"./data/utility_cache/--topic {args.topic} --dataset {dataset} --model {model_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if topic == "effective":
        return f"./data/utility_cache/--topic effective --dataset {dataset} --model {model_name}/--num_epoch {num_epoch} --lr {lr} --hidden_layer_size {hidden_layer_size} --batch_size {batch_size} --alpha {alpha} {distribution} --seed {seed} {value_functions}.pkl"
    elif topic == "robust" or topic == "effective_in_robust_setting":
        path = f"./data/utility_cache/--topic robust --dataset {dataset} --model {model_name}/--num_epoch {num_epoch} --lr {lr} --hidden_layer_size {hidden_layer_size} --batch_size {batch_size} --alpha {alpha} {distribution} --seed {seed} {attack} {num_attack_clients} --attack_arg {attack_arg} {value_functions}.pkl"
        return path


def score_clients(loader, model, seed, distribution, dataset, value_functions, cache, log_con,
                  log_time, attack_method, logging_path, log_remove, alpha, method, model_name):
    con = dict()
    tm = dict()
    instances = []
    method_instance = method(loader, model, cache, value_functions=value_functions)
    print(f"{method_instance.name} running...")
    instances.append(method_instance)
    con[method_instance.name], tm[method_instance.name] = method_instance.get_contributions(seed=seed,
                                                                                            num_samples=config.num_samples,
                                                                                            decfac=config.dec_fac,
                                                                                            num_local_epochs=config.num_local_epochs,
                                                                                            )
    # print(f"contributions{method_instance.contributions}")
    if log_con:
        for idx_val in range(len(value_functions)):
            log_contributions(seed, con[method_instance.name][idx_val], method_instance.name, distribution, dataset,
                              model_name, value_functions[idx_val], attack_method, logging_path)

    if log_remove:
        x, y_remove_best, y_remove_worst, num_removed_client_best, num_removed_client_worst = method_instance.get_remove_client_data()
        method_name = method_instance.name
        for idx_val in range(len(value_functions)):
            log_remove_client(seed, distribution, dataset, method_name, x, y_remove_best[idx_val],
                              y_remove_worst[idx_val], alpha, logging_path, value_functions[idx_val], model_name,
                              num_removed_client_best[idx_val], num_removed_client_worst[idx_val],
                              attack_method=attack_method, attack_arg=attack_arg)

    if log_time:
        log_process_time(seed, method_instance.name, tm[method_instance.name], distribution, dataset, logging_path,
                         model_name)

    return con, instances


def eval_remove_client(seed, distribution, dataset, logging_path, method, alpha, model_name):
    loader = get_data(seed=seed, dataset=dataset, distribution=distribution, alpha=alpha)

    # model
    model = return_model(model_name, seed=seed, num_epoch=num_epoch, lr=lr, device=device,
                         hidden_layer_size=hidden_layer_size, batch_size=batch_size)

    # ---------------------------------------
    # contribution evaluation and time

    if args.topic == "effective":
        common_cache_path = cache_path(seed)
        if os.path.exists(common_cache_path):
            with open(common_cache_path, 'rb') as file:
                common_cache = pickle.load(file)
        else:
            common_cache = dict()
        original_cache_size = len(common_cache)
        score_clients(loader, model, seed, distribution, dataset, value_functions, common_cache, True, False, None,
                      logging_path, True, alpha, method, model_name)

        if original_cache_size < len(common_cache):
            with open(common_cache_path, 'wb') as file:
                pickle.dump(common_cache, file)
    elif args.topic == "efficient":
        score_clients(loader, model, seed, distribution, dataset, value_functions, dict(), True, True, None,
                      logging_path, True, alpha, method, model_name)

    return


def eval_ratios(seed, distribution, dataset, attack_method, logging_path, method, alpha, attack_arg, model_name,
                num_attack_clients):
    loader = get_attack_data(seed=seed, dataset=dataset, distribution=distribution,
                             attack_method=attack_method, alpha=alpha, attack_arg=attack_arg,
                             num_attack_clients=num_attack_clients)
    model = return_model(model_name, seed=seed, num_epoch=num_epoch, lr=lr, device=device,
                         hidden_layer_size=hidden_layer_size, batch_size=batch_size)

    common_cache_path = cache_path(seed)
    if os.path.exists(common_cache_path):
        with open(common_cache_path, 'rb') as file:
            common_cache = pickle.load(file)
    else:
        common_cache = dict()
    original_cache_size = len(common_cache)
    score_clients(loader, model, seed, distribution, dataset, value_functions, common_cache, True, False, attack_method,
                  logging_path, False, alpha, method, model_name)

    if original_cache_size < len(common_cache):
        with open(common_cache_path, 'wb') as file:
            pickle.dump(common_cache, file)
    return


def eval_remove_client_robust_setting(seed, distribution, dataset, attack_method, logging_path, method, alpha,
                                      attack_arg, model_name, num_attack_clients):
    loader = get_attack_data(seed=seed, dataset=dataset, distribution=distribution,
                             attack_method=attack_method, alpha=alpha, attack_arg=attack_arg,
                             num_attack_clients=num_attack_clients)
    model = return_model(model_name, seed=seed, num_epoch=num_epoch, lr=lr, device=device,
                         hidden_layer_size=hidden_layer_size, batch_size=batch_size)

    common_cache_path = cache_path(seed)
    if os.path.exists(common_cache_path):
        with open(common_cache_path, 'rb') as file:
            common_cache = pickle.load(file)
    else:
        common_cache = dict()
    original_cache_size = len(common_cache)
    score_clients(loader, model, seed, distribution, dataset, value_functions, common_cache, True, False, attack_method,
                  logging_path, True, alpha, method, model_name)

    if original_cache_size < len(common_cache):
        with open(common_cache_path, 'wb') as file:
            pickle.dump(common_cache, file)
    return


def identify_method(method):
    # print(method)
    method_dict = {
        "RandomMethod": RandomMethod,
        "Individual": Individual,
        "LeaveOneOut": LeaveOneOut,
        "ShapleyValue": ShapleyValue,
        "LeastCore": LeastCore,
        "TMC_Shapley": TMC_ShapleyValue,
        "MC_LeastCore": MC_LeastCore,
        "TMC_GuidedSampling_Shapley": TMC_GuidedSampling_Shapley,
        "MC_StructuredSampling_Shapley": MC_StructuredSampling_Shapley,
        "Multi_Rounds": Multi_Rounds,
        "GTG_Shapley": GTG_Shapley,
    }
    return method_dict[method]


if __name__ == "__main__":
    # input args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--topic", help="topic", required=False)
    parser.add_argument("-m", "--method", type=identify_method, required=False)
    parser.add_argument("--dataset", help="dataset", required=False)
    parser.add_argument("--model", required=False)
    parser.add_argument("-a", "--alpha", type=float, required=False)
    parser.add_argument("--distribution", required=False)
    parser.add_argument("-s", "--seed", type=int, required=False)
    parser.add_argument("--num_repeat", type=int, required=False)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--num_try", type=str, required=True)
    parser.add_argument("--attack", required=False)
    parser.add_argument("--value_functions", nargs="+", type=str, help="a list of value functions")
    parser.add_argument("--attack_arg", type=float)
    parser.add_argument("--num_attack_clients", type=int, )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--hidden_layer_size", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch_size", type=int)

    # parser.add_argument("--run_robust", help="Whether to run robustness comparison", type=str_to_bool, required=False)
    # parser.add_argument("--rebuild", type=str_to_bool, required=False)
    # parser.add_argument("--is_local", type=str_to_bool)

    args = parser.parse_args()
    topic = args.topic
    method = args.method
    dataset = args.dataset
    alpha = args.alpha
    distribution = args.distribution
    original_seed = args.seed
    num_repeat = args.num_repeat
    start_date = args.start_date
    num_try = args.num_try
    attack = args.attack
    value_functions = args.value_functions
    attack_arg = args.attack_arg
    model_name = args.model
    lr = args.lr
    num_epoch = args.num_epoch
    hidden_layer_size = args.hidden_layer_size
    batch_size = args.batch_size
    num_attack_clients = args.num_attack_clients

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + args.device)
        torch.cuda.set_device(device)

    # 建立文件夹
    log_path = f"./logs/{start_date}/{num_try}"

    num_try = int(num_try)
    if topic == "effective" or topic == "efficient":
        for temp_seed in range(original_seed, original_seed + num_repeat):
            eval_remove_client(temp_seed, distribution, dataset, log_path, method, alpha, model_name)
    elif topic == "robust":
        for temp_seed in range(original_seed, original_seed + num_repeat):
            eval_ratios(temp_seed, distribution, dataset, attack, log_path, method, alpha, attack_arg, model_name,
                        num_attack_clients)
    elif topic == "effective_in_robust_setting":
        for temp_seed in range(original_seed, original_seed + num_repeat):
            eval_remove_client_robust_setting(temp_seed, distribution, dataset, attack, log_path, method, alpha,
                                              attack_arg, model_name, num_attack_clients)
    else:
        raise Exception

    print(f"Running finished at {datetime.datetime.now()}")
