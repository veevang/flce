import torch
from sklearn.metrics import accuracy_score
from torch import nn
import time
from tqdm import tqdm
import numpy as np

import module.model
from module.data_manager import CreditCard, TicTacToe, Dota2, Adult, Bank


def eval_unit_model(temp_max_num_epoch, temp_lr, temp_hidden_layer_size, batch_size):
    t_origin = time.process_time()
    # v = np.zeros([repeat_time, temp_max_num_epoch])
    v = np.zeros([repeat_time, ])
    for seed in tqdm(range(6694, 6694 + repeat_time)):
        loader = loader_class()
        loader.read(test_ratio=0.1, shuffle_seed=seed)
        loader.X_train = loader.X_train[:round(ratio * len(loader.X_train))]
        loader.y_train = loader.y_train[:round(ratio * len(loader.y_train))]
        # loader.uniform_split(8)
        # loader.non_iid_split(8, [0.8]*8, seed)
        mdl = model_class(seed=seed, device=device, lr=temp_lr, num_epoch=temp_max_num_epoch,
                          hidden_layer_size=temp_hidden_layer_size, batch_size=batch_size)
    #     # print(accuracy_score(y_true=loader.y_test, y_pred=y_pred))
    #     for i in range(temp_max_num_epoch):
    #         mdl._fit(loader.X_train, loader.y_train,
    #                  num_epochs=1,
    #                  loss_fun=nn.BCELoss(),
    #                  lr=temp_lr,
    #                  incremental=True,
    #                  batch_size=batch_size)
    #         y_pred = mdl.predict(loader.X_test)
    #         v[seed - 6694][i] = (accuracy_score(y_true=loader.y_test, y_pred=y_pred))
    #     # print(f"num epoch:{temp_num_epoch}, lr: {temp_lr}, accu:{v}")
    # for i in range(temp_max_num_epoch):
    #     print(f"hidden size: {temp_hidden_layer_size}, batch_size: {batch_size}, lr: {temp_lr}, num_global_rounds: {temp_max_num_epoch}", np.average(v.T[i]) * 100)

        mdl.fed_train_and_score(X_train_parts=loader.X_train_parts[:4], y_train_parts=loader.y_train_parts[:4],
                            X_test=loader.X_test, y_test=loader.y_test, value_functions=["accuracy"])
        y_pred = mdl.predict(loader.X_test)
        v[seed - 6694] = (accuracy_score(y_true=loader.y_test, y_pred=y_pred))
    print(
        f"hidden size: {temp_hidden_layer_size}, batch_size: {batch_size}, lr: {temp_lr}, num_global_rounds: {temp_max_num_epoch}",
        np.average(v) * 100)

    t_end = time.process_time()
    print(f"batch size: {batch_size}, cpu time: {t_end - t_origin}")


repeat_time = 4
# 训练数据占总训练集的比例（用于测试移除5个参与方后的accuracy）
ratio = 1
# ratio = 0.02

device = torch.device("cuda:3")
# print(device)

# loader_class = TicTacToe
# model_class = module.model.net.TicTacToeMLP
# loader_class = CreditCard
# model_class = module.model.net.CreditCardMLP

# loader_class = Adult
# model_class = module.model.net.AdultMLP
# eval_unit_model(25, 0.001, 24, 64)

# loader_class = Bank
# model_class = module.model.net.BankMLP
# eval_unit_model(20, 0.001, 8, 64)

loader_class = Dota2
model_class = module.model.net.Dota2MLP
eval_unit_model(5, 0.001, 4, 128)

# eval_unit_model(60, 0.0005, 16, 16)
# eval_unit_model(70, 0.008, 16, 16)
# eval_unit_model(80, 0.001, 16, 8)
# eval_unit_model(60, 0.004, 16, 16)
# eval_unit_model(60, 0.005, 16, 16)
# eval_unit_model(5, 0.005, 4, 256)


# eval_unit_model(3, 0.01, 4, 256)

# Adult batch size 128


# eval_unit_model(40, 0.001, 16)
# processes = []
# for num_epoch in [20, 40, 60]:
#     processes.append(
#         Process(target=eval_unit_model, args=(num_epoch, 0.001), daemon=True))
#     processes[-1].start()
# for p in processes:
#     p.join()
