import torch
from collections import defaultdict

from sklearn.metrics import accuracy_score

from module.method.measure import Measure
import copy
import torch.nn.functional as F


# https://github.com/AI-secure/Shapley-Study/blob/master/shapley/measures/LOO.py

class CGSV(Measure):
    name = 'CGSV'

    def __init__(self, loader, model, cache, value_functions):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        return

    # 没有补充q相关的参与方参数
    def get_contributions(self, **kwargs):
        gamma_scheduler = kwargs.get("gamma_scheduler")
        num_local_epochs = kwargs.get("num_local_epochs")
        gamma_gs = kwargs.get("gamma_gs")
        gs_alpha = kwargs.get("gs_alpha")

        shard_sizes = [len(self.X_train_parts[i]) for i in range(self.num_parts)]
        shard_sizes = torch.tensor(shard_sizes).float()

        model_architecture = self.model.__class__

        seed = self.model.seed
        lr = self.model.lr
        num_epoch = self.model.num_epoch
        hidden_layer_size = self.model.hidden_layer_size
        device = self.model.device
        batch_size = self.model.batch_size

        # ---- init the clients ----
        client_models, client_optimizers, client_schedulers = [], [], []
        # server_model 定为 self.model
        self.model = self.model.load_state_dict(self.model.initial_state_dict)

        for i in range(self.num_parts):
            model = copy.deepcopy(self.model)
            # try:
            # optimizer = optimizer_fn(model.parameters(), lr=args['lr'], momentum=args['momentum'])
            # except:

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300], gamma=0.1)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma_scheduler)

            client_models.append(model)
            client_optimizers.append(optimizer)
            client_schedulers.append(scheduler)

        # ---- book-keeping variables ----------

        rs_dict = []
        # qs_dict = []
        rs = torch.zeros(self.num_parts, device=device)
        past_phis = []

        # for performance analysis
        valid_perfs, local_perfs, fed_perfs = defaultdict(list), defaultdict(list), defaultdict(list)

        # for gradient/model parameter analysis
        dist_all_layer, dist_last_layer = defaultdict(list), defaultdict(list)
        reward_all_layer, reward_last_layer = defaultdict(list), defaultdict(list)

        # ---- CML/FL begins ----
        for iteration in range(num_epoch):

            gradients = []
            for i in range(self.num_parts):
                X_i, y_i = self.X_train_parts[i], self.y_train_parts[i]
                model = client_models[i]
                optimizer = client_optimizers[i]
                scheduler = client_schedulers[i]

                model.train()
                model = model.to(device)

                backup = copy.deepcopy(model)

                model.fit(X_i, y_i, incremental=True, num_epochs=num_local_epochs)
                client_optimizers[i].step()

                gradient = self.compute_grad_update(old_model=backup, new_model=model, device=device)

                # SUPPOSE DO NOT TOP UP WITH OWN GRADIENTS
                model.load_state_dict(backup.state_dict())
                # add_update_to_model(model, gradient, device=device)

                # append the normalzied gradient
                flattened = self.flatten(gradient)
                norm_value = torch.linalg.norm(flattened) + 1e-7  # to prevent division by zero

                gradient = self.unflatten(torch.multiply(torch.tensor(gamma_gs), torch.div(flattened, norm_value)),
                                          gradient)

                gradients.append(gradient)

            # until now, all the normalized (based on gamma_gs) gradients of clients have been generated and
            # stored in gradients

            # ---- Server Aggregate ----

            aggregated_gradient = [torch.zeros(param.shape).to(device) for param in self.model.parameters()]

            # aggregate and update server model

            if iteration == 0:
                # first iteration use FedAvg
                weights = torch.div(shard_sizes, torch.sum(shard_sizes))
            else:
                weights = rs

            for gradient, weight in zip(gradients, weights):
                self.add_gradient_updates(aggregated_gradient, gradient, weight=weight)

            self.add_update_to_model(self.model, aggregated_gradient)

            # update reputation and calculate reward gradients
            flat_aggre_grad = self.flatten(aggregated_gradient)

            # phis = torch.zeros(N, device=device)
            phis = torch.tensor(
                [F.cosine_similarity(self.flatten(gradient), flat_aggre_grad, 0, 1e-10) for gradient in gradients],
                device=device)
            past_phis.append(phis)

            rs = gs_alpha * rs + (1 - gs_alpha) * phis

            rs = torch.clamp(rs, min=1e-3)  # make sure the rs do not go negative
            rs = torch.div(rs, rs.sum())  # normalize the weights to 1

            # # --- altruistic degree function
            # q_ratios = torch.tanh(args['beta'] * rs)
            # q_ratios = torch.div(q_ratios, torch.max(q_ratios))
            #
            # qs_dict.append(q_ratios)

            rs_dict.append(rs)

            # clients download the gradient of the grand coalition
            for i in range(self.num_parts):
                # reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratios[i],
                # mode='layer')

                reward_gradient = aggregated_gradient
                self.add_update_to_model(client_models[i], reward_gradient)

                ''' Analysis of rewarded gradients in terms cosine to the aggregated gradient '''
                # reward_all_layer[str(i) + 'cos'].append(
                #     F.cosine_similarity(flatten(reward_gradient), flat_aggre_grad, 0, 1e-10).item())
                # reward_all_layer[str(i) + 'l2'].append(norm(flatten(reward_gradient) - flat_aggre_grad).item())
                #
                # reward_last_layer[str(i) + 'cos'].append(
                #     F.cosine_similarity(flatten(reward_gradient[-2]), flatten(aggregated_gradient[-2]), 0,
                #                         1e-10).item())
                # reward_last_layer[str(i) + 'l2'].append(
                #     norm(flatten(reward_gradient[-2]) - flatten(aggregated_gradient[-2])).item())

            # weights = torch.div(shard_sizes, torch.sum(shard_sizes)) if iteration == 0 else rs

            # 对于所有参与方，然后对于全局模型，计算所有的

            # for i, model in enumerate(agent_models + [server_model]):
            #     valid_perfs[str(i) + '_loss'].append(loss.item())
            #     valid_perfs[str(i) + '_accu'].append(accuracy.item())
            #
            #     fed_loss, fed_accu = 0, 0
            #     for j, train_loader in enumerate(train_loaders):
            #         loss, accuracy = evaluate(model, train_loader, loss_fn=loss_fn, device=device)
            #
            #         fed_loss += weights[j] * loss.item()
            #         fed_accu += weights[j] * accuracy.item()
            #         if j == i:
            #             local_perfs[str(i) + '_loss'].append(loss.item())
            #             local_perfs[str(i) + '_accu'].append(accuracy.item())
            #
            #     fed_perfs[str(i) + '_loss'].append(fed_loss.item())
            #     fed_perfs[str(i) + '_accu'].append(fed_accu.item())

            # # ---- Record model distance to the server model ----
            # for i, model in enumerate(client_models + [init_backup]):
            #     percents, dists = compute_distance_percentage(model, server_model)
            #
            #     dist_all_layer[str(i) + 'dist'].append(np.mean(dists))
            #     dist_last_layer[str(i) + 'dist'].append(dists[-1])
            #
            #     dist_all_layer[str(i) + 'perc'].append(np.mean(percents))
            #     dist_last_layer[str(i) + 'perc'].append(percents[-1])

        # training finished!
        # report 一下最后的模型的 accuracy，看看是不是接近正常optimal的accuracy.
        y_pred = self.model.predict(self.X_test)
        print(f"accuracy of CG global model: {accuracy_score(self.y_test, y_pred)}")

        # rs 其实就是最后的归一化后的（importance）贡献值了，所以他其实更像是一个contribution estimation scheme.
        return rs

    # --------------gradient similarity begins----------------------
    @staticmethod
    def compute_grad_update(old_model, new_model, device=None):
        # maybe later to implement on selected layers/parameters
        if device:
            old_model, new_model = old_model.to(device), new_model.to(device)
        return [(new_param.data - old_param.data) for old_param, new_param in
                zip(old_model.parameters(), new_model.parameters())]

    @staticmethod
    def flatten(grad_update):
        return torch.cat([update.data.view(-1) for update in grad_update])

    @staticmethod
    def unflatten(flattened, normal_shape):
        grad_update = []
        for param in normal_shape:
            n_params = len(param.view(-1))
            grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
            flattened = flattened[n_params:]
        return grad_update

    def cosine_similarity(self, grad1, grad2, normalized=False):
        """
        Input: two sets of gradients of the same shape
        Output range: [-1, 1]
        """
        cos_sim = F.cosine_similarity(self.flatten(grad1), self.flatten(grad2), 0, 1e-10)
        if normalized:
            return (cos_sim + 1) / 2.0
        else:
            return cos_sim

    @staticmethod
    def add_gradient_updates(grad_update_1, grad_update_2, weight=1.0):
        assert len(grad_update_1) == len(grad_update_2), "Lengths of the two grad_updates not equal"

        for param_1, param_2 in zip(grad_update_1, grad_update_2):
            param_1.data += param_2.data * weight

    @staticmethod
    def add_update_to_model(model, update, weight=1.0, device=None):
        if not update:
            return model
        if device:
            model = model.to(device)
            update = [param.to(device) for param in update]

        for param_model, param_update in zip(model.parameters(), update):
            param_model.data += weight * param_update.data
        return model
