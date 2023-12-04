import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from ours.components import BinarizeLayer, UnionLayer, LRLayer

TEST_CNT_MOD = 40


class MLLP(nn.Module):
    def __init__(self, dim_list, use_not=False, left=None, right=None, estimated_grad=False):
        super(MLLP, self).__init__()

        self.dim_list = dim_list
        self.use_not = use_not
        self.left = left
        self.right = right
        self.layer_list = nn.ModuleList([])

        prev_layer_dim = dim_list[0]
        for i in range(1, len(dim_list)):
            num = prev_layer_dim
            if i >= 4:
                num += self.layer_list[-2].output_dim

            if i == 1:
                layer = BinarizeLayer(dim_list[i], num, self.use_not, self.left, self.right)
                layer_name = 'binary{}'.format(i)
            elif i == len(dim_list) - 1:
                layer = LRLayer(dim_list[i], num)
                layer_name = 'lr{}'.format(i)
            else:
                layer = UnionLayer(dim_list[i], num, estimated_grad=estimated_grad)
                layer_name = 'union{}'.format(i)
            prev_layer_dim = layer.output_dim
            self.add_module(layer_name, layer)
            self.layer_list.append(layer)

    def forward(self, x):
        return self.continuous_forward(x), self.binarized_forward(x)

    def continuous_forward(self, x):
        x_res = None
        for i, layer in enumerate(self.layer_list):
            if i <= 1:
                x = layer(x)
            else:
                x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                x_res = x
                x = layer(x_cat)
        return x

    def binarized_forward(self, x):
        with torch.no_grad():
            x_res = None
            for i, layer in enumerate(self.layer_list):
                if i <= 1:
                    x = layer.binarized_forward(x)
                else:
                    x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                    x_res = x
                    x = layer.binarized_forward(x_cat)
            return x


class MyDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    @property
    def layer_list(self):
        return self.module.layer_list


class RuleModel:
    def __init__(self, dim_list, device_id, use_not=False, is_rank0=False, log_file=None, writer=None, left=None,
                 right=None, save_best=True, estimated_grad=False, save_path=None, distributed=False):
        super(RuleModel, self).__init__()
        self.dim_list = dim_list
        self.use_not = use_not
        self.best_acc = -1.

        self.device_id = device_id
        self.is_rank0 = is_rank0
        self.save_best = save_best
        self.estimated_grad = estimated_grad
        self.save_path = save_path
        if self.is_rank0:
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            log_format = '%(asctime)s - [%(levelname)s] - %(message)s'
            if log_file is None:
                logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format=log_format)
            else:
                logging.basicConfig(level=logging.DEBUG, filename=log_file, filemode='w', format=log_format)
        self.writer = writer

        self.net = MLLP(dim_list, use_not=use_not, left=left, right=right, estimated_grad=estimated_grad)
        self.net.to(device_id)

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for layer in self.net.layer_list[: -1]:
            layer.clip()

    def data_transform(self, X, y):
        X = X.astype(np.float)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float)
        return torch.tensor(X), torch.tensor(y)

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train_model(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, valid_loader=None,
                    epoch=50, lr=0.01, lr_decay_epoch=100, lr_decay_rate=0.75, batch_size=64, weight_decay=0.0,
                    log_iter=50):

        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        accuracy = []
        accuracy_b = []
        f1_score = []
        f1_score_b = []

        criterion = nn.CrossEntropyLoss().cuda(self.device_id)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        cnt = -1
        avg_batch_loss_cont = 0.0
        avg_batch_loss_bina = 0.0
        epoch_histc = defaultdict(list)
        self.save_model()
        TEST_CNT_MOD = len(data_loader) // 10 + 1
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            epoch_loss_cont = 0.0
            epoch_loss_bina = 0.0
            abs_gradient_max = 0.0
            abs_gradient_avg = 0.0

            ba_cnt = 1e-5
            for X, y in data_loader:
                ba_cnt += 1
                X = X.to(self.device_id)
                y = y.to(self.device_id)
                optimizer.zero_grad()  # Zero the gradient buffers.
                y_pred_cont, y_pred_bina = self.net.forward(X)
                with torch.no_grad():
                    y_prob = torch.softmax(y_pred_bina, dim=1)
                    y_arg = torch.argmax(y, dim=1)
                    loss_cont = criterion(y_pred_cont, y_arg)
                    loss_bina = criterion(y_pred_bina, y_arg)
                    ba_loss_cont = loss_cont.item()
                    ba_loss_bina = loss_bina.item()
                    epoch_loss_cont += ba_loss_cont
                    epoch_loss_bina += ba_loss_bina
                    avg_batch_loss_cont += ba_loss_cont
                    avg_batch_loss_bina += ba_loss_bina
                y_pred_cont.backward((y_prob - y) / y.shape[0])  # gradients for CrossEntropy Loss of loss_bina
                cnt += 1

                if self.is_rank0 and cnt % log_iter == 0 and cnt != 0 and self.writer is not None:
                    self.writer.add_scalar('Avg_Batch_Loss_cont', avg_batch_loss_cont / log_iter, cnt)
                    self.writer.add_scalar('Avg_Batch_Loss_GradGrafting', avg_batch_loss_bina / log_iter, cnt)
                    avg_batch_loss_cont = 0.0
                    avg_batch_loss_bina = 0.0
                optimizer.step()
                if self.is_rank0:
                    for i, param in enumerate(self.net.parameters()):
                        abs_gradient_max = max(abs_gradient_max, abs(torch.max(param.grad)))
                        abs_gradient_avg += torch.sum(torch.abs(param.grad)) / (param.grad.numel())
                self.clip()

                if self.is_rank0 and cnt % TEST_CNT_MOD == 0:
                    if X_validation is not None and y_validation is not None:
                        acc, acc_b, f1, f1_b = self.test(X_validation, y_validation, batch_size=batch_size,
                                                         need_transform=False, set_name='Validation')
                    elif valid_loader is not None:
                        acc, acc_b, f1, f1_b = self.test(test_loader=valid_loader, need_transform=False,
                                                         set_name='Validation')
                    elif data_loader is not None:
                        acc, acc_b, f1, f1_b = self.test(test_loader=data_loader, need_transform=False,
                                                         set_name='Training')
                    else:
                        acc, acc_b, f1, f1_b = self.test(X, y, batch_size=batch_size, need_transform=False,
                                                         set_name='Training')
                    if self.save_best and acc_b > self.best_acc:
                        self.best_acc = acc_b
                        self.save_model()
                    accuracy.append(acc)
                    accuracy_b.append(acc_b)
                    f1_score.append(f1)
                    f1_score_b.append(f1_b)
                    if self.writer is not None:
                        self.writer.add_scalar('Accuracy_cont', acc, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('Accuracy_bina', acc_b, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_cont', f1, cnt // TEST_CNT_MOD)
                        self.writer.add_scalar('F1_Score_bina', f1_b, cnt // TEST_CNT_MOD)
            if self.is_rank0:
                logging.info('epoch: {}, loss_cont: {}, loss_bina: {}'.format(epo, epoch_loss_cont, epoch_loss_bina))
                for name, param in self.net.named_parameters():
                    maxl = 1 if 'con_layer' in name or 'dis_layer' in name else 0
                    epoch_histc[name].append(torch.histc(param.data, bins=10, max=maxl).cpu().numpy())
                if self.writer is not None:
                    self.writer.add_scalar('Training_Loss_cont', epoch_loss_cont, epo)
                    self.writer.add_scalar('Training_Loss_bina', epoch_loss_bina, epo)
                    self.writer.add_scalar('Abs_Gradient_Max', abs_gradient_max, epo)
                    self.writer.add_scalar('Abs_Gradient_Avg', abs_gradient_avg / ba_cnt, epo)
                loss_log.append(epoch_loss_bina)
        if self.is_rank0 and not self.save_best:
            self.save_model()
        return epoch_histc

    def test(self, X=None, y=None, test_loader=None, batch_size=32, need_transform=True, set_name='Validation'):
        if X is not None and y is not None and need_transform:
            X, y = self.data_transform(X, y)
        with torch.no_grad():
            if X is not None and y is not None:
                test_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)

            y_list = []
            for X, y in test_loader:
                y_list.append(y)
            y_true = torch.cat(y_list, dim=0)
            y_true = y_true.cpu().numpy().astype(np.int64)
            y_true = np.argmax(y_true, axis=1)
            data_num = y_true.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('y_true: {} {}'.format(y_true.shape, y_true[:: slice_step]))

            y_pred_list = []
            y_pred_b_list = []
            for X, y in test_loader:
                X = X.to(self.device_id)  # , non_blocking=True)
                output = self.net.forward(X)
                y_pred_list.append(output[0])
                y_pred_b_list.append(output[1])

            y_pred = torch.cat(y_pred_list).cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('y_cont: {} {}'.format(y_pred.shape, y_pred[:: slice_step]))

            y_pred_b = torch.cat(y_pred_b_list).cpu().numpy()
            y_pred_b_arg = np.argmax(y_pred_b, axis=1)
            logging.debug('y_bina_: {} {}'.format(y_pred_b_arg.shape, y_pred_b_arg[:: slice_step]))
            logging.debug('y_bina: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step)]))

            accuracy = metrics.accuracy_score(y_true, y_pred)
            accuracy_b = metrics.accuracy_score(y_true, y_pred_b_arg)

            f1_score = metrics.f1_score(y_true, y_pred, average='macro', zero_division=1)
            f1_score_b = metrics.f1_score(y_true, y_pred_b_arg, average='macro', zero_division=1)

            logging.info('-' * 60)
            logging.info('On {} Set:\n\tAccuracy of Rule-based Model: {}'
                         '\n\tF1 Score of Rule-based Model: {}'.format(set_name, accuracy_b, f1_score_b))
            logging.info('On {} Set:\nPerformance of Rule-based Model: \n{}\n{}'.format(
                set_name, metrics.confusion_matrix(y_true, y_pred_b_arg),
                metrics.classification_report(y_true, y_pred_b_arg, zero_division=1)))
            logging.info('-' * 60)
        return accuracy, accuracy_b, f1_score, f1_score_b

    def save_model(self):
        model_args = {'dim_list': self.dim_list, 'use_not': self.use_not, 'estimated_grad': self.estimated_grad}
        torch.save({'model_state_dict': self.net.state_dict(), 'model_args': model_args}, self.save_path)

    def detect_dead_node(self, data_loader=None):
        acts = []
        with torch.no_grad():
            for layer in self.net.layer_list[:-1]:
                layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double, device=self.device_id)
                layer.forward_tot = 0
            for x, y in data_loader:
                x = x.to(self.device_id)
                x_res = None
                for i, layer in enumerate(self.net.layer_list[:-1]):
                    if i <= 1:
                        x = layer.binarized_forward(x)
                    else:
                        x_cat = torch.cat([x, x_res], dim=1) if x_res is not None else x
                        x_res = x
                        x = layer.binarized_forward(x_cat)
                    layer.node_activation_cnt += torch.sum(x, dim=0)
                    layer.forward_tot += x.shape[0]
                acts.append(x.cpu().detach().numpy())
            return np.concatenate(acts)

    def preds(self, loader):
        preds = []
        with torch.no_grad():
            for X, y in loader:
                pred = self.net.forward(X.to(self.device_id))
                preds.append(pred[1])
        return torch.cat(preds).argmax(1).tolist()  # .cpu().numpy().argmax(1)

    def rule_print(self, feature_name, label_name, train_loader, file=sys.stdout, mean=None, std=None):
        if self.net.layer_list[1] is None and train_loader is None:
            raise Exception("Need train_loader for the dead nodes detection.")
        if self.net.layer_list[1].node_activation_cnt is None:
            self.detect_dead_node(train_loader)

        bound_name = self.net.layer_list[0].get_bound_name(feature_name, mean, std)
        self.net.layer_list[1].get_rules(self.net.layer_list[0], None)
        self.net.layer_list[1].get_rule_description((None, bound_name))

        if len(self.net.layer_list) >= 4:
            self.net.layer_list[2].get_rules(self.net.layer_list[1], None)
            self.net.layer_list[2].get_rule_description((None, self.net.layer_list[1].rule_name), wrap=True)

        if len(self.net.layer_list) >= 5:
            for i in range(3, len(self.net.layer_list) - 1):
                self.net.layer_list[i].get_rules(self.net.layer_list[i - 1], self.net.layer_list[i - 2])
                self.net.layer_list[i].get_rule_description(
                    (self.net.layer_list[i - 2].rule_name, self.net.layer_list[i - 1].rule_name), wrap=True)

        prev_layer = self.net.layer_list[-2]
        skip_connect_layer = self.net.layer_list[-3]
        always_act_pos = (prev_layer.node_activation_cnt == prev_layer.forward_tot)
        if skip_connect_layer.layer_type == 'union':
            shifted_dim2id = {(k + prev_layer.output_dim): (-2, v) for k, v in skip_connect_layer.dim2id.items()}
            prev_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}
            merged_dim2id = defaultdict(lambda: -1, {**shifted_dim2id, **prev_dim2id})
            always_act_pos = torch.cat(
                [always_act_pos, (skip_connect_layer.node_activation_cnt == skip_connect_layer.forward_tot)])
        else:  # single union layer condition
            merged_dim2id = {k: (-1, v) for k, v in prev_layer.dim2id.items()}

        Wl, bl = list(self.net.layer_list[-1].parameters())
        bl = torch.sum(Wl.T[always_act_pos], dim=0) + bl
        Wl = Wl.cpu().detach().numpy()  # do*di
        bl = bl.cpu().detach().numpy()

        marked = defaultdict(lambda: defaultdict(float))
        rid2dim = {}
        for label_id, wl in enumerate(Wl):
            for i, w in enumerate(wl):
                rid = merged_dim2id[i]
                if rid == -1 or rid[1] == -1: continue  # no/all action
                marked[rid][label_id] += w
                rid2dim[rid] = i % prev_layer.output_dim

        kv_list = sorted(marked.items(), key=lambda x: max(map(abs, x[1].values())), reverse=True)
        print('Frequent rules learned from training data are as follows:')
        print('RID', end='\t', file=file)
        print('Negative(-)/Postive(+)', end='\t', file=file)
        # for i, ln in enumerate(label_name):
        #     print('{}'.format(ln, bl[i]), end='\t', file=file)
        print('Support\tRule', file=file)
        for k, v in kv_list[:10]:
            rid = k
            print(rid[1], end='\t', file=file)
            print('-', end='\t', file=file) if v[0] > v[1] else print('+', end='\t', file=file)
            # for li in range(len(label_name)):
            #     print('{:.4f}'.format(v[li]), end='\t', file=file)
            now_layer = self.net.layer_list[-1 + rid[0]]
            print(now_layer.rule_name[rid[1]], end='\n', file=file)
        # print('#' * 30, file=file)
        print('\n\n')
        return kv_list, rid2dim, merged_dim2id
