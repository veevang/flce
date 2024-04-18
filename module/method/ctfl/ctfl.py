from interface.base import Base
from interface.valuation import Valuation
from utils.args import args
from train import train_model
import numpy as np
from dataprep.dataprep import get_data_loader
from utils import config
from collections import defaultdict


class CTFL(Base):
    def __init__(self, v: Valuation):
        super(CTFL, self).__init__('CTFL', v)
        self.db_enc, self.dfed = v.db_enc, v.dfed
        train_loader, valid_loader, test_loader = get_data_loader(v.dfed.tr, v.dfed.te, args.batch_size,
                                                                  pin_memory=True, save_best=args.save_best)
        self.model = train_model(v.db_enc, train_loader, valid_loader)
        self.scores = None

    def phi(self):
        if not self.scores: self.scores = self.compute_scores()
        return self.scores

    def compute_scores(self):
        tr_acts, te_acts = self.model.detect_dead_node(self.dfed.tr_loader), self.model.detect_dead_node(
            self.dfed.te_loader)
        kv_list, rid2dim, dim2rid = self.model.rule_print(self.db_enc.X_fname, self.db_enc.y_fname, self.dfed.tr_loader,
                                                          mean=self.db_enc.mean, std=self.db_enc.std)
        te_preds = self.model.preds(self.dfed.te_loader)
        r2w = {k: v for k, v in kv_list}
        te_rules = [set([dim2rid[i] for i, stat in enumerate(act) if stat and dim2rid[i][1] != -1]) for act in te_acts]
        tr_rules = [set([dim2rid[i] for i, stat in enumerate(act) if stat and dim2rid[i][1] != -1]) for act in tr_acts]
        micro_scores = np.zeros(len(tr_rules))
        ruleW4clients = [defaultdict(lambda: 0) for _ in range(self.dfed.nparts)]
        for teidx in range(len(te_preds)):
            flag1 = self.dfed.te[1][teidx].argmax().item() != te_preds[teidx]  # te record correctly classified?
            support_rules = {r: r2w[r][te_preds[teidx]] for r in te_rules[teidx] if r2w[r][te_preds[teidx]] > 0}
            support_rules = {k: v / sum(support_rules.values()) for k, v in support_rules.items()}
            support_rules_set = set(support_rules)
            traces, traces_weights = [], []  # trace tr_records for each te
            for user_i, idxs in enumerate(self.dfed.idxs):  # iter tr_records of each client
                for idx in idxs:
                    flag2 = self.dfed.tr[1][idx].argmax().item() != te_preds[teidx]  # tr record agree with te?
                    common_rules_set = support_rules_set.intersection(tr_rules[idx])
                    if not flag2 and not flag1:
                        for r in common_rules_set:
                            ruleW4clients[user_i][r] += support_rules[r]
                    weight_pcts = sum(support_rules[r] for r in common_rules_set)
                    if weight_pcts >= config.TRACE_THRES and not flag1 and not flag2:
                        traces.append(idx)
                        traces_weights.append(weight_pcts)
            if len(traces): micro_scores[traces] += np.array(traces_weights) * 1. / len(traces)
        return {'micro': [sum(micro_scores[idxs]) / sum(micro_scores) for idxs in self.dfed.idxs]}
