import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from learner_1 import Learner_1
from learner_2 import Learner_2

from sklearn import metrics
from chemical import Chemical
from scaling_sgc import Scaling
from translation_sgc import Translation
import math


class Meta_MSNA(nn.Module):
    def __init__(self, config, config_chemi, config_scal, config_trans, args, num_attri, label_dim):
        super(Meta_MSNA, self).__init__()
        self.task_lr = args.task_lr
        self.meta_lr = args.meta_lr
        self.label_dim = label_dim
        self.hidden = args.hidden

        self.query_weight = nn.ParameterList()
        query_weight_0 = nn.Parameter(torch.FloatTensor(self.hidden, self.hidden))
        query_weight_1 = nn.Parameter(torch.FloatTensor(label_dim, label_dim))
        stdv_0 = 1. / math.sqrt(self.hidden)
        stdv_1 = 1. / math.sqrt(label_dim)
        query_weight_0.data.uniform_(-stdv_0, stdv_0)
        query_weight_1.data.uniform_(-stdv_1, stdv_1)
        self.query_weight.append(query_weight_0)
        self.query_weight.append(query_weight_1)
        
        self.net = Learner_1(config)
        self.chemical = Chemical(config_chemi)
        self.scaling = Scaling(config_scal, args, num_attri, label_dim)
        self.translation = Translation(config_trans, args, num_attri, label_dim)
        self.meta_optim = optim.Adam([{'params': self.net.parameters()}, {'params': self.chemical.parameters()},
                                      {'params': self.scaling.parameters()}, {'params': self.translation.parameters()},
                                      {'params': self.query_weight}], lr=self.meta_lr)
        self.dataset = args.dataset

    def forward(self, x_spt, y_spt, x_qry, y_qry, idx_spt_list, idx_qry_list, features_list, neighs_list, 
                l2_coef, Lab, update_step, batch_size, training):
        training = training
        task_num = len(x_spt)
        update_step = update_step
        Losses_q = [0 for _ in range(update_step + 1)]
        accs = 0

        all_predictions = []
        all_predictions_f = []
        all_trues = []
        all_trues_f = []
        for j in range(int(task_num / batch_size) if task_num % batch_size == 0 else int(task_num / batch_size) + 1):
            start_idx = j * batch_size
            end_idx = min(start_idx + batch_size, task_num)
            losses_q = [0 for _ in range(update_step + 1)]
            for i in range(start_idx, end_idx):
                # print("neighs_list[i]:", neighs_list[i])
                logits_1, logits_2 = self.net(features_list[i], neighs_list[i])

                instant_1 = torch.relu(torch.mm(torch.mean(logits_1, dim=0, keepdim=True), self.query_weight[0]))
                inside_sigmoid_1 = torch.mm(logits_1, instant_1.T)
                att_weight_1 = torch.sigmoid(inside_sigmoid_1)
                graph_signal_1 = torch.sum(att_weight_1 * logits_1, dim=0)

                instant_2 = torch.relu(torch.mm(torch.mean(logits_2, dim=0, keepdim=True), self.query_weight[1]))
                inside_sigmoid_2 = torch.mm(logits_2, instant_2.T)
                att_weight_2 = torch.sigmoid(inside_sigmoid_2)
                graph_signal_2 = torch.sum(att_weight_2 * logits_2, dim=0)

                chemical = torch.cat((graph_signal_1, graph_signal_2), dim=0)
                chemical = self.chemical(chemical)
                scaling = self.scaling(chemical)
                translation = self.translation(chemical)
                adapted_prior = []
                for s in range(len(scaling)):
                    adapted_prior.append(torch.mul(self.net.parameters()[s], (scaling[s] + 1)) + translation[s])
                logits_1, logits_2 = self.net(features_list[i], neighs_list[i], adapted_prior)
                logit_spt = logits_2[idx_spt_list[i]]
                if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                    loss = torch.nn.BCEWithLogitsLoss()
                    loss = loss(logit_spt, y_spt[i])
                else:
                    loss = torch.nn.functional.cross_entropy(logit_spt, y_spt[i])
                grad = torch.autograd.grad(loss, adapted_prior)
                fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, adapted_prior)))

                if update_step == 1:
                    logits_1, logits_2 = self.net(features_list[i], neighs_list[i], fast_weights)
                    logits_q = logits_2[idx_qry_list[i]]
                    if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                        loss_q = torch.nn.BCEWithLogitsLoss()
                        loss_q = loss_q(logits_q, y_qry[i])
                    else:
                        loss_q = F.cross_entropy(logits_q, y_qry[i])
                    l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                    l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                    l2_loss = l2_loss * l2_coef
                    losses_q[1] += (loss_q + l2_loss)
                    Losses_q[1] += (loss_q + l2_loss)
                else:
                    for k in range(1, update_step):
                        logits_1, logits_2 = self.net(features_list[i], neighs_list[i], fast_weights)
                        logit_spt = logits_2[idx_spt_list[i]]
                        if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                            loss = torch.nn.BCEWithLogitsLoss()
                            loss = loss(logit_spt, y_spt[i])
                        else:
                            loss = F.cross_entropy(logit_spt, y_spt[i])
                        grad = torch.autograd.grad(loss, fast_weights)
                        fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))
                        if k == update_step - 1:
                            logits_1, logits_2 = self.net(features_list[i], neighs_list[i], fast_weights)
                            logits_q = logits_2[idx_qry_list[i]]
                            if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                                loss_q = torch.nn.BCEWithLogitsLoss()
                                loss_q = loss_q(logits_q, y_qry[i])
                            else:
                                loss_q = F.cross_entropy(logits_q, y_qry[i])
                            l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                            l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                            l2_loss = l2_loss * l2_coef
                            losses_q[k + 1] += (loss_q + l2_loss)
                            Losses_q[k + 1] += (loss_q + l2_loss)
                with torch.no_grad():
                    if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                        pred_q = torch.sigmoid(logits_q)
                        pred_q = torch.round(pred_q)
                        y_true = []
                        y_pred = []
                        for m in range(len(y_qry[i])):
                            for n in range(self.label_dim):
                                y_true.append((y_qry[i])[m, n].cpu())
                        for m in range(len(pred_q)):
                            for n in range(self.label_dim):
                                y_pred.append(pred_q[m, n].cpu())
                    else:
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                        all_trues.append(y_true)
                        all_predictions.append(y_pred)
                        all_trues_f.append((y_qry[i].cpu()).numpy())
                        all_predictions_f.append((pred_q.cpu()).numpy())
                    else:
                        all_trues.append((y_qry[i].cpu()).numpy())
                        all_predictions.append((pred_q.cpu()).numpy())

            loss_q = losses_q[-1] / batch_size
            if training == True:
                self.meta_optim.zero_grad()
                loss_q.backward()
                self.meta_optim.step()

        all_trues = np.concatenate(all_trues)
        all_predictions = np.concatenate(all_predictions)

        acc = metrics.accuracy_score(all_trues, all_predictions, normalize=True)
        if self.dataset in ['Cuneiform', 'Sub_Yelp']:
            all_trues_f = np.concatenate(all_trues_f)
            all_predictions_f = np.concatenate(all_predictions_f)
            MiF1s = metrics.f1_score(all_trues_f, all_predictions_f, labels=Lab,
                                     average='micro')
        else:
            MiF1s = metrics.f1_score(all_trues, all_predictions, labels=Lab, average='micro')

        Loss_q = Losses_q[-1] / task_num
        return acc, Loss_q, MiF1s



class Meta_NA(nn.Module):
    def __init__(self, config, config_chemi, config_scal, config_trans, args, num_attri, label_dim):
        super(Meta_NA, self).__init__()
        self.task_lr = args.task_lr
        self.meta_lr = args.meta_lr
        self.label_dim = label_dim

        self.query_weight = nn.ParameterList()
        query_weight_1 = nn.Parameter(torch.FloatTensor(label_dim, 1))
        stdv = 1. / math.sqrt(label_dim)
        query_weight_1.data.uniform_(-stdv, stdv)
        self.query_weight.append(query_weight_1)

        self.net = Learner_2(config)
        self.chemical = Chemical(config_chemi)
        self.scaling = Scaling(config_scal, args, num_attri, label_dim)
        self.translation = Translation(config_trans, args, num_attri, label_dim)
        self.meta_optim = optim.Adam([{'params': self.net.parameters()}, {'params': self.chemical.parameters()},
                                      {'params': self.scaling.parameters()}, {'params': self.translation.parameters()}],
                                     lr=self.meta_lr)
        self.dataset = args.dataset

    def forward(self, x_spt, y_spt, x_qry, y_qry, idx_spt_list, idx_qry_list, features_list, neighs_list,
                l2_coef, Lab, update_step, batch_size, training):
        training = training
        task_num = len(x_spt)
        all_predictions = []
        all_predictions_f = []
        all_trues = []
        all_trues_f = []
        Losses_q = [0 for _ in range(update_step + 1)]
        for j in range(int(task_num / batch_size) if task_num % batch_size == 0 else int(task_num / batch_size) + 1):
            start_idx = j * batch_size
            end_idx = min(start_idx + batch_size, task_num)
            losses_q = [0 for _ in range(update_step + 1)]
            for i in range(start_idx, end_idx):
                logits_2 = self.net(features_list[i])
                att_weight_2 = F.softmax(torch.mm(logits_2, self.query_weight[0]), dim=0)
                graph_signal_2 = torch.sum(att_weight_2 * logits_2, dim=0)
                chemical = self.chemical(graph_signal_2)
                scaling = self.scaling(chemical)
                translation = self.translation(chemical)
                adapted_prior = []
                for s in range(len(scaling)):
                    adapted_prior.append(torch.mul(self.net.parameters()[s], (scaling[s] + 1)) + translation[s])
                logits = self.net(x_spt[i], adapted_prior)
                if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                    loss = torch.nn.BCEWithLogitsLoss()
                    loss = loss(logits, y_spt[i])
                else:
                    loss = torch.nn.functional.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, adapted_prior)
                fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, adapted_prior)))

                if update_step == 1:
                    logits_q = self.net(x_qry[i], fast_weights)
                    if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                        loss_q = torch.nn.BCEWithLogitsLoss()
                        loss_q = loss_q(logits_q, y_qry[i])
                    else:
                        loss_q = F.cross_entropy(logits_q, y_qry[i])
                    l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                    l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                    l2_loss = l2_loss * l2_coef
                    losses_q[1] += (loss_q + l2_loss)
                    Losses_q[1] += (loss_q + l2_loss)
                else:
                    for k in range(1, update_step):
                        logits = self.net(x_spt[i], fast_weights)
                        if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                            loss = torch.nn.BCEWithLogitsLoss()
                            loss = loss(logits, y_spt[i])
                        else:
                            loss = F.cross_entropy(logits, y_spt[i])
                        grad = torch.autograd.grad(loss, fast_weights)
                        fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))
                        if k == update_step - 1:
                            logits_q = self.net(x_qry[i], fast_weights)
                            if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                                loss_q = torch.nn.BCEWithLogitsLoss()
                                loss_q = loss_q(logits_q, y_qry[i])
                            else:
                                loss_q = F.cross_entropy(logits_q, y_qry[i])
                            l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                            l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                            l2_loss = l2_loss * l2_coef
                            losses_q[k + 1] += (loss_q + l2_loss)
                            Losses_q[k + 1] += (loss_q + l2_loss)
                with torch.no_grad():
                    if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                        pred_q = torch.sigmoid(logits_q)
                        pred_q = torch.round(pred_q)
                        y_true = []
                        y_pred = []
                        for m in range(len(y_qry[i])):
                            for n in range(self.label_dim):
                                y_true.append((y_qry[i])[m, n].cpu())
                        for m in range(len(pred_q)):
                            for n in range(self.label_dim):
                                y_pred.append(pred_q[m, n].cpu())
                    else:
                        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    if self.dataset in ['Cuneiform', 'Sub_Yelp']:
                        all_trues.append(y_true)
                        all_predictions.append(y_pred)
                        all_trues_f.append((y_qry[i].cpu()).numpy())
                        all_predictions_f.append((pred_q.cpu()).numpy())
                    else:
                        all_trues.append((y_qry[i].cpu()).numpy())
                        all_predictions.append((pred_q.cpu()).numpy())

            loss_q = losses_q[-1] / batch_size
            if training == True:
                self.meta_optim.zero_grad()
                loss_q.backward()
                self.meta_optim.step()

        all_trues = np.concatenate(all_trues)
        all_predictions = np.concatenate(all_predictions)

        acc = metrics.accuracy_score(all_trues, all_predictions, normalize=True)
        if self.dataset in ['Cuneiform', 'Sub_Yelp']:
            all_trues_f = np.concatenate(all_trues_f)
            all_predictions_f = np.concatenate(all_predictions_f)
            MiF1s = metrics.f1_score(all_trues_f, all_predictions_f, labels=Lab,
                                     average='micro')
        else:
            MiF1s = metrics.f1_score(all_trues, all_predictions, labels=Lab, average='micro')

        Loss_q = Losses_q[-1] / task_num
        return acc, Loss_q, MiF1s
