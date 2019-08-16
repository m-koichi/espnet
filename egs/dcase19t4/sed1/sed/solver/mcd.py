import sys
sys.path.append('./DCASE2019_task4/baseline')
sys.path.append('..')
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import AverageMeterSet, weights_init, ManyHotEncoder, SaveBest
from evaluation_measures import compute_strong_metrics
import os

from evaluate import get_batch_predictions_mcd


from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class Generator(nn.Module):
    def __init__(self, n_in_channel=1, activation="Relu", dropout=0, **kwargs):
        super(Generator, self).__init__()
        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)

    def forward(self, x):
        x = self.cnn(x)
        # ipdb.set_trace()
        bs, chan, frames, freq = x.size()
        # if freq != 1:
        #     warnings.warn("Output shape is: {}".format((bs, frames, chan * freq)))
        #     x = x.permute(0, 2, 1, 3)
        #     x = x.contiguous().view(bs, frames, chan * freq)
        # else:
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)  # [bs, frames, chan]
        return x


class Classifier(nn.Module):
    def __init__(self, prob=0.5, lambd=1.0, n_class=10, attention=False, dropout=0,
                 n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(Classifier, self).__init__()

        self.prob = prob
        self.lambd = lambd

        self.rnn = BidirectionalGRU(64,
                                    n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, n_class)
        self.attention = attention
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, n_class)
            self.softmax = nn.Softmax(dim=-1)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        # ipdb.set_trace()
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)
        return strong, weak


class MCDSolver:
    def __init__(self,
                 exp_name,
                 source_loader,
                 target_loader,
                 generator,
                 classifier1,
                 classifier2,
                 optimizer_g,
                 optimizer_f,
                 num_k=4,
                 num_multiply_d_loss=1):
        self.exp_name = exp_name
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.G = generator
        self.F1 = classifier1
        self.F2 = classifier2
        self.optimizer_g = optimizer_g
        self.optimizer_f = optimizer_f
        self.num_k = num_k
        self.num_multiply_d_loss = num_multiply_d_loss

    def train(self, epoch=0, lam=1.0):
        self.G = self.G.train()
        self.F1 = self.F1.train()
        self.F2 = self.F2.train()
        self.lam = lam

        class_criterion = nn.BCELoss().to('cuda')
        discrepancy_criterion = nn.L1Loss().cuda()
        if torch.cuda.is_available():
            self.G.cuda()
            self.F1.cuda()
            self.F2.cuda()

        # rampup_length = len(self.strong_loader) * cfg.n_epoch // 2
        for i, ((s_batch_input, s_target, _), (w_batch_input, w_target, _)) in \
                enumerate(zip(self.source_loader, self.target_loader)):

            s_batch_input = s_batch_input.to('cuda')
            s_target = s_target.to('cuda')
            w_batch_input = w_batch_input.to('cuda')
            w_target = w_target.to('cuda')

            # Step A
            # Update generator and classifier by source strong sample and target weak sample
            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()
            strong_class_loss = 0
            weak_class_loss = 0

            source_feature = self.G(s_batch_input)
            target_feature = self.G(w_batch_input)

            s_strong_pred1, s_weak_pred1 = self.F1(source_feature)
            w_strong_pred1, w_weak_pred1 = self.F1(target_feature)

            s_strong_pred2, s_weak_pred2 = self.F2(source_feature)
            w_strong_pred2, w_weak_pred2 = self.F2(target_feature)

            strong_class_loss += class_criterion(s_strong_pred1, s_target)
            strong_class_loss += class_criterion(s_strong_pred2, s_target)

            weak_class_loss += class_criterion(w_weak_pred1, w_target)
            weak_class_loss += class_criterion(w_weak_pred2, w_target)

            classification_loss = strong_class_loss + weak_class_loss
            classification_loss.backward(retain_graph=True)

            self.F1.set_lambda(self.lam)
            self.F2.set_lambda(self.lam)

            # self.optimizer_g.step()
            # self.optimizer_f.step()
            #
            # # Step B
            # # Update classifiers
            # self.optimizer_g.zero_grad()
            # self.optimizer_f.zero_grad()
            strong_class_loss = 0
            weak_class_loss = 0

            source_feature = self.G(s_batch_input)
            target_feature = self.G(w_batch_input)

            s_strong_pred1, s_weak_pred1 = self.F1(source_feature, reverse=True)
            w_strong_pred1, w_weak_pred1 = self.F1(target_feature, reverse=True)

            s_strong_pred2, s_weak_pred2 = self.F2(source_feature, reverse=True)
            w_strong_pred2, w_weak_pred2 = self.F2(target_feature, reverse=True)

            discrepancy_loss = - discrepancy_criterion(w_strong_pred1, w_strong_pred2)
            discrepancy_loss.backward()
            self.optimizer_f.step()
            self.optimizer_g.step()

            # strong_class_loss += class_criterion(s_strong_pred1, s_target)
            # strong_class_loss += class_criterion(s_strong_pred2, s_target)
            #
            # weak_class_loss += class_criterion(w_weak_pred1, w_target)
            # weak_class_loss += class_criterion(w_weak_pred2, w_target)
            #
            # discrepancy_loss = discrepancy_criterion(w_strong_pred1, w_strong_pred2)
            #
            # loss = strong_class_loss + weak_class_loss
            # loss -= discrepancy_loss
            #
            #
            #
            # self.optimizer_f.step()
            #
            # # Step C
            # # Update generator
            # for k in range(self.num_k):
            #     self.optimizer_g.zero_grad()
            #     loss = 0
            #
            #     source_feature = self.G(s_batch_input)
            #     target_feature = self.G(w_batch_input)
            #
            #     s_strong_pred1, s_weak_pred1 = self.F1(source_feature, reverse=True)
            #     w_strong_pred1, w_weak_pred1 = self.F1(target_feature, reverse=True)
            #
            #     s_strong_pred2, s_weak_pred2 = self.F2(source_feature, reverse=True)
            #     w_strong_pred2, w_weak_pred2 = self.F2(target_feature, reverse=True)
            #     loss += discrepancy_criterion(w_strong_pred1, w_strong_pred2) * self.num_multiply_d_loss
            #     loss.backward()
            #     self.optimizer_g.step()

    def train_no_adapt(self, epoch=0):
        self.G = self.G.train()
        self.F1 = self.F1.train()
        self.F2 = self.F2.train()

        class_criterion = nn.BCELoss().to('cuda')
        discrepancy_criterion = nn.L1Loss().cuda()
        if torch.cuda.is_available():
            self.G.cuda()
            self.F1.cuda()
            self.F2.cuda()

        # rampup_length = len(self.strong_loader) * cfg.n_epoch // 2
        for i, ((s_batch_input, s_target, _), (w_batch_input, w_target, _)) in \
                tqdm(enumerate(zip(self.source_loader, self.target_loader))):

            s_batch_input = s_batch_input.to('cuda')
            s_target = s_target.to('cuda')
            w_batch_input = w_batch_input.to('cuda')
            w_target = w_target.to('cuda')

            # Step A
            # Update generator and classifier by source strong sample and target weak sample
            self.optimizer_g.zero_grad()
            self.optimizer_f.zero_grad()
            strong_class_loss = 0
            weak_class_loss = 0

            source_feature = self.G(s_batch_input)
            target_feature = self.G(w_batch_input)

            s_strong_pred1, s_weak_pred1 = self.F1(source_feature)
            w_strong_pred1, w_weak_pred1 = self.F1(target_feature)

            s_strong_pred2, s_weak_pred2 = self.F2(source_feature)
            w_strong_pred2, w_weak_pred2 = self.F2(target_feature)

            strong_class_loss += class_criterion(s_strong_pred1, s_target)
            strong_class_loss += class_criterion(s_strong_pred2, s_target)

            weak_class_loss += class_criterion(w_weak_pred1, w_target)
            weak_class_loss += class_criterion(w_weak_pred2, w_target)

            loss = strong_class_loss + weak_class_loss
            loss.backward()

            self.optimizer_g.step()
            self.optimizer_f.step()


    def test(self, validation_df, test_loader, many_hot_encoder, epoch):
        self.G = self.G.eval()
        self.F1 = self.F1.eval()
        self.F2 = self.F2.eval()

        with torch.no_grad():
            predictions = get_batch_predictions_mcd(self.G, self.F1, test_loader, many_hot_encoder.decode_strong,
                                                save_predictions=os.path.join(self.exp_name, 'predictions',
                                                                              f'result_{epoch}.csv'))
            valid_events_metric = compute_strong_metrics(predictions, validation_df, 8)

    def set_eval(self):
        self.G = self.G.eval()
        self.F1 = self.F1.eval()
        self.F2 = self.F2.eval()

    def get_batch_predictions(self, test_loader, many_hot_encoder, epoch):
        with torch.no_grad():
            predictions = get_batch_predictions_mcd(self.G, self.F1, test_loader, many_hot_encoder.decode_strong,
                                                save_predictions=os.path.join(self.exp_name, 'predictions',
                                                                              f'result_{epoch}.csv'))
        return predictions



