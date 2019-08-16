import torch

from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from dcase_util.data import ProbabilityEncoder
from utils.utils import AverageMeterSet, weights_init, ManyHotEncoder, SaveBest
from utils import ramps
import pandas as pd
import re

import logging
import math

from solver.baseline_model import CNN

from logger import Logger

logging.basicConfig(level=logging.DEBUG)


class Transformer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 n_class,
                 args,
                 pooling='attention',
                 input_conv=False,
                 cnn_kwargs=None):
        super(Transformer, self).__init__()
        self.args = args
        self.input_conv = input_conv
        if input_conv:
            self.cnn = CNN(n_in_channel=1, activation="Relu", conv_dropout=args.dropout, **cnn_kwargs)
        self.encoder = Encoder(input_dim, args)
        self.classifier = torch.nn.Linear(args.adim, n_class)
        self.dense = torch.nn.Linear(args.adim, n_class)
        self.sigmoid = torch.sigmoid
        self.softmax = torch.nn.Softmax(dim=-1)
        self.pooling = pooling
        self.reset_parameters(args)

    def forward(self, x, mask=None):
        if self.input_conv:
            x = self.cnn(x)
            x = x.squeeze(-1).permute(0, 2, 1)
        if self.args.input_layer_type == 3:
            x = x.squeeze(1)
        # import ipdb
        # ipdb.set_trace()
        x, _ = self.encoder(x, mask)
        strong = torch.sigmoid(self.classifier(x))
        if self.pooling == 'attention':
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        elif self.pooling == 'mean':
            weak = strong.mean(1)
        elif self.pooling == 'max':
            weak = strong.max(1)[0]
        return strong, weak

    def reset_parameters(self, args):
        if args.transformer_init == "pytorch":
            return
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if args.transformer_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif args.transformer_init == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif args.transformer_init == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif args.transformer_init == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + args.transformer_init)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()
        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (torch.nn.Embedding, LayerNorm)):
                m.reset_parameters()

    def save(self, filename):
        parameters = {
            'encoder'   : self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'dense'     : self.dense.state_dict()
        }
        if self.input_conv:
            parameters['cnn'] = self.cnn.state_dict()
        torch.save(parameters, filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        self.encoder.load_state_dict(parameters["encoder"])
        self.classifier.load_state_dict(parameters["classifier"])
        self.dense.load_state_dict(parameters["dense"])
        if self.input_conv:
            self.cnn.load_state_dict(parameters["cnn"])


class TransformerSolver(object):
    def __init__(self,
                 model,
                 ema_model,
                 strong_loader,
                 weak_loader,
                 unlabel_loader,
                 args,
                 criterion,
                 consistency_criterion,
                 accum_grad,
                 rampup_length,
                 exp_name='tensorboard/log',
                 optimizer='noam',
                 consistency_cost=2,
                 data_parallel=True):
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.ema_model = ema_model.cuda() if torch.cuda.is_available() else ema_model
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.ema_model = torch.nn.DataParallel(self.ema_model)

        self.criterion = criterion
        self.consistency_criterion = consistency_criterion
        self.optimizer = optimizer
        self.accum_grad = accum_grad
        self.grad_clip_threshold = 5

        self.strong_loader = strong_loader
        self.weak_loader = weak_loader
        self.unlabel_loader = unlabel_loader

        self.forward_count = 0
        self.rampup_length = rampup_length
        self.set_optimizer(args)
        self.args = args

        self.max_consistency_cost = consistency_cost
        self.strong_losses = AverageMeterSet()
        self.weak_losses = AverageMeterSet()
        self.logger = Logger(exp_name.replace('exp', 'tensorboard'))
        # self.criterion = LabelSmoothingLoss(n_class, -1, 0.1, False, criterion)

    def set_optimizer(self, args):
        self.optimizer = get_std_opt(self.model, args.adim, args.transformer_warmup_steps, args.transformer_lr)

    def train_one_step(self, log_interbal=100, warm_start=True):
        self.model.train()
        try:
            strong_sample, strong_target, _ = next(self.strong_iter)
        except:
            self.strong_iter = iter(self.strong_loader)
            strong_sample, strong_target, _ = next(self.strong_iter)
        try:
            weak_sample, weak_target, _ = next(self.weak_iter)
        except:
            self.weak_iter = iter(self.weak_loader)
            weak_sample, weak_target, _ = next(self.weak_iter)
        # try:
        #     unlabel_sample, _, _ = next(self.unlabel_iter)
        # except:
        #     self.unlabel_iter = iter(self.unlabel_loader)
        #     unlabel_sample, _, _ = next(self.unlabel_iter)

        strong_sample = strong_sample.cuda()
        weak_sample = weak_sample.cuda()
        # unlabel_sample = unlabel_sample.squeeze(1).cuda()
        strong_target = strong_target.cuda()
        weak_target = weak_target.cuda()

        pred_strong, pred_weak = self.model(strong_sample)
        strong_loss = self.criterion(pred_strong, strong_target)

        pred_strong, pred_weak = self.model(weak_sample)
        weak_loss = self.criterion(pred_weak, weak_target)

        loss = (strong_loss + weak_loss) / self.accum_grad
        loss.backward() # Backprop
        loss.detach() # Truncate the graph

        self.logger.scalar_summary('train_strong_loss', strong_loss.item(), self.forward_count)
        self.logger.scalar_summary('train_weak_loss', weak_loss.item(), self.forward_count)

        self.forward_count += 1
        if self.forward_count % log_interbal == 0:
            logging.info('After {} iteration'.format(self.forward_count))
            logging.info('\tstrong loss: {}'.format(strong_loss.item()))
            logging.info('\tweak loss: {}'.format(weak_loss.item()))
        if self.forward_count % self.accum_grad == 0:
            return
        # self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

    def train_one_step_ema(self, log_interbal=100, warm_start=True):
        self.model.train()
        try:
            strong_sample, strong_target, strong_ids = next(self.strong_iter)
            strong_sample_ema, strong_target_ema, strong_ids_ema = next(self.strong_iter_ema)
        except:
            self.strong_iter = iter(self.strong_loader)
            strong_sample, strong_target, strong_ids = next(self.strong_iter)
            self.strong_iter_ema = iter(self.strong_loader)
            strong_sample_ema, strong_target_ema, strong_ids_ema = next(self.strong_iter_ema)
        try:
            weak_sample, weak_target, _ = next(self.weak_iter)
            weak_sample_ema, weak_target_ema, _ = next(self.weak_iter_ema)
        except:
            self.weak_iter = iter(self.weak_loader)
            weak_sample, weak_target, _ = next(self.weak_iter)
            self.weak_iter_ema = iter(self.weak_loader)
            weak_sample_ema, weak_target_ema, _ = next(self.weak_iter_ema)
        try:
            unlabel_sample, _, _ = next(self.unlabel_iter)
            unlabel_sample_ema, _, _ = next(self.unlabel_iter_ema)
        except:
            self.unlabel_iter = iter(self.unlabel_loader)
            unlabel_sample, _, _ = next(self.unlabel_iter)
            self.unlabel_iter_ema = iter(self.unlabel_loader)
            unlabel_sample_ema, _, _ = next(self.unlabel_iter_ema)

        assert strong_ids == strong_ids_ema


        if self.forward_count < self.rampup_length:
            rampup_value = ramps.sigmoid_rampup(self.forward_count, self.rampup_length)
        else:
            rampup_value = 1.0

        strong_sample, strong_sample_ema = strong_sample.cuda(), strong_sample_ema.cuda()
        weak_sample, weak_sample_ema = weak_sample.cuda(), weak_sample_ema.cuda()
        unlabel_sample, unlabel_sample_ema = unlabel_sample.cuda(), unlabel_sample_ema.cuda()
        strong_target, strong_target_ema = strong_target.cuda(), strong_target_ema.cuda()
        weak_target, weak_target_ema = weak_target.cuda(), weak_target_ema.cuda()

        # pred_strong, pred_weak = self.model(strong_sample)
        # strong_loss = self.criterion(pred_strong, strong_target)
        #
        # pred_strong, pred_weak = self.model(weak_sample)
        # weak_loss = self.criterion(pred_weak, weak_target)


        if warm_start and self.forward_count < self.args.transformer_warmup_steps / 2:
            pred_strong_ema_w, pred_weak_ema_w = self.ema_model(weak_sample_ema)
            pred_strong_ema_u, pred_weak_ema_u = self.ema_model(unlabel_sample_ema)
            pred_strong_ema_w, pred_strong_ema_u = \
                pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
            pred_weak_ema_u, pred_weak_ema_w = \
                pred_weak_ema_w.detach(), pred_weak_ema_u.detach()
        else:
            pred_strong_ema_s, pred_weak_ema_s = self.ema_model(strong_sample_ema)
            pred_strong_ema_w, pred_weak_ema_w = self.ema_model(weak_sample_ema)
            pred_strong_ema_u, pred_weak_ema_u = self.ema_model(unlabel_sample_ema)
            pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
                pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
            pred_weak_ema_s, pred_weak_ema_u, pred_weak_ema_w = \
                pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()

        if warm_start and self.forward_count < self.args.transformer_warmup_steps / 2:
            pred_strong_w, pred_weak_w = self.model(weak_sample)
            pred_strong_u, pred_weak_u = self.model(unlabel_sample)
            weak_class_loss = self.criterion(pred_weak_w, weak_target)
        else:
            pred_strong_s, pred_weak_s = self.model(strong_sample)
            pred_strong_w, pred_weak_w = self.model(weak_sample)
            pred_strong_u, pred_weak_u = self.model(unlabel_sample)
            strong_class_loss = self.criterion(pred_strong_s, strong_target)
            weak_class_loss = self.criterion(pred_weak_w, weak_target)

        # compute consistency loss
        consistency_cost = self.max_consistency_cost * rampup_value
        if warm_start and self.forward_count < self.args.transformer_warmup_steps / 2:
            consistency_loss_weak = consistency_cost * self.consistency_criterion(pred_weak_w, pred_weak_ema_w) \
                                    + consistency_cost * self.consistency_criterion(pred_weak_u, pred_weak_ema_u)

            loss = (
                           weak_class_loss + consistency_loss_weak) / self.accum_grad
            loss.backward()  # Backprop
            loss.detach()  # Truncate the graph

            # self.logger.scalar_summary('train_strong_loss', strong_class_loss.item(), self.forward_count)
            self.logger.scalar_summary('train_weak_loss', weak_class_loss.item(), self.forward_count)

            self.forward_count += 1
            if self.forward_count % log_interbal == 0:
                logging.info('After {} iteration'.format(self.forward_count))
                # logging.info('\tstrong loss: {}'.format(strong_class_loss.item()))
                logging.info('\tweak loss: {}'.format(weak_class_loss.item()))
        else:
            consistency_loss_strong = consistency_cost * self.consistency_criterion(pred_strong_s, pred_strong_ema_s) \
                                      + consistency_cost * self.consistency_criterion(pred_strong_w, pred_strong_ema_w) \
                                      + consistency_cost * self.consistency_criterion(pred_strong_u, pred_strong_ema_u)
            consistency_loss_weak = consistency_cost * self.consistency_criterion(pred_weak_s, pred_weak_ema_s) \
                                    + consistency_cost * self.consistency_criterion(pred_weak_w, pred_weak_ema_w) \
                                    + consistency_cost * self.consistency_criterion(pred_weak_u, pred_weak_ema_u)

            loss = (strong_class_loss + weak_class_loss + consistency_loss_strong + consistency_loss_weak) / self.accum_grad
            loss.backward() # Backprop
            loss.detach() # Truncate the graph

            self.logger.scalar_summary('train_strong_loss', strong_class_loss.item(), self.forward_count)
            self.logger.scalar_summary('train_weak_loss', weak_class_loss.item(), self.forward_count)

            self.forward_count += 1
            if self.forward_count % log_interbal == 0:
                logging.info('After {} iteration'.format(self.forward_count))
                logging.info('\tstrong loss: {}'.format(strong_class_loss.item()))
                logging.info('\tweak loss: {}'.format(weak_class_loss.item()))
        if self.forward_count % self.accum_grad == 0:
            return
        # self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_threshold)
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()

        self.update_ema_variables(alpha=0.999)

    def update_ema_variables(self, alpha):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (self.forward_count + 1), alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


    def get_predictions(self, data_loader, decoder, threshold=0.5, binarization_type='global_threshold',
                              post_processing=None, save_predictions=None, mode='validation'):
        prediction_df = pd.DataFrame()

        avg_strong_loss = 0
        avg_weak_loss = 0

        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_input, batch_target, data_ids) in enumerate(data_loader):
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()

                pred_strong, pred_weak = self.model(batch_input)

                if mode == 'validation':
                    class_criterion = torch.nn.BCELoss().cuda()
                    target = batch_target.cuda()
                    strong_class_loss = class_criterion(pred_strong, target)
                    weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
                    avg_strong_loss += strong_class_loss.item() / len(data_loader)
                    avg_weak_loss += weak_class_loss.item() / len(data_loader)

                pred_strong = pred_strong.cpu().data.numpy()
                pred_weak = pred_weak.cpu().data.numpy()

                if binarization_type == 'class_threshold':
                    for i in range(pred_strong.shape[0]):
                        pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i],
                                                                           binarization_type=binarization_type,
                                                                           threshold=threshold, time_axis=0)
                else:
                    pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
                                                                    threshold=threshold)

                if post_processing is not None:
                    for i in range(pred_strong.shape[0]):
                        for post_process_fn in post_processing:
                            pred_strong[i] = post_process_fn(pred_strong[i])

                for pred, data_id in zip(pred_strong, data_ids):
                    pred = decoder(pred)
                    pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
                    pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
                    prediction_df = prediction_df.append(pred)

        if save_predictions is not None:
            logging.info("Saving predictions at: {}".format(save_predictions))
            prediction_df.to_csv(save_predictions, index=False, sep="\t")

        if mode == 'validation':
            logging.info(f'\tAve. valid strong class loss: {avg_strong_loss}')
            logging.info(f'\tAve. valid weak class loss: {avg_weak_loss}')
            self.logger.scalar_summary('valid_strong_loss', avg_strong_loss, self.forward_count)
            self.logger.scalar_summary('valid_weak_loss', avg_weak_loss, self.forward_count)

        return prediction_df

    def save(self, filename, ema_filename):
        self.model.save(filename)
        self.ema_model.save(ema_filename)

#
# class MeanTeacherTrainer():
#     pass

# class TransformerSolver(torch.nn.Module):
#
#     def __init__(self,
#                  strong_loader,
#                  weak_loader,
#                  unlabel_loader,
#                  input_dim,
#                  n_class,
#                  args,
#                  criterion,
#                  accum_grad,
#                  exp_name='tensorboard/log',
#                  optimizer='noam',
#                  pooling='attention',
#                  data_parallel=True,
#                  arch=1):
#         torch.nn.Module.__init__(self)
#         # self.strong_iter = strong_iter
#         # self.weak_iter = weak_iter
#         # self.unlabel_iter = unlabel_iter
#
#         if arch == 1:
#             self.cnn = CNN(n_in_channel=1, activation="Relu", conv_dropout=args.dropout)
#             self.encoder = Encoder(64, args)
#         elif arch == 2:
#             self.encoder = Encoder(1, args, no_stride=True)
#         elif arch == 3:
#             self.encoder = Encoder(1, args)
#         self.classifier = torch.nn.Linear(args.adim, n_class)
#         self.dense = torch.nn.Linear(args.adim, n_class)
#         if data_parallel:
#             self.encoder = torch.nn.DataParallel(self.encoder)
#             self.classifier = torch.nn.DataParallel(self.classifier)
#             self.dense = torch.nn.DataParallel(self.dense)
#
#         self.sigmoid = torch.sigmoid
#         self.softmax = torch.nn.Softmax(dim=-1)
#         self.pooling = pooling
#
#         # self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
#         #                                     args.transformer_length_normalized_loss)
#         self.reset_parameters(args)
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.accum_grad = accum_grad
#         self.grad_clip_threshold = 5
#
#         self.strong_loader = strong_loader
#         self.weak_loader = weak_loader
#         self.unlabel_loader = unlabel_loader
#
#         self.forward_count = 0
#         self.set_optimizer(args)
#
#         self.strong_losses = AverageMeterSet()
#         self.weak_losses = AverageMeterSet()
#         self.logger = Logger(exp_name.replace('exp', 'tensorboard'))
#         # self.criterion = LabelSmoothingLoss(n_class, -1, 0.1, False, criterion)
#
#     def forward(self, x, mask=None):
#         x = self.cnn(x)
#
#         bs, chan, frames, freq = x.size()
#         if freq != 1:
#             # warnings.warn("Output shape is: {}".format((bs, frames, chan * freq)))
#             x = x.permute(0, 2, 1, 3)
#             x = x.contiguous().view(bs, frames, chan * freq)
#         else:
#             x = x.squeeze(-1)
#             x = x.permute(0, 2, 1)  # [bs, frames, chan]
#
#         x, _ = self.encoder(x, mask)
#         strong = torch.sigmoid(self.classifier(x))
#         if self.pooling == 'attention':
#             sof = self.dense(x)  # [bs, frames, nclass]
#             sof = self.softmax(sof)
#             sof = torch.clamp(sof, min=1e-7, max=1)
#             weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
#         elif self.pooling == 'mean':
#             weak = strong.mean(1)
#         elif self.pooling == 'max':
#             weak = strong.max(1)[0]
#         return strong, weak
#
#     def set_optimizer(self, args):
#         self.optimizer = get_std_opt(self, args.adim, args.transformer_warmup_steps, args.transformer_lr)
#
#     def reset_parameters(self, args):
#         if args.transformer_init == "pytorch":
#             return
#         # weight init
#         for p in self.parameters():
#             if p.dim() > 1:
#                 if args.transformer_init == "xavier_uniform":
#                     torch.nn.init.xavier_uniform_(p.data)
#                 elif args.transformer_init == "xavier_normal":
#                     torch.nn.init.xavier_normal_(p.data)
#                 elif args.transformer_init == "kaiming_uniform":
#                     torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
#                 elif args.transformer_init == "kaiming_normal":
#                     torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
#                 else:
#                     raise ValueError("Unknown initialization: " + args.transformer_init)
#         # bias init
#         for p in self.parameters():
#             if p.dim() == 1:
#                 p.data.zero_()
#         # reset some modules with default init
#         for m in self.modules():
#             if isinstance(m, (torch.nn.Embedding, LayerNorm)):
#                 m.reset_parameters()
#
#     def train_one_step(self, log_interbal=100):
#         try:
#             strong_sample, strong_target, _ = next(self.stong_iter)
#         except:
#             self.strong_iter = iter(self.strong_loader)
#             strong_sample, strong_target, _ = next(self.strong_iter)
#         try:
#             weak_sample, weak_target, _ = next(self.weak_iter)
#         except:
#             self.weak_iter = iter(self.weak_loader)
#             weak_sample, weak_target, _ = next(self.weak_iter)
#         try:
#             unlabel_sample, _, _ = next(self.unlabel_iter)
#         except:
#             self.unlabel_iter = iter(self.unlabel_loader)
#             unlabel_sample, _, _ = next(self.unlabel_iter)
#
#         strong_sample = strong_sample.squeeze(1).cuda()
#         weak_sample = weak_sample.squeeze(1).cuda()
#         unlabel_sample = unlabel_sample.squeeze(1).cuda()
#         strong_target = strong_target.cuda()
#         weak_target = weak_target.cuda()
#
#         pred_strong, pred_weak = self.forward(strong_sample)
#         strong_loss = self.criterion(pred_strong, strong_target)
#
#         pred_strong, pred_weak = self.forward(weak_sample)
#         weak_loss = self.criterion(pred_weak, weak_target)
#
#         loss = (strong_loss + weak_loss) / self.accum_grad
#         loss.backward() # Backprop
#         loss.detach() # Truncate the graph
#
#         self.logger.scalar_summary('train_strong_loss', strong_loss.item(), self.forward_count)
#         self.logger.scalar_summary('train_weak_loss', weak_loss.item(), self.forward_count)
#
#         self.forward_count += 1
#         if self.forward_count % log_interbal == 0:
#             logging.info('After {} iteration'.format(self.forward_count))
#             logging.info('\tstrong loss: {}'.format(strong_loss.item()))
#             logging.info('\tweak loss: {}'.format(weak_loss.item()))
#         if self.forward_count % self.accum_grad == 0 and self.accum_grad != 1:
#             return
#         # self.forward_count = 0
#         # compute the gradient norm to check if it is normal or not
#         grad_norm = torch.nn.utils.clip_grad_norm_(
#                 self.encoder.parameters(), self.grad_clip_threshold)
#         logging.info('grad norm={}'.format(grad_norm))
#         if math.isnan(grad_norm):
#             logging.warning('grad norm is nan. Do not update model.')
#         else:
#             self.optimizer.step()
#         self.optimizer.zero_grad()
#
#     def get_predictions(self, data_loader, decoder, threshold=0.5, binarization_type='global_threshold',
#                               post_processing=None, save_predictions=None, mode='validation'):
#         prediction_df = pd.DataFrame()
#
#         avg_strong_loss = 0
#         avg_weak_loss = 0
#
#         with torch.no_grad():
#             for batch_idx, (batch_input, batch_target, data_ids) in enumerate(data_loader):
#                 if torch.cuda.is_available():
#                     batch_input = batch_input.cuda()
#
#                 batch_input = batch_input.squeeze(1)
#                 pred_strong, pred_weak = self.forward(batch_input)
#
#                 if mode == 'validation':
#                     class_criterion = torch.nn.BCELoss().cuda()
#                     target = batch_target.cuda()
#                     strong_class_loss = class_criterion(pred_strong, target)
#                     weak_class_loss = class_criterion(pred_weak, target.max(-2)[0])
#                     avg_strong_loss += strong_class_loss.item() / len(data_loader)
#                     avg_weak_loss += weak_class_loss.item() / len(data_loader)
#
#                 pred_strong = pred_strong.cpu().data.numpy()
#                 pred_weak = pred_weak.cpu().data.numpy()
#
#                 if binarization_type == 'class_threshold':
#                     for i in range(pred_strong.shape[0]):
#                         pred_strong[i] = ProbabilityEncoder().binarization(pred_strong[i],
#                                                                            binarization_type=binarization_type,
#                                                                            threshold=threshold, time_axis=0)
#                 else:
#                     pred_strong = ProbabilityEncoder().binarization(pred_strong, binarization_type=binarization_type,
#                                                                     threshold=threshold)
#
#                 if post_processing is not None:
#                     for i in range(pred_strong.shape[0]):
#                         for post_process_fn in post_processing:
#                             pred_strong[i] = post_process_fn(pred_strong[i])
#
#                 for pred, data_id in zip(pred_strong, data_ids):
#                     pred = decoder(pred)
#                     pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
#                     pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
#                     prediction_df = prediction_df.append(pred)
#
#         if save_predictions is not None:
#             logging.info("Saving predictions at: {}".format(save_predictions))
#             prediction_df.to_csv(save_predictions, index=False, sep="\t")
#
#         if mode == 'validation':
#             logging.info(f'\tAve. valid strong class loss: {avg_strong_loss}')
#             logging.info(f'\tAve. valid weak class loss: {avg_weak_loss}')
#             self.logger.scalar_summary('valid_strong_loss', avg_strong_loss, self.forward_count)
#             self.logger.scalar_summary('valid_weak_loss', avg_weak_loss, self.forward_count)
#
#         return prediction_df
#
#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         state_dict = {
#             'encoder': self.encoder.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
#             'classifier': self.classifier.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
#             'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
#         }
#         return state_dict
#
#     def save(self, filename):
#         parameters = {
#             'encoder': self.encoder.state_dict(),
#             'classifier': self.classifier.state_dict(),
#             'dense': self.dense.state_dict()
#         }
#         torch.save(parameters, filename)
#
#     def load(self, filename=None, parameters=None):
#         if filename is not None:
#             parameters = torch.load(filename)
#         if parameters is None:
#             raise NotImplementedError("load is a filename or a list of parameters (state_dict)")
#
#         self.encoder.load_state_dict(parameters["encoder"])
#         self.classifier.load_state_dict(parameters["classifier"])
#         self.dense.load_state_dict(parameters["dense"])
