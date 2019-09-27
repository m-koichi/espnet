import torch

from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

from solver.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from dcase_util.data import ProbabilityEncoder
from utils.utils import AverageMeter, weights_init, ManyHotEncoder, SaveBest
from utils import ramps
import pandas as pd
import re

import logging
import math

from solver.baseline_model import CNN, BidirectionalGRU

from logger import Logger
from tensorboardX import SummaryWriter
import mlflow
import ipdb
from focal_loss import FocalLoss

from my_utils import ConfMat
from sklearn.metrics import confusion_matrix
import numpy as np
from CB_loss import CBLoss

logging.basicConfig(level=logging.DEBUG)


CLASSES = {
    'Alarm_bell_ringing'        : 0,
    'Blender'                   : 1,
    'Cat'                       : 2,
    'Dishes'                    : 3,
    'Dog'                       : 4,
    'Electric_shaver_toothbrush': 5,
    'Frying'                    : 6,
    'Running_water'             : 7,
    'Speech'                    : 8,
    'Vacuum_cleaner'            : 9
}

weak_samples_list = [192, 125, 164, 177, 208, 97, 165, 322, 522, 162]
strong_samples_list = [40092, 69093, 28950, 23370, 25153, 51504, 34489, 30453, 122494, 53418]

weak_class_weights = np.sum(weak_samples_list) / (len(CLASSES) * np.array(weak_samples_list))
strong_class_weights = np.sum(strong_samples_list) / (len(CLASSES) * np.array(strong_samples_list))

def cycle_iteration(iterable):
    while True:
        for i in iterable:
            yield i

def log_scalar(writer, name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    writer.add_scalar(name, value, step)
    mlflow.log_metric(name, value)


class Transformer(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 n_class,
                 args,
                 pooling='attention',
                 classifier='linear',
                 input_conv=False,
                 cnn_kwargs=None,
                 cnn_pretrained=False,
                 pos_enc=True,
                 n_frames=496):
        super(Transformer, self).__init__()
        self.args = args
        self.pooling = pooling
        self.input_conv = input_conv
        if input_conv:
            if self.args.input_layer_type == 1:
                self.cnn = CNN(n_in_channel=1, activation="Relu", conv_dropout=args.dropout, **cnn_kwargs)
            if self.args.input_layer_type == 2:
                self.cnn = Conv2dSubsampling(idim=args.mels, odim=args.mels)

        self.encoder = Encoder(input_dim, args, pos_enc=pos_enc)
        if classifier == 'linear':
            self.classifier = torch.nn.Linear(args.adim, n_class)
            self.weak_classifier = torch.nn.Linear(args.adim, n_class)
        elif classifier == 'conv':
            raise NotImplementedError
        elif classifier == 'dense':
            self.classifier = torch.nn.Linear(args.adim, n_class)
            self.weak_classifier = torch.nn.Linear(args.adim * n_frames, n_class)
        elif classifier == 'rnn':
            self.classifier = torch.nn.Sequential(
                                BidirectionalGRU(args.adim, args.adim, dropout=args.dropout, num_layers=2),
                                torch.nn.Linear(args.adim*2, n_class)
                              )
            self.weak_classifier = torch.nn.Sequential(
                                BidirectionalGRU(args.adim, args.adim, dropout=args.dropout, num_layers=2),
                                torch.nn.Linear(args.adim*2, n_class)
                              )
        elif classifier == 'transformer':
            self.pooling = 'transformer'
            self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(args.adim, n_class)
                    )
            self.weak_classifier = torch.nn.Sequential(
                        torch.nn.Linear(args.adim, n_class)
                    )
        elif classifier == 'transformer2':
            self.pooling = 'transformer2'
            self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(args.adim, n_class)
                    )
        else:
            ValueError
        self.dense = torch.nn.Linear(args.adim, n_class)
        self.sigmoid = torch.sigmoid
        self.softmax = torch.nn.Softmax(dim=-1)      
        self.reset_parameters(args)        
        self.pool = torch.nn.MaxPool2d((args.pooling_time_ratio, 1))
        
        print(f'transformer structure; \n\t input_conv:{input_conv} \n\t classifier:{classifier}')
        
    def forward(self, x, mask=None):
        if self.args.input_layer_type == 1:
            x = self.cnn(x)
            x = x.squeeze(-1).permute(0, 2, 1)
        if self.args.input_layer_type == 2:
            x, _ = self.cnn(x, None)

        if self.args.input_layer_type == 3:
            x = x.squeeze(1)
            x = self.pool(x)
            
        elif self.args.input_layer_type == 4:
            x = self.rnn(x)
            x[:, 0, :] = 1
        if self.pooling == 'transformer' or self.pooling == 'transformer2':
            mask = None
            class_frame = torch.ones(x.size(0), 1, x.size(2)).cuda() * 0.2
            x = torch.cat([class_frame, x], dim=1)
#             x[:, 0, :] = 1 # replace first frame to one vector
        
        # Encoder
        x, x_mask = self.encoder(x, mask)            
        
        
        if self.pooling == 'attention':
            strong = torch.sigmoid(self.classifier(x))
            sof = self.dense(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        elif self.pooling == 'mean':
            strong = torch.sigmoid(self.classifier(x))
            weak = strong.mean(1)
        elif self.pooling == 'max':
            strong = torch.sigmoid(self.classifier(x))
            weak = strong.max(1)[0]
        elif self.pooling == 'test':
            strong = torch.sigmoid(self.classifier(x))
            weak = torch.sigmoid(self.weak_classifier(x[:, 0, :]))
            strong[:, 0, :] = 0
        elif self.pooling == 'dense':
            strong = torch.sigmoid(self.classifier(x))
            weak = torch.sigmoid(self.weak_classifier(x.view(-1, x.size(1) * x.size(2))))
        elif self.pooling == 'transformer':
            strong = torch.sigmoid(self.classifier(x[:, 1:, :]))
            first_frame = x[:, 0, :]
            weak = torch.sigmoid(self.weak_classifier(first_frame))
        elif self.pooling == 'transformer2':
            x = torch.sigmoid(self.classifier(x))
            strong = x[:, 1:, :]
            weak = x[:, 0, :]
            
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

    def _save(self, filename):
        parameters = {
            'encoder'   : self.encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'dense'     : self.dense.state_dict()
        }
        if self.input_conv:
            parameters['cnn'] = self.cnn.state_dict()
        torch.save(parameters, filename)

    def _load(self, filename=None, parameters=None):
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
                 data_parallel=False,
                 writer=None,
                 mode='SED'):
        self.model = model.cuda() if torch.cuda.is_available() else model
        self.ema_model = ema_model.cuda() if torch.cuda.is_available() else ema_model
        if data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            self.ema_model = torch.nn.DataParallel(self.ema_model)

        self.criterion = criterion
        if criterion == 'BCE':
            self.strong_criterion = torch.nn.BCELoss().cuda()
            self.weak_criterion = torch.nn.BCELoss().cuda()
        elif criterion == 'FocalLoss':
            self.strong_criterion = FocalLoss(gamma=2).cuda()
            self.weak_criterion = torch.nn.BCELoss().cuda()
        elif criterion == 'CBLoss':
            self.strong_criterion = CBLoss(samples_per_cls=torch.from_numpy(strong_class_weights),
                                       loss_type='FocalLoss').to('cuda')
            self.weak_criterion = CBLoss(samples_per_cls=torch.from_numpy(weak_class_weights),
                                     ).to('cuda')
            
        self.consistency_criterion = consistency_criterion
        self.optimizer = optimizer
        self.accum_grad = accum_grad
        self.grad_clip_threshold = 5

        self.strong_iter = cycle_iteration(strong_loader)
        self.weak_iter = cycle_iteration(weak_loader)
        self.unlabel_iter = cycle_iteration(unlabel_loader)

        self.forward_count = 0
        self.rampup_length = rampup_length
        self.set_optimizer(args)
        self.args = args

        self.max_consistency_cost = consistency_cost
        self.strong_losses = AverageMeter()
        self.weak_losses = AverageMeter()
        self.logger = Logger(exp_name.replace('exp', 'tensorboard'))
        self.writer = writer
        # self.criterion = LabelSmoothingLoss(n_class, -1, 0.1, False, criterion)

    def set_optimizer(self, args):
        if args.opt == 'noam':
            self.optimizer = get_std_opt(self.model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
        elif args.opt == 'adam':
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.transformer_lr, betas=(0.9, 0.98), eps=1e-9)
            
    def train_one_step(self, log_interbal=100, warm_start=True):
        self.model.train()
        strong_sample, strong_target, _, strong_mask = next(self.strong_iter)
        weak_sample, weak_target, _, weak_mask = next(self.weak_iter)

        # try:
        #     unlabel_sample, _, _ = next(self.unlabel_iter)
        # except:
        #     self.unlabel_iter = iter(self.unlabel_loader)
        #     unlabel_sample, _, _ = next(self.unlabel_iter)

        strong_sample = strong_sample.cuda()
        weak_sample = weak_sample.cuda()

        strong_target = strong_target.cuda()
        weak_target = weak_target.cuda()
        
        strong_mask = strong_mask.cuda()
        weak_mask = weak_mask.cuda()

        pred_strong, pred_weak = self.model(strong_sample, strong_mask)
        strong_loss = self.strong_criterion(pred_strong, strong_target)

        pred_strong, pred_weak = self.model(weak_sample, weak_mask)
        weak_loss = self.weak_criterion(pred_weak, weak_target)

        if self.model.pooling == 'test':
            loss = (strong_loss + weak_loss) / self.accum_grad
        elif self.criterion == 'BCE':
            loss = (strong_loss + weak_loss) / self.accum_grad
        elif self.criterion == 'FocalLoss':
            loss = (10 * strong_loss + weak_loss) / self.accum_grad
        elif self.criterion == 'CBLoss':
            loss = (strong_loss + weak_loss) / self.accum_grad
        loss.backward() # Backprop
        loss.detach() # Truncate the graph

        self.strong_losses.update(strong_loss.item())
        self.weak_losses.update(weak_loss.item())
#         self.logger.scalar_summary('train_strong_loss', strong_loss.item(), self.forward_count)
#         self.logger.scalar_summary('train_weak_loss', weak_loss.item(), self.forward_count)

        self.forward_count += 1
        if self.forward_count % log_interbal == 0:
            self.logger.scalar_summary('train_strong_loss', self.strong_losses.avg, self.forward_count)
            self.logger.scalar_summary('train_weak_loss', self.weak_losses.avg, self.forward_count)
            
            logging.info('After {} iteration'.format(self.forward_count))
            logging.info('\t Ave. strong loss: {}'.format(self.strong_losses.avg))
            logging.info('\t Ave. weak loss: {}'.format(self.weak_losses.avg))
            
            
            log_scalar(self.writer, 'train_strong_loss', self.strong_losses.avg, self.forward_count)
            log_scalar(self.writer, 'train_weak_loss', self.weak_losses.avg, self.forward_count)
            
            self.strong_losses.reset()
            self.weak_losses.reset()
        if self.forward_count % self.accum_grad != 0:
            return
        # self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_threshold)
        # logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()
        
    def train_one_step_at(self, log_interbal=100, warm_start=True):
        self.model.train()
#         strong_sample, strong_target, _, strong_mask = next(self.strong_iter)
        weak_sample, weak_target, _, weak_mask = next(self.weak_iter)
#         try:
#             strong_sample, strong_target, _ = next(self.strong_iter)
#         except:
#             self.strong_iter = iter(self.strong_loader)
#             strong_sample, strong_target, _ = next(self.strong_iter)
#         try:
#             weak_sample, weak_target, _ = next(self.weak_iter)
#         except:
#             self.weak_iter = iter(self.weak_loader)
#             weak_sample, weak_target, _ = next(self.weak_iter)
        # try:
        #     unlabel_sample, _, _ = next(self.unlabel_iter)
        # except:
        #     self.unlabel_iter = iter(self.unlabel_loader)
        #     unlabel_sample, _, _ = next(self.unlabel_iter)

#         strong_sample = strong_sample.cuda()
        weak_sample = weak_sample.cuda()
        # unlabel_sample = unlabel_sample.squeeze(1).cuda()
#         strong_target = strong_target.cuda()
        weak_target = weak_target.cuda()
        
#         strong_mask = strong_mask.cuda()
        weak_mask = weak_mask.cuda()

#         pred_strong, pred_weak = self.model(strong_sample, strong_mask, strong_target)
#         strong_loss = self.strong_criterion(pred_strong, strong_target)

        pred_strong, pred_weak = self.model(weak_sample, weak_mask)
        weak_loss = self.weak_criterion(pred_weak, weak_target)

        if self.model.pooling == 'test':
            loss = weak_loss / self.accum_grad
        elif self.criterion == 'BCE':
            loss = weak_loss / self.accum_grad
        elif self.criterion == 'FocalLoss':
            loss = weak_loss / self.accum_grad
        elif self.criterion == 'CBLoss':
            loss = weak_loss / self.accum_grad
        loss.backward() # Backprop
        loss.detach() # Truncate the graph

#         self.strong_losses.update(strong_loss.item())
        self.weak_losses.update(weak_loss.item())
#         self.logger.scalar_summary('train_strong_loss', strong_loss.item(), self.forward_count)
#         self.logger.scalar_summary('train_weak_loss', weak_loss.item(), self.forward_count)

        self.forward_count += 1
        if self.forward_count % log_interbal == 0:
#             self.logger.scalar_summary('train_strong_loss', self.strong_losses.avg, self.forward_count)
            self.logger.scalar_summary('train_weak_loss', self.weak_losses.avg, self.forward_count)
            
            logging.info('After {} iteration'.format(self.forward_count))
#             logging.info('\t Ave. strong loss: {}'.format(self.strong_losses.avg))
            logging.info('\t Ave. weak loss: {}'.format(self.weak_losses.avg))
            
            
#             log_scalar(self.writer, 'train_strong_loss', self.strong_losses.avg, self.forward_count)
            log_scalar(self.writer, 'train_weak_loss', self.weak_losses.avg, self.forward_count)
            
#             self.strong_losses.reset()
            self.weak_losses.reset()
        if self.forward_count % self.accum_grad != 0:
            return
        # self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_threshold)
        # logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            self.optimizer.step()
        self.optimizer.zero_grad()


    def train_one_step_ema(self, log_interbal=100, warm_start=True):
        self.model.train()
        self.ema_model.train()
        
        strong_sample, strong_sample_ema, strong_target, _, strong_mask = next(self.strong_iter)
        weak_sample, weak_sample_ema, weak_target, _, weak_mask = next(self.weak_iter)
        unlabel_sample, unlabel_sample_ema, _, _, unlabel_mask = next(self.unlabel_iter)
        
#         try:
#             strong_sample, strong_target, strong_ids = next(self.strong_iter)
#             strong_sample_ema, strong_target_ema, strong_ids_ema = next(self.strong_iter_ema)
#         except:
#             self.strong_iter = iter(self.strong_loader)
#             strong_sample, strong_target, strong_ids = next(self.strong_iter)
#             self.strong_iter_ema = iter(self.strong_loader)
#             strong_sample_ema, strong_target_ema, strong_ids_ema = next(self.strong_iter_ema)
#         try:
#             weak_sample, weak_target, _ = next(self.weak_iter)
#             weak_sample_ema, weak_target_ema, _ = next(self.weak_iter_ema)
#         except:
#             self.weak_iter = iter(self.weak_loader)
#             weak_sample, weak_target, _ = next(self.weak_iter)
#             self.weak_iter_ema = iter(self.weak_loader)
#             weak_sample_ema, weak_target_ema, _ = next(self.weak_iter_ema)
#         try:
#             unlabel_sample, _, _ = next(self.unlabel_iter)
#             unlabel_sample_ema, _, _ = next(self.unlabel_iter_ema)
#         except:
#             self.unlabel_iter = iter(self.unlabel_loader)
#             unlabel_sample, _, _ = next(self.unlabel_iter)
#             self.unlabel_iter_ema = iter(self.unlabel_loader)
#             unlabel_sample_ema, _, _ = next(self.unlabel_iter_ema)

#         assert strong_ids == strong_ids_ema


        if self.forward_count < self.rampup_length:
            rampup_value = ramps.sigmoid_rampup(self.forward_count, self.rampup_length)
        else:
            rampup_value = 1.0

        strong_sample, strong_sample_ema = strong_sample.cuda(), strong_sample_ema.cuda()
        weak_sample, weak_sample_ema = weak_sample.cuda(), weak_sample_ema.cuda()
        unlabel_sample, unlabel_sample_ema = unlabel_sample.cuda(), unlabel_sample_ema.cuda()
        strong_target = strong_target.cuda()
        weak_target = weak_target.cuda()

        # pred_strong, pred_weak = self.model(strong_sample)
        # strong_loss = self.criterion(pred_strong, strong_target)
        #
        # pred_strong, pred_weak = self.model(weak_sample)
        # weak_loss = self.criterion(pred_weak, weak_target)


#         if warm_start and self.forward_count < self.args.transformer_warmup_steps / 2:
#             pred_strong_ema_w, pred_weak_ema_w = self.ema_model(weak_sample_ema)
#             pred_strong_ema_u, pred_weak_ema_u = self.ema_model(unlabel_sample_ema)
#             pred_strong_ema_w, pred_strong_ema_u = \
#                 pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
#             pred_weak_ema_u, pred_weak_ema_w = \
#                 pred_weak_ema_w.detach(), pred_weak_ema_u.detach()
#         else:
        pred_strong_ema_s, pred_weak_ema_s = self.ema_model(strong_sample_ema)
        pred_strong_ema_w, pred_weak_ema_w = self.ema_model(weak_sample_ema)
        pred_strong_ema_u, pred_weak_ema_u = self.ema_model(unlabel_sample_ema)
        pred_strong_ema_s, pred_strong_ema_w, pred_strong_ema_u = \
            pred_strong_ema_s.detach(), pred_strong_ema_w.detach(), pred_strong_ema_u.detach()
        pred_weak_ema_s, pred_weak_ema_w, pred_weak_ema_u = \
            pred_weak_ema_s.detach(), pred_weak_ema_w.detach(), pred_weak_ema_u.detach()

#         if warm_start and self.forward_count < self.args.transformer_warmup_steps / 2:
#             pred_strong_w, pred_weak_w = self.model(weak_sample)
#             pred_strong_u, pred_weak_u = self.model(unlabel_sample)
#             weak_class_loss = self.criterion(pred_weak_w, weak_target)
#         else:

        pred_strong_s, pred_weak_s = self.model(strong_sample)
        pred_strong_w, pred_weak_w = self.model(weak_sample)
        pred_strong_u, pred_weak_u = self.model(unlabel_sample)
        strong_class_loss = self.strong_criterion(pred_strong_s, strong_target)
        weak_class_loss = self.weak_criterion(pred_weak_w, weak_target)

        # compute consistency loss
        consistency_cost = self.max_consistency_cost * rampup_value
#         if warm_start and self.forward_count < self.args.transformer_warmup_steps / 2:
#             consistency_loss_weak = consistency_cost * self.consistency_criterion(pred_weak_w, pred_weak_ema_w) \
#                                     + consistency_cost * self.consistency_criterion(pred_weak_u, pred_weak_ema_u)

#             loss = (
#                            weak_class_loss + consistency_loss_weak) / self.accum_grad
#             loss.backward()  # Backprop
#             loss.detach()  # Truncate the graph

#             # self.logger.scalar_summary('train_strong_loss', strong_class_loss.item(), self.forward_count)
#             self.logger.scalar_summary('train_weak_loss', weak_class_loss.item(), self.forward_count)

#             self.forward_count += 1
#             if self.forward_count % log_interbal == 0:
#                 logging.info('After {} iteration'.format(self.forward_count))
#                 # logging.info('\tstrong loss: {}'.format(strong_class_loss.item()))
#                 logging.info('\tweak loss: {}'.format(weak_class_loss.item()))
#         else:

        strong_class_ema_loss = self.consistency_criterion(pred_strong_s, pred_strong_ema_s) \
                                + self.consistency_criterion(pred_strong_w, pred_strong_ema_w) \
                                + self.consistency_criterion(pred_strong_u, pred_strong_ema_u)
        weak_class_ema_loss = self.consistency_criterion(pred_weak_s, pred_weak_ema_s) \
                                + self.consistency_criterion(pred_weak_w, pred_weak_ema_w) \
                                + self.consistency_criterion(pred_weak_u, pred_weak_ema_u)
        consistency_loss_strong = consistency_cost * strong_class_ema_loss
        consistency_loss_weak = consistency_cost * weak_class_ema_loss

        loss = (strong_class_loss + weak_class_loss + consistency_loss_strong + consistency_loss_weak) / self.accum_grad
        loss.backward() # Backprop
        loss.detach() # Truncate the graph

        self.logger.scalar_summary('train_strong_loss', strong_class_loss.item(), self.forward_count)
        self.logger.scalar_summary('train_weak_loss', weak_class_loss.item(), self.forward_count)

        self.forward_count += 1
        if self.forward_count % log_interbal == 0:
            logging.info('After {} iteration'.format(self.forward_count))
            logging.info('\t strong loss: {}'.format(strong_class_loss.item()))
            logging.info('\t weak loss: {}'.format(weak_class_loss.item()))
            logging.info('\t consistency loss strong: {}'.format(consistency_loss_strong.item()))
            logging.info('\t consistency loss weak: {}'.format(consistency_loss_weak.item()))
        if self.forward_count % self.accum_grad != 0:
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
                        post_processing=None, save_predictions=None, mode='validation',
                        logger=None, pooling_time_ratio=1., sample_rate=22050, hop_length=365):
        
        prediction_df = pd.DataFrame()

        avg_strong_loss = 0
        avg_weak_loss = 0

        # Flame level 
        frame_measure = [ConfMat() for i in range(len(CLASSES))]
        tag_measure = ConfMat()
        self.model.eval()
        self.ema_model.eval()
        with torch.no_grad():
            for batch_idx, (batch_input, batch_target, data_ids, _) in enumerate(data_loader):
                batch_target_np = batch_target.numpy()
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()

                if self.args.ssl:
                    pred_strong, pred_weak = self.model(batch_input)
                else:
                    pred_strong, pred_weak = self.model(batch_input)
#                 if self.forward_count > 5000:
#                     ipdb.set_trace()

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
                    pred_weak = ProbabilityEncoder().binarization(pred_weak, binarization_type=binarization_type,
                                                                    threshold=threshold)

                if post_processing is not None:
                    for i in range(pred_strong.shape[0]):
                        for post_process_fn in post_processing:
                            pred_strong[i] = post_process_fn(pred_strong[i])
                            
                for i in range(len(pred_strong)):
                    tn, fp, fn, tp = confusion_matrix(batch_target_np[i].max(axis=0), pred_weak[i], labels=[0,1]).ravel()
                    tag_measure.add_cf(tn, fp, fn, tp)
                    for j in range(len(CLASSES)):
#                         import ipdb
#                         ipdb.set_trace()
                        tn, fp, fn, tp = confusion_matrix(batch_target_np[i][:, j], pred_strong[i][:, j], labels=[0,1]).ravel()
                        frame_measure[j].add_cf(tn, fp, fn, tp)

                for pred, data_id in zip(pred_strong, data_ids):
                    pred = decoder(pred)
                    pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
                    pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
                    prediction_df = prediction_df.append(pred)
                    
        # In seconds
        prediction_df.onset = prediction_df.onset * pooling_time_ratio / (sample_rate / hop_length)
        prediction_df.offset = prediction_df.offset * pooling_time_ratio / (sample_rate / hop_length)
        
        # Compute frame level macro f1 score
        ave_precision = 0
        ave_recall = 0
        macro_f1 = 0
        for i in range(len(CLASSES)):
            ave_precision_, ave_recall_, macro_f1_ = frame_measure[i].calc_f1()
            ave_precision += ave_precision_
            ave_recall += ave_recall_
            macro_f1 += macro_f1_
        ave_precision /= len(CLASSES)
        ave_recall /= len(CLASSES)
        macro_f1 /= len(CLASSES)

        if save_predictions is not None:
            logging.info("Saving predictions at: {}".format(save_predictions))
            prediction_df.to_csv(save_predictions, index=False, sep="\t")

        weak_f1 = tag_measure.calc_f1()[2]    
        if mode == 'validation':
            logging.info(f'\tAve. valid strong class loss: {avg_strong_loss}')
            logging.info(f'\tAve. valid weak class loss: {avg_weak_loss}')
            logging.info(f'\tAve. frame level precision: {ave_precision}')
            logging.info(f'\tAve. frame level recall: {ave_recall}')
            logging.info(f'\tAve. frame level macro f1: {macro_f1}')
            logging.info(f'\tAve. weak f1: {weak_f1}')
            self.logger.scalar_summary('valid_strong_loss', avg_strong_loss, self.forward_count)
            self.logger.scalar_summary('valid_weak_loss', avg_weak_loss, self.forward_count)
            self.logger.scalar_summary('frame_level_precision', ave_precision, self.forward_count)
            self.logger.scalar_summary('frame_level_recall', ave_recall, self.forward_count)
            self.logger.scalar_summary('frame_level_macro_f1', macro_f1, self.forward_count)
            self.logger.scalar_summary('weak_f1', weak_f1, self.forward_count)
            log_scalar(self.writer, 'valid_strong_loss', avg_strong_loss, self.forward_count)
            log_scalar(self.writer, 'valid_weak_loss', avg_weak_loss, self.forward_count)
            log_scalar(self.writer, 'frame_level_precision', ave_precision, self.forward_count)
            log_scalar(self.writer, 'frame_level_recall', ave_recall, self.forward_count)
            log_scalar(self.writer, 'frame_level_macro_f1', macro_f1, self.forward_count)
            log_scalar(self.writer, 'weak_f1', weak_f1, self.forward_count)

        return prediction_df, ave_precision, ave_recall, macro_f1, weak_f1

    def save(self, filename, ema_filename):
        torch.save(self.model.state_dict(), filename)
        torch.save(self.ema_model.state_dict(), ema_filename)
        
    def load(self, filename, ema_filename):
        self.model.load_state_dict(torch.load(filename))
        self.ema_model.load_state_dict(torch.load(ema_filename))

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
