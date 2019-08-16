import torch
import torch.nn as nn

class cATP(nn.Module):
    def __init__(self):
        super(cATP, self).__init__()


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class PT_model(nn.Module):
    def __init__(self):
        super(PT_model, self).__init__()
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = ConvBlock(1, 160, 5, 2)
        self.conv2 = ConvBlock(160, 160, 5, 2)
        self.conv3 = ConvBlock(160, 160, 5, 2)
        self.maxpool = nn.MaxPool2d((1, 4))
        self.dense = nn.Linear(160 * 500, 10)
        self.attention = cATP()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)

        strong = self.dense(x)
        weak = self.attention(x)

        return strong, weak


class PS_model(nn.Module):
    def __init__(self):
        super(PS_model, self).__init__()
        self.noise = gaussian(0.15)
        self.bn = nn.BatchNorm2d(1)
        self.conv1 = nn.ModuleList([ConvBlock(1, 16, 3, 1), ConvBlock(16, 16, 3, 1)])
        self.conv2 = nn.ModuleList([ConvBlock(16, 32, 3, 1), ConvBlock(32, 32, 3, 1)])
        self.conv3 = nn.ModuleList([ConvBlock(32, 64, 3, 1), ConvBlock(64, 64, 3, 1)])
        self.conv4 = nn.ModuleList([ConvBlock(64, 128, 3, 1), ConvBlock(128, 128, 3, 1)])
        self.conv5 = ConvBlock(128, 256, 1, 0)
        self.maxpool = nn.MaxPool2d((1, 4))
        self.dropout = nn.Dropout2d(0.3)
        self.dense = nn.Linear(160 * 500, 10)
        self.attention = cATP()

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = nn.MaxPool2d((5, 4))(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = nn.MaxPool2d((5, 2))(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = nn.MaxPool2d((2, 2))(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = nn.MaxPool2d((2, 2))(x)
        x = self.dropout(x)

        x = self.conv5(x)

        strong = self.dense(x)
        weak = self.attention(x)

        return strong, weak

class GuidedLearningSolver(object):
    def __init__(self,
                 ps_model,
                 pt_model,
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
        self.ps_model = ps_model.cuda() if torch.cuda.is_available() else ps_model
        self.pt_model = pt_model.cuda() if torch.cuda.is_available() else pt_model
        if data_parallel:
            self.ps_model = torch.nn.DataParallel(self.ps_model)
            self.pt_model = torch.nn.DataParallel(self.pt_model)

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