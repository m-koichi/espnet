
import argparse
from contextlib import contextmanager
import dataclasses
from dataclasses import is_dataclass
from distutils.version import LooseVersion
from os import stat_result

from torch.distributed.distributed_c10d import group
import logging
from pathlib import Path
import time
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import humanfriendly
import numpy as np
import torch
import torch.nn
from torch.nn.modules import pooling
import torch.optim
from typeguard import check_argument_types

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import AbsBatchStepScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochStepScheduler
from espnet2.schedulers.abs_scheduler import AbsScheduler
from espnet2.schedulers.abs_scheduler import AbsValEpochStepScheduler
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter
from espnet2.train.reporter import SubReporter
from espnet2.utils.build_dataclass import build_dataclass

from espnet2.train.trainer import Trainer
# from espnet2.train.trainer import TrainerOptions


if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter
if torch.distributed.is_available():
    if LooseVersion(torch.__version__) > LooseVersion("1.0.1"):
        from torch.distributed import ReduceOp
    else:
        from torch.distributed import reduce_op as ReduceOp
else:
    ReduceOp = None

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
    from torch.cuda.amp import GradScaler
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

    GradScaler = None

import pandas as pd
import h5py
from dcase_util.data import ProbabilityEncoder
import re
import copy
from egs2.dcase20t4.sed1.pyscripts.sed.utils.utils import ManyHotEncoder
from egs2.dcase20t4.sed1.pyscripts.sed.utils.utils import get_durations_df
from egs2.dcase20t4.sed1.pyscripts.sed.utils.evaluation_measures import compute_metrics
from egs2.dcase20t4.sed1.pyscripts.sed.utils.evaluation_measures import ConfusionMatrix
from sklearn.metrics import confusion_matrix


from espnet2.torch_utils.initialize import initialize
from egs2.dcase20t4.sed1.pyscripts.sed.postprocess import PostProcess
import math


@dataclasses.dataclass
class MeanTeacherTrainerOptions:
    ngpu: int
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_tensorboard: bool
    use_wandb: bool

    # mean teacher training related
    batch_split: Optional[List[int]]

    
    rampup_length=15000
    max_consistency_cost=2.0
    max_len_seconds=10.

    # validation related
    # class_list=list(CLASSES.keys())
    # threshold: float=0.5
    # many_hot_encoder=ManyHotEncoder(labels=class_list, n_frames=62)
    # decoder=many_hot_encoder.decode_strong
    # binarization_type: str="global_threshold"
    # post_processing=None
    valid_csv: str="./dcase20_task4/dataset/metadata/validation/validation.tsv"
    valid_audio_dir: str="./dcase20_task4/dataset/audio/validation"

    pooling_time_ratio: int=8
    sample_rate: int=16000
    hop_size: int=323

    def set_other_options(self):
        self.validation_df = pd.read_csv(self.valid_csv, header=0, sep="\t")
        self.durations_validation = get_durations_df(self.valid_csv, self.valid_audio_dir)
        self.classes = self.validation_df.event_label.dropna().sort_values().unique()
        max_frames = math.ceil(self.max_len_seconds * self.sample_rate / self.hop_size)
        self.many_hot_encoder = ManyHotEncoder(labels=self.classes, n_frames=max_frames)
        self.decoder = self.many_hot_encoder.decode_strong
        self.consistency_criterion=torch.nn.MSELoss().cuda()

class MeanTeacherTrainer(Trainer):

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group =  parser.add_argument_group(description="Trainer related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

        # hyper -parameters for mean teacher 
        group.add_argument(
            "--valid_csv",
            type=str,
            default="./dcase20_task4/dataset/metadata/validation/validation.tsv",
            help="path to metadata file of the validation set"
        )
        group.add_argument(
            "--valid_audio_dir",
            type=str,
            default="./dcase20_task4/dataset/audio/validation",
            help="path to audio directory of the validation set"
        )

        group.add_argument(
            "--pooling_time_ratio",
            type=int,
            default=8,
            help="pooling time ratio between input frames and output frames"
        )
        group.add_argument(
            "--hop_size",
            type=int,
            default=323,
        )


    @classmethod
    def build_options(cls, args: argparse.Namespace) -> MeanTeacherTrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        options = build_dataclass(MeanTeacherTrainerOptions, args)
        options.set_other_options()
        return options


    @classmethod
    def run(
        cls,
        model: AbsESPnetModel,
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        train_iter_factory: AbsIterFactory,
        valid_iter_factory: AbsIterFactory,
        plot_attention_iter_factory: Optional[AbsIterFactory],
        reporter: Reporter,
        scaler: Optional[GradScaler],
        output_dir: Path,
        max_epoch: int,
        seed: int,
        patience: Optional[int],
        keep_nbest_models: int,
        early_stopping_criterion: Sequence[str],
        best_model_criterion: Sequence[Sequence[str]],
        val_scheduler_criterion: Sequence[str],
        trainer_options,
        distributed_option: DistributedOption,
        find_unused_parameters: bool = False,
    ) -> None:
        """Perform training. This method performs the main process of training."""
        assert check_argument_types()
        # NOTE(kamo): Don't check the type more strictly as far trainer_options
        assert is_dataclass(trainer_options), type(trainer_options)

        model_ema = copy.deepcopy(model)
        for param in model_ema.parameters():
            param.detach_()

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )
        
        if distributed_option.distributed:
            dp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=(
                    # Perform multi-Process with multi-GPUs
                    [torch.cuda.current_device()]
                    if distributed_option.ngpu == 1
                    # Perform single-Process with multi-GPUs
                    else None
                ),
                output_device=(
                    torch.cuda.current_device()
                    if distributed_option.ngpu == 1
                    else None
                ),
                find_unused_parameters=find_unused_parameters,
            )
        elif distributed_option.ngpu > 1:
            dp_model = torch.nn.parallel.DataParallel(
                model,
                device_ids=list(range(distributed_option.ngpu)),
                find_unused_parameters=find_unused_parameters,
            )
        else:
            # NOTE(kamo): DataParallel also should work with ngpu=1,
            # but for debuggability it's better to keep this block.
            dp_model = model

        if  trainer_options.use_tensorboard and (
            not distributed_option.distributed or distributed_option.dist_rank == 0
        ):
            summary_writer = SummaryWriter(str(output_dir / "tensorboard"))
        else:
            summary_writer = None

        start_time = time.perf_counter()

        # 9. Search optimal post-processing parameters
        # cls.optimize_post_processing(dp_model, valid_iter_factory.build_iter(0), output_dir, trainer_options)
        for iepoch in range(start_epoch, max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{}epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{max_epoch}epoch started")
            set_all_random_seed(seed + iepoch)

            reporter.set_epoch(iepoch)
            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train_one_epoch_ema(
                    model=dp_model,
                    model_ema=model_ema,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    scaler=scaler,
                    summary_writer=summary_writer,
                    options=trainer_options,
                )

            with reporter.observe("valid") as sub_reporter:
                cls.validate_one_epoch(
                    model=dp_model,
                    iterator=valid_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    options=trainer_options,
                )

            # with reporter.observe("valid_close") as sub_reporter:
            #     cls.validate_one_epoch(
            #         model=dp_model,
            #         iterator=train_iter_factory.build_iter(iepoch),
            #         reporter=sub_reporter,
            #         options=trainer_options,
            #     )

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # att_plot doesn't support distributed
                if plot_attention_iter_factory is not None:
                    with reporter.observe("att_plot") as sub_reporter:
                        cls.plot_attention(
                            model=model,
                            output_dir=output_dir / "att_ws",
                            summary_writer=summary_writer,
                            iterator=plot_attention_iter_factory.build_iter(iepoch),
                            reporter=sub_reporter,
                            options=trainer_options,
                        )

            # 2. LR Scheduler step
            for scheduler in schedulers:
                if isinstance(scheduler, AbsValEpochStepScheduler):
                    scheduler.step(reporter.get_value(*val_scheduler_criterion))
                elif isinstance(scheduler, AbsEpochStepScheduler):
                    scheduler.step()

            if not distributed_option.distributed or distributed_option.dist_rank == 0:
                # 3. Report the results
                logging.info(reporter.log_message())
                reporter.matplotlib_plot(output_dir / "images")
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer)
                if trainer_options.use_wandb:
                    reporter.wandb_log()

                # 4. Save/Update the checkpoint
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_ema": model_ema.state_dict(),
                        "reporter": reporter.state_dict(),
                        "optimizers": [o.state_dict() for o in optimizers],
                        "schedulers": [
                            s.state_dict() if s is not None else None
                            for s in schedulers
                        ],
                        "scaler": scaler.state_dict() if scaler is not None else None,
                    },
                    output_dir / "checkpoint.pth",
                )

                # 5. Save the model and update the link to the best model
                torch.save(model.state_dict(), output_dir / f"{iepoch}epoch.pth")

                # Creates a sym link latest.pth -> {iepoch}epoch.pth
                p = output_dir / "latest.pth"
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(f"{iepoch}epoch.pth")

                _improved = []
                for _phase, k, _mode in best_model_criterion:
                    # e.g. _phase, k, _mode = "train", "loss", "min"
                    if reporter.has(_phase, k):
                        best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                        # Creates sym links if it's the best result
                        if best_epoch == iepoch:
                            p = output_dir / f"{_phase}.{k}.best.pth"
                            if p.is_symlink() or p.exists():
                                p.unlink()
                            p.symlink_to(f"{iepoch}epoch.pth")
                            _improved.append(f"{_phase}.{k}")
                if len(_improved) == 0:
                    logging.info("There are no improvements in this epoch")
                else:
                    logging.info(
                        "The best model has been updated: " + ", ".join(_improved)
                    )

                # 6. Remove the model files excluding n-best epoch and latest epoch
                _removed = []
                # Get the union set of the n-best among multiple criterion
                nbests = set().union(
                    *[
                        set(reporter.sort_epochs(ph, k, m)[:keep_nbest_models])
                        for ph, k, m in best_model_criterion
                        if reporter.has(ph, k)
                    ]
                )
                for e in range(1, iepoch):
                    p = output_dir / f"{e}epoch.pth"
                    if p.exists() and e not in nbests:
                        p.unlink()
                        _removed.append(str(p))
                if len(_removed) != 0:
                    logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if patience is not None:
                if reporter.check_early_stopping(patience, *early_stopping_criterion):
                    break

        else:
            logging.info(f"The training was finished at {max_epoch} epochs ")
            # 9. Search optimal post-processing parameters
            cls.optimize_post_processing(dp_model, valid_iter_factory.build_iter(0), output_dir, trainer_options)

    @classmethod
    def train_one_epoch_ema(
        cls,
        model: torch.nn.Module,
        model_ema: torch.nn.Module,
        iterator: Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizers: Sequence[torch.optim.Optimizer],
        schedulers: Sequence[Optional[AbsScheduler]],
        scaler: Optional[GradScaler],
        reporter: SubReporter,
        summary_writer: Optional[SummaryWriter],
        options: MeanTeacherTrainerOptions,
    ) -> bool:
        assert check_argument_types()

        # Note(kamo): assumes one optimizer
        assert cls.num_optimizers == 1, cls.num_optimizers
        assert len(optimizers) == 1, len(optimizers)
        optimizer = optimizers[0]
        scheduler = schedulers[0]

        consistency_criterion = options.consistency_criterion
        rampup_length = options.rampup_length

        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        max_consistency_cost = options.max_consistency_cost
        max_len_seconds = options.max_len_seconds
        use_wandb = options.use_wandb
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)

        bs = {
            "strong": slice(0, sum(options.batch_split[:1])),
            "weak": slice(sum(options.batch_split[:1]), sum(options.batch_split[:2])),
            "unlabeled": slice(sum(options.batch_split[:2]), sum(options.batch_split[:3])),
            }

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100

        model.train()
        model_ema.train()
        all_steps_are_invalid = True
        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")

        start_time = time.perf_counter()
        for iiter, (data_ids, batch) in enumerate(
            reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            assert isinstance(batch, dict), type(batch)
            
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                continue

            with autocast(scaler is not None):
                with reporter.measure_time("forward_time"):

                    # feature extraction
                    audio, audio_lengths, label, label_lengths \
                        = batch['audio'], batch['audio_lengths'], batch['label'], batch['label_lengths']

                    # 1. Extract feats
                    feats, feats_lengths = model._extract_feats(audio, audio_lengths)

                    # 2. Data augmentation for spectrogram
                    if model.dataaug is not None and model.training:
                        # Apply data augmentaion along each label type
                        # strongly labeled data
                        feats[bs["strong"]], feats_lengths[bs["strong"]], \
                            label[bs["strong"]], label_lengths[bs["strong"]] \
                            = model.dataaug(
                            feats[bs["strong"]], feats_lengths[bs["strong"]], \
                             label[bs["strong"]], label_lengths[bs["strong"]])
                        # weakly labeled data
                        feats[bs["weak"]], feats_lengths[bs["weak"]], label[bs["weak"]], label_lengths[bs["weak"]] \
                            = model.dataaug(
                            feats[bs["weak"]], feats_lengths[bs["weak"]], label[bs["weak"]], label_lengths[bs["weak"]])

                        # unlabeled data
                        feats[bs["unlabeled"]], feats_lengths[bs["unlabeled"]], \
                            label[bs["unlabeled"]], label_lengths[bs["unlabeled"]] \
                            = model.dataaug(
                            feats[bs["unlabeled"]], feats_lengths[bs["unlabeled"]], \
                            label[bs["unlabeled"]], label_lengths[bs["unlabeled"]])

                    # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                    if model.normalize is not None:
                        feats, feats_lengths = model.normalize(feats, feats_lengths)

                    batch['audio'], batch['audio_lengths'] = feats, feats_lengths

                    # 4. Forward step
                    predicts, loss, stats, weight = model(batch_split=bs, **batch)
                    with torch.no_grad():
                        predicts_, loss_, stats_, weight_ = model_ema(batch_split=bs, **batch)
                    rampup_value = cls.exp_rampup(reporter.get_total_count(), rampup_length)
                    consistency_cost = max_consistency_cost * rampup_value
                    consistency_loss_strong = consistency_cost * consistency_criterion(
                            torch.sigmoid(predicts["strong"]), torch.sigmoid(predicts_["strong"]))
                    consistency_loss_weak = consistency_cost * consistency_criterion(
                            torch.sigmoid(predicts["weak"]), torch.sigmoid(predicts_["weak"]))
                    
                    consistency_loss = consistency_loss_strong + consistency_loss_weak
                    loss = loss + consistency_loss
                    stats["consistency_loss_strong"] = consistency_loss_strong.detach()
                    stats["consistency_loss_weak"] = consistency_loss_weak.detach()
                    stats["consistency_loss"] = consistency_loss.detach()
                stats = {k: v for k, v in stats.items() if v is not None}
                if ngpu > 1 or distributed:
                    # Apply weighted averaging for loss and stats
                    loss = (loss * weight.type(loss.dtype)).sum()

                    # if distributed, this method can also apply all_reduce()
                    stats, weight = recursive_average(stats, weight, distributed)

                    # Now weight is summation over all workers
                    loss /= weight
                if distributed:
                    # NOTE(kamo): Multiply world_size because DistributedDataParallel
                    # automatically normalizes the gradient by world_size.
                    loss *= torch.distributed.get_world_size()

                loss /= accum_grad

            reporter.register(stats, weight)

            with reporter.measure_time("backward_time"):
                if scaler is not None:
                    # Scales loss.  Calls backward() on scaled loss
                    # to create scaled gradients.
                    # Backward passes under autocast are not recommended.
                    # Backward ops run in the same dtype autocast chose
                    # for corresponding forward ops.
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if iiter % accum_grad == 0:
                if scaler is not None:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                # gradient noise injection
                if grad_noise:
                    add_gradient_noise(
                        model,
                        reporter.get_total_count(),
                        duration=100,
                        eta=1.0,
                        scale_factor=0.55,
                    )

                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                # PyTorch<=1.4, clip_grad_norm_ returns float value
                if not isinstance(grad_norm, torch.Tensor):
                    grad_norm = torch.tensor(grad_norm)

                if not torch.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. Skipping updating the model."
                    )
                else:
                    all_steps_are_invalid = False
                    with reporter.measure_time("optim_step_time"):
                        if scaler is not None:
                            # scaler.step() first unscales the gradients of
                            # the optimizer's assigned params.
                            scaler.step(optimizer)
                            # Updates the scale for next iteration.
                            scaler.update()
                        else:
                            optimizer.step()
                    cls.update_ema_variables(model, model_ema, steps=reporter.get_total_count(), alpha=0.999)
                    if isinstance(scheduler, AbsBatchStepScheduler):
                        scheduler.step()
                    # FIXME: override absscheduler
                    if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                        scheduler.step()
                optimizer.zero_grad()

                # Register lr and train/load time[sec/step],
                # where step refers to accum_grad * mini-batch
                reporter.register(
                    dict(
                        {
                            f"lr_{i}": pg["lr"]
                            for i, pg in enumerate(optimizer.param_groups)
                            if "lr" in pg
                        },
                        train_time=time.perf_counter() - start_time,
                    ),
                )
                start_time = time.perf_counter()

            # NOTE(kamo): Call log_message() after next()
            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

        return all_steps_are_invalid
    
    @classmethod
    @torch.no_grad()
    def validate_one_epoch(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: MeanTeacherTrainerOptions,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        prediction_score_df = pd.DataFrame()
        prediction_df = pd.DataFrame()

        threshold=0.5
        decoder=options.decoder
        binarization_type="global_threshold"
        post_processing=None

        ptr=options.pooling_time_ratio

        # Frame level measure
        frame_measure = [ConfusionMatrix() for i in range(len(options.classes))]
        tag_measure = ConfusionMatrix()

        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (data_ids, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            audio, audio_lengths, label, label_lengths \
                = batch['audio'], batch['audio_lengths'], batch['label'], batch['label_lengths']

            # 1. Extract feats
            feats, feats_lengths = model._extract_feats(audio, audio_lengths)

            # 2. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if model.normalize is not None:
                feats, feats_lengths = model.normalize(feats, feats_lengths)

            batch['audio'], batch['audio_lengths'] = feats, feats_lengths

            predicts, loss, stats, weight = model(**batch)
            predicts["strong"] = torch.sigmoid(predicts["strong"]).cpu().data.numpy()
            predicts["weak"] = torch.sigmoid(predicts["weak"]).cpu().data.numpy()

            if binarization_type == 'class_threshold':
                for i in range(predicts["strong"].shape[0]):
                    predicts["strong"][i] = ProbabilityEncoder().binarization(predicts["strong"][i],
                                                                        binarization_type=binarization_type,
                                                                        threshold=threshold, time_axis=0)
            else:
                predicts["strong"] = ProbabilityEncoder().binarization(predicts["strong"],
                                                                binarization_type=binarization_type,
                                                                threshold=threshold)
                predicts["weak"] = ProbabilityEncoder().binarization(predicts["weak"],
                                                                binarization_type=binarization_type,
                                                                threshold=threshold)

            # For debug, frame level measure
            for i in range(len(predicts["strong"])):
                target_np = batch["label"].cpu().numpy()
                tn, fp, fn, tp = confusion_matrix(target_np[i].max(axis=0), predicts["weak"][i], labels=[0,1]).ravel()
                tag_measure.add_cf(tn, fp, fn, tp)
                for j in range(len(options.classes)):
                    tn, fp, fn, tp = confusion_matrix(target_np[i][ptr//2::ptr, j], predicts["strong"][i][:, j], labels=[0,1]).ravel()
                    frame_measure[j].add_cf(tn, fp, fn, tp)

            if post_processing is not None:
                for i in range(predicts["strong"].shape[0]):
                    for post_process_fn in post_processing:
                        predicts["strong"][i] = post_process_fn(predicts["strong"][i])

            for pred, data_id in zip(predicts["strong"], data_ids):
                    pred = decoder(pred)
                    pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])

                    # Put them in seconds
                    pred.loc[:, ["onset", "offset"]] *= ptr / (options.sample_rate / options.hop_size)
                    pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(0, options.max_len_seconds)

                    pred["filename"] = re.sub('^.*?-', '', data_id + '.wav')
                    prediction_df = prediction_df.append(pred)

            if ngpu > 1 or distributed:
                # Apply weighted averaging for stats.
                # if distributed, this method can also apply all_reduce()
                stats, weight = recursive_average(stats, weight, distributed)

            reporter.register(stats, weight)
            reporter.next()

        else:
            if distributed:
                iterator_stop.fill_(1)
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)

            # Compute evaluation metrics
            events_metric, segments_metric, psds_m_f1 = compute_metrics(
                prediction_df, options.validation_df, options.durations_validation)
            macro_f1_event = events_metric.results_class_wise_average_metrics()['f_measure']['f_measure']
            macro_f1_segment  = segments_metric.results_class_wise_average_metrics()['f_measure']['f_measure']

            # Compute frame level macro f1 score
            ave_precision = 0
            ave_recall = 0
            macro_f1 = 0
            for i in range(len(options.classes)):
                ave_precision_, ave_recall_, macro_f1_ = frame_measure[i].calc_f1()
                ave_precision += ave_precision_
                ave_recall += ave_recall_
                macro_f1 += macro_f1_
            ave_precision /= len(options.classes)
            ave_recall /= len(options.classes)
            macro_f1 /= len(options.classes)
            weak_f1 = tag_measure.calc_f1()[2]

            metrics = {
                "event_m_f1": macro_f1_event,
                "segment_m_f1": macro_f1_segment,
                "psds_m_f1": psds_m_f1,
                'frame_level_precision': ave_precision,
                'frame_level_recall': ave_recall,
                'frame_level_macro_f1': macro_f1,
                'weak_f1': weak_f1,
            }

            reporter.register(metrics)
            reporter.next()
    
    @classmethod
    @torch.no_grad()
    def optimize_post_processing(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        output_dir: Path,
        options: MeanTeacherTrainerOptions
    ) -> None:
        assert check_argument_types()
        post_process = PostProcess(model, iterator, output_dir, options)
        post_process.compute_psds()
        pp_params = post_process.tune_all()
        np.savez(output_dir/"post_process_params.npz", **pp_params)

    @classmethod
    @torch.no_grad()
    def validate_close(
        cls,
        model: torch.nn.Module,
        iterator: Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        options: MeanTeacherTrainerOptions,
    ) -> None:
        assert check_argument_types()
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run
        distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        
        model.eval()

        # [For distributed] Because iteration counts are not always equals between
        # processes, send stop-flag to the other processes if iterator is finished
        iterator_stop = torch.tensor(0).to("cuda" if ngpu > 0 else "cpu")
        for (data_ids, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            if distributed:
                torch.distributed.all_reduce(iterator_stop, ReduceOp.SUM)
                if iterator_stop > 0:
                    break

            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                continue

            _, _, stats, weight = model(**batch)

            reporter.register(stats, weight)
            reporter.next()

    @classmethod
    def update_ema_variables(
        cls,
        model: torch.nn.Module,
        model_ema: torch.nn.Module,
        steps: int,
        alpha: float) -> None:
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (steps + 1), alpha)
        for ema_param, param in zip(model_ema.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    @classmethod
    def exp_rampup(
        cls,
        current: int,
        rampup_length: int
        ) -> float:
        """Exponential rampup inspired by https://arxiv.org/abs/1610.02242
            Args:
                current: float, current step of the rampup
                rampup_length: float: length of the rampup
        """
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
