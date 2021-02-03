import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Sequence
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.pytorch_backend.transformer.encoder import Encoder as Transformer
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as Conformer
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder as Conformer2

from espnet2.train.dataset import ESPnetDataset
from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.sequence_iter_factory import SequenceIterFactory

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from egs2.dcase20t4.sed1.pyscripts.sed.trainer import MeanTeacherTrainer
from egs2.dcase20t4.sed1.pyscripts.sed.abs_sed import AbsSED
from egs2.dcase20t4.sed1.pyscripts.sed.espnet_model import ESPnetSEDModel
from egs2.dcase20t4.sed1.pyscripts.sed.frontend.abs_frontend import AbsFrontend
from egs2.dcase20t4.sed1.pyscripts.sed.frontend.default import DefaultFrontend
from egs2.dcase20t4.sed1.pyscripts.sed.dataaug.abs_dataaug import AbsDataAug
from egs2.dcase20t4.sed1.pyscripts.sed.dataaug.dataaug import Compose, GaussianNoise, FrequencyMask, TimeShifting, Mixup
from egs2.dcase20t4.sed1.pyscripts.sed.nets.prenet import CNN
from egs2.dcase20t4.sed1.pyscripts.sed.nets.classifier import LinearClassifier
from egs2.dcase20t4.sed1.pyscripts.sed.preprocessor import CommonPreprocessor
from egs2.dcase20t4.sed1.pyscripts.sed.dataset.sampler import MultiStreamBatchSampler, build_batch_sampler
from espnet2.train.distributed_utils import DistributedOption
from espnet2.tasks.abs_task import IteratorOptions
from espnet2.layers.log_mel import LogMel
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

import yaml
from egs2.dcase20t4.sed1.pyscripts.sed.nets.conformer.conformer_encoder import ConformerEncoder

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(default=DefaultFrontend),
    type_check=AbsFrontend,
    default="default",
)
dataaug_choices = ClassChoices(
    name="dataaug",
    classes=dict(dataaug=AbsDataAug),
    type_check=AbsDataAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    name="normalize",
    classes=dict(
            global_mvn=GlobalMVN,),
    type_check=AbsNormalize,
    default=None,
    optional=True,
)
prenet_choices = ClassChoices(
    name="prenet",
    classes=dict(
        cnn=CNN,
    ),
    type_check=torch.nn.Module,
    default="cnn",
)
encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        transformer=Transformer,
        # conformer=Conformer,
        my_conformer=ConformerEncoder,
        # conformer=Conformer,
    ),
    # type_check=Encoder,
    default="conformer",
)
classifier_choices = ClassChoices(
    name="classifier",
    classes=dict(
        linear=LinearClassifier,
        # linear2=LinearClassifier2,
        # rnn=RNN,
    ),
    type_check=torch.nn.Module,
    default="linear",
)


class SEDTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --dataaug and --dataaug_conf
        dataaug_choices,
        # --feats_extractor and --feats_extractor_conf
        # feats_extractor_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --prenet and --prenet_conf
        prenet_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --classifier and --classifier_conf
        classifier_choices,
        # --learning_type and --learning_type_conf
        # learning_type_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = MeanTeacherTrainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        group.add_argument(
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimension of output feature",
        )
        group.add_argument(
            "--batch_split",
            default=None,
            help="The number of dimension of output feature",
        )
        group.add_argument(
            "--label_list",
        )
        group.add_argument(
            "--train_conf",
            help="The configuration file for training settings"
        )
        group.add_argument(
            "--max_len_seconds",
        )
        group.add_argument(
            "--sample_rate",
        )
        group.add_argument(
            "--hop_length",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(
            float_pad_value=0.0, int_pad_value=0
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool,
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                max_len_seconds=args.max_len_seconds,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                label_list=args.label_list
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("audio", "label")
        else:
            # Inference mode
            retval = ("audio",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ("train_label", "valid_label", "eval_label")
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_sequence_iter_factory(
        cls, args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ) -> AbsIterFactory:
        assert check_argument_types()

        dataset = ESPnetDataset(
            iter_options.data_path_and_name_and_type,
            float_dtype=args.train_dtype,
            preprocess=iter_options.preprocess_fn,
            max_cache_size=iter_options.max_cache_size,
            max_cache_fd=iter_options.max_cache_fd,
        )
        cls.check_task_requirements(
            dataset, args.allow_variable_data_keys, train=iter_options.train
        )
        batch_sampler = MultiStreamBatchSampler(
            batch_size=iter_options.batch_size,
            batch_split=args.batch_split,
            key_file=iter_options.shape_files[0],
            target_keys=["synth", "weak", "unlabel"],
            shuffle=True,
            name=mode
        )

        batches = list(batch_sampler)
        if iter_options.num_batches is not None:
            batches = batches[: iter_options.num_batches]

        bs_list = [len(batch) for batch in batches]

        logging.info(f"[{mode}] dataset:\n{dataset}")
        logging.info(f"[{mode}] Batch sampler: {batch_sampler}")
        logging.info(
            f"[{mode}] mini-batch sizes summary: N-batch={len(bs_list)}, "
            f"mean={np.mean(bs_list):.1f}, min={np.min(bs_list)}, max={np.max(bs_list)}"
        )

        if iter_options.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            for batch in batches:
                if len(batch) < world_size:
                    raise RuntimeError(
                        f"The batch-size must be equal or more than world_size: "
                        f"{len(batch)} < {world_size}"
                    )
            batches = [batch[rank::world_size] for batch in batches]

        return SequenceIterFactory(
            dataset=dataset,
            batches=batches,
            seed=args.seed,
            num_iters_per_epoch=iter_options.num_iters_per_epoch,
            shuffle=iter_options.train,
            num_workers=args.num_workers,
            collate_fn=iter_options.collate_fn,
            pin_memory=args.ngpu > 0,
        )

    @classmethod
    def build_iter_factory(
        cls,
        args: argparse.Namespace,
        distributed_option: DistributedOption,
        mode: str,
        kwargs: dict = None,
    ) -> AbsIterFactory:
        """Build a factory object of mini-batch iterator.

        This object is invoked at every epochs to build the iterator for each epoch
        as following:

        >>> iter_factory = cls.build_iter_factory(...)
        >>> for epoch in range(1, max_epoch):
        ...     for keys, batch in iter_fatory.build_iter(epoch):
        ...         model(**batch)

        The mini-batches for each epochs are fully controlled by this class.
        Note that the random seed used for shuffling is decided as "seed + epoch" and
        the generated mini-batches can be reproduces when resuming.

        Note that the definition of "epoch" doesn't always indicate
        to run out of the whole training corpus.
        "--num_iters_per_epoch" option restricts the number of iterations for each epoch
        and the rest of samples for the originally epoch are left for the next epoch.
        e.g. If The number of mini-batches equals to 4, the following two are same:

        - 1 epoch without "--num_iters_per_epoch"
        - 4 epoch with "--num_iters_per_epoch" == 4

        """
        assert check_argument_types()
        iter_options = cls.build_iter_options(args, distributed_option, mode)

        # Overwrite iter_options if any kwargs is given
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(iter_options, k, v)

        if args.iterator_type == "sequence":
            return cls.build_sequence_iter_factory(
                args=args,
                iter_options=iter_options,
                mode=mode,
            )
        elif args.iterator_type == "chunk":
            return cls.build_chunk_iter_factory(
                args=args,
                iter_options=iter_options,
                mode=mode,
            )
        elif args.iterator_type == "task":
            return cls.build_task_iter_factory(
                args=args,
                iter_options=iter_options,
                mode=mode,
            )
        else:
            raise RuntimeError(f"Not supported: iterator_type={args.iterator_type}")


    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetSEDModel:
        assert check_argument_types()

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. Data augmentation for spectrogram
        if args.dataaug is not None:
            transforms =[]
            if args.dataaug_conf["gaussian_noise"]:
                transforms.append(GaussianNoise())
            if args.dataaug_conf["time_shifting"]:
                transforms.append(TimeShifting())
            if args.dataaug_conf["frequency_masking"]:
                transforms.append(FrequencyMask())
            if args.dataaug_conf["mixup"]:
                transforms.append(Mixup())

            assert len(transforms) != 0
            dataaug = Compose(transforms)

        else:
            dataaug = None

        # 4. Feature embedding
        prenet_class = prenet_choices.get_class(args.prenet)
        prenet = prenet_class(**args.prenet_conf)
                 
        # 3. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(**args.encoder_conf)
        
        # 4. Classifier
        classifier_class = classifier_choices.get_class(args.classifier)
        classifier = classifier_class(**args.classifier_conf)

        # 5. Build model
        model = ESPnetSEDModel(
            frontend=frontend,
            normalize=normalize,
            dataaug=dataaug,
            prenet=prenet,
            encoder=encoder,
            classifier=classifier,
        )

        # FIXME(kamo): Should be done in model?
        # 6. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
