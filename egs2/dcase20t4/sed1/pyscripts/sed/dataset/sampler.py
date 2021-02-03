from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Iterator
from typing import Optional

from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler
from espnet2.samplers.length_batch_sampler import LengthBatchSampler
from espnet2.samplers.num_elements_batch_sampler import NumElementsBatchSampler
from espnet2.samplers.sorted_batch_sampler import SortedBatchSampler
from espnet2.samplers.unsorted_batch_sampler import UnsortedBatchSampler

import itertools
import math


BATCH_TYPES = dict(
    unsorted="UnsortedBatchSampler has nothing in paticular feature and "
    "just creates mini-batches which has constant batch_size. "
    "This sampler doesn't require any length "
    "information for each feature. "
    "'key_file' is just a text file which describes each sample name."
    "\n\n"
    "    utterance_id_a\n"
    "    utterance_id_b\n"
    "    utterance_id_c\n"
    "\n"
    "The fist column is referred, so 'shape file' can be used, too.\n\n"
    "    utterance_id_a 100,80\n"
    "    utterance_id_b 400,80\n"
    "    utterance_id_c 512,80\n",
    sorted="SortedBatchSampler sorts samples by the length of the first input "
    " in order to make each sample in a mini-batch has close length. "
    "This sampler requires a text file which describes the length for each sample "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
    folded="FoldedBatchSampler supports variable batch_size. "
    "The batch_size is decided by\n"
    "    batch_size = base_batch_size // (L // fold_length)\n"
    "L is referred to the largest length of samples in the mini-batch. "
    "This samples requires length information as same as SortedBatchSampler\n",
    length="LengthBatchSampler supports variable batch_size. "
    "This sampler makes mini-batches which have same number of 'bins' as possible "
    "counting by the total lengths of each feature in the mini-batch. "
    "This sampler requires a text file which describes the length for each sample. "
    "\n\n"
    "    utterance_id_a 1000\n"
    "    utterance_id_b 1453\n"
    "    utterance_id_c 1241\n"
    "\n"
    "The first element of feature dimensions is referred, "
    "so 'shape_file' can be also used.\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
    numel="NumElementsBatchSampler supports variable batch_size. "
    "Just like LengthBatchSampler, this sampler makes mini-batches"
    " which have same number of 'bins' as possible "
    "counting by the total number of elements of each feature "
    "instead of the length. "
    "Thus this sampler requires the full information of the dimension of the features. "
    "\n\n"
    "    utterance_id_a 1000,80\n"
    "    utterance_id_b 1453,80\n"
    "    utterance_id_c 1241,80\n",
)

from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.fileio.read_text import read_2column_text
from espnet2.samplers.abs_sampler import AbsSampler
import logging
import numpy as np


class MultiStreamBatchSampler(AbsSampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self,
        batch_size: int,
        target_keys: List[str],
        shuffle: bool = True,
        key_file=None,
        name=None,
        drop_last=False,
        batch_split: Optional[List[int]]=None,
    ):
        assert check_argument_types()
        if batch_split is not None:
            assert len(batch_split) == len(target_keys)
            assert sum(batch_split) == batch_size

        self.name = name
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key_file = key_file
        if name == "train":
            self.batch_sizes = batch_split
            self.target_keys = ["synth", "weak", "unlabel"]
        elif name == "valid":
            self.batch_sizes = [batch_size]
            self.target_keys = ["valid"]
        elif name == "eval":
            self.batch_sizes = [batch_sizes]
            self.target_keys = ["eval"]
        elif name == "plot_att":
            return None
        else:
            raise ValueError

        utt2any = read_2column_text(key_file)
        if len(utt2any) == 0:
            logging.warning(f"{key_file} is empty")
        # In this case the, the first column in only used
        keys = list(utt2any)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {self.key_file}")
        # split target keys
        self.target = {}
        for key in self.target_keys:
            self.target[key] = [k for k in keys if key in k]

        l_bs = len(self.batch_sizes)
        nb_dataset = len(self.target_keys)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.max_target_batch_val = 0
        for i, key in enumerate(self.target):
            if not self.drop_last:
                val = math.ceil(len(self.target[key]) / self.batch_sizes[i])
            else:
                val = len(self.target[key]) // self.batch_sizes[i]
            if val > self.max_target_batch_val:
                self.max_target_batch_key = key
                self.max_target_batch_val = val

    def __iter__(self):
        indices = self.target
        if self.shuffle:
            for key in self.target:
                indices[key] = np.random.permutation(indices[key])
        iterators = []
        if self.name == "train":
            for i, key in enumerate(self.target):
                if key == self.max_target_batch_key:
                    iterators.append(grouper(indices[key], self.batch_sizes[i]))
                else:
                    iterators.append(itertools.cycle(grouper(indices[key], self.batch_sizes[i])))

            return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))
        else:
            key = list(self.target.keys())[0]

            cur_batch_list = [
                    tuple(indices[key][i * self.batch_sizes[0] : (i + 1) * self.batch_sizes[0]])
                    for i in range(self.max_target_batch_val)
                ]
            return iter(cur_batch_list)

    def __len__(self):
        return self.max_target_batch_val
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_sizes={self.batch_sizes}, "
            f"key_file={self.key_file}, "
        )

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def build_batch_sampler(
    type: str,
    batch_size: int,
    batch_bins: int,
    shape_files: Union[Tuple[str, ...], List[str]],
    sort_in_batch: str = "descending",
    sort_batch: str = "ascending",
    drop_last: bool = False,
    min_batch_size: int = 1,
    fold_lengths: Sequence[int] = (),
    padding: bool = True,
) -> AbsSampler:
    """Helper function to instantiate BatchSampler.

    Args:
        type: mini-batch type. "unsorted", "sorted", "folded", "numel", or, "length"
        batch_size: The mini-batch size. Used for "unsorted", "sorted", "folded" mode
        batch_bins: Used for "numel" model
        shape_files: Text files describing the length and dimension
            of each features. e.g. uttA 1330,80
        sort_in_batch:
        sort_batch:
        drop_last:
        min_batch_size:  Used for "numel" or "folded" mode
        fold_lengths: Used for "folded" mode
        padding: Whether sequences are input as a padded tensor or not.
            used for "numel" mode
    """
    assert check_argument_types()

    if type == "unsorted":
        retval = UnsortedBatchSampler(
            batch_size=batch_size, key_file=shape_files[0], drop_last=drop_last
        )

    elif type == "sorted":
        retval = SortedBatchSampler(
            batch_size=batch_size,
            shape_file=shape_files[0],
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
        )

    elif type == "folded":
        if len(fold_lengths) != len(shape_files):
            raise ValueError(
                f"The number of fold_lengths must be equal to "
                f"the number of shape_files: "
                f"{len(fold_lengths)} != {len(shape_files)}"
            )
        retval = FoldedBatchSampler(
            batch_size=batch_size,
            shape_files=shape_files,
            fold_lengths=fold_lengths,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            min_batch_size=min_batch_size,
        )

    elif type == "numel":
        retval = NumElementsBatchSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    elif type == "length":
        retval = LengthBatchSampler(
            batch_bins=batch_bins,
            shape_files=shape_files,
            sort_in_batch=sort_in_batch,
            sort_batch=sort_batch,
            drop_last=drop_last,
            padding=padding,
            min_batch_size=min_batch_size,
        )

    else:
        raise ValueError(f"Not supported: {type}")
    assert check_return_type(retval)
    return retval
