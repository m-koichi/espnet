from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import Union

import numpy as np
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import Iterable

from typing import Collection

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.cleaner import TextCleaner
from espnet2.text.token_id_converter import TokenIDConverter
import math


class AbsPreprocessor(ABC):
    def __init__(self, train: bool):
        self.train = train

    @abstractmethod
    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class CommonPreprocessor(AbsPreprocessor):
    def __init__(
        self,
        train: bool,
        max_len_seconds: float = 10.,
        sample_rate: int = 16000,
        hop_length: int = 323,
        label_list: Union[Path, str, Iterable[str]] = None,
    ):
        super().__init__(train)
        self.train = train
        self.max_len_seconds = max_len_seconds
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        if label_list is None:
            raise ValueError("token_list is required if token_type is not None")
        with open(label_list) as f:
            self.label_list = [label.strip() for label in f.readlines()]
        
        self.label_dict = {key: value for value, key in enumerate(sorted(self.label_list))}

    def __call__(
        self, uid: str, data: Dict[str, Union[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        assert check_argument_types()
        assert "label" in data

        data["audio"] = self._adjust_length(data["audio"], int(self.max_len_seconds * self.sample_rate))
        data["label"] = np.array(self._convert_label(data["label"]), dtype=np.float64)

        assert check_return_type(data)
        return data

    def _convert_label(self, label):
        labels = label.strip().split(" ")
        n_frames = math.ceil(self.max_len_seconds * self.sample_rate / self.hop_length)
        label = np.zeros((n_frames, len(self.label_dict)), dtype=float)
        # check label type
        # unlabeled
        if labels[0] == "unlabeled":
            pass
        # strong label
        elif labels[0][0] == "(" and labels[0][-1] == ")":
            for metadata in labels:
                onset, offset, event_class = metadata[1:-1].split(",")
                onset = int(float(onset) * self.sample_rate // self.hop_length)
                offset = int(float(offset) * self.sample_rate // self.hop_length)
                label[onset:offset, self.label_dict[event_class]] = 1
        # weak label
        else:
            for l in labels[0].strip().split(','):
                label[:, self.label_dict[l]] = 1
        return label

    def _adjust_length(self, data, length):
        if len(data) >= length:
            data = data[:length]
        else:
            data = np.pad(data, [0, length-len(data)], "constant")
        return data
