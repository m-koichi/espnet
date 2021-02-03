from typing import Optional
from typing import Tuple

import torch
import random


class AbsDataAug(torch.nn.Module):
    """Abstract class for the augmentation of spectrogram
    """
    def __init__(self):
        super().__init__()
        self.p = 0.5

    def transform_data(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor = None):
        raise NotImplementedError

    def transform_data_with_label(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor = None,
        label: torch.Tensor = None,
        label_lengths: torch.Tensor = None):
        x, x_lengths = self.transform_data(x, x_lengths)
        return x, x_lengths, label, label_lengths

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor = None,
        label: torch.Tensor = None,
        label_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            if label is None:
                if self.p > random.random():
                    x, x_lengths = self.transform_data(x, x_lengths)
                return x, x_lengths, label, label_lengths
            else:
                if self.p > random.random():
                    x, x_lengths, label, label_lengths = self.transform_data_with_label(
                        x, x_lengths, label, label_lengths)
                return x, x_lengths, label, label_lengths