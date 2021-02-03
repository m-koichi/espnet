from distutils.version import LooseVersion
from typing import Sequence
from typing import Union

import torch
import numpy as np
import random

from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from egs2.dcase20t4.sed1.pyscripts.sed.dataaug.abs_dataaug import AbsDataAug
from espnet2.layers.mask_along_axis import MaskAlongAxis
from espnet2.layers.time_warp import TimeWarp


if LooseVersion(torch.__version__) >= LooseVersion("1.1"):
    DEFAULT_TIME_WARP_MODE = "bicubic"
else:
    # pytorch1.0 doesn't implement bicubic
    DEFAULT_TIME_WARP_MODE = "bilinear"

class GaussianNoise(AbsDataAug):
    """ Pad or truncate a sequence given a number of frames
           Args:
               mean: float, mean of the Gaussian noise to add
           Attributes:
               std: float, std of the Gaussian noise to add
    """

    def __init__(self, mean=0., std=None, snr=None):
        super().__init__()
        self.mean = mean
        self.std = std
        self.snr = snr

    @staticmethod
    def gaussian_noise(features, snr):
        """Apply gaussian noise on each point of the data
                Args:
                    features: numpy.array, features to be modified
                Returns:
                    numpy.ndarray
                    Modified features
                """
        # If using source separation, using only the first audio (the mixture) to compute the gaussian noise,
        # Otherwise it just removes the first axis if it was an extended one
        if len(features.shape) == 3:
            feat_used = features[0]
        else:
            feat_used = features
        std = np.sqrt(np.mean((feat_used ** 2) * (10 ** (-snr / 10)), axis=-2))
        try:
            noise = np.random.normal(0, std, features.shape)
        except Exception as e:
            warnings.warn(f"the computed noise did not work std: {std}, using 0.5 for std instead")
            noise = np.random.normal(0, 0.5, features.shape)

        return features + noise

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified
            Returns:
                (np.array, np.array)
                (original data, noisy_data (data + noise))
                Note: return 2 values! needed for mean teacher!
        """
        if self.std is not None:
            noisy_data = data + np.abs(np.random.normal(0, 0.5 ** 2, data.shape))
        elif self.snr is not None:
            noisy_data = self.gaussian_noise(data, self.snr)
        else:
            raise NotImplementedError("Only (mean, std) or snr can be given")
        # return data, noisy_data
        return noisy_data


class FrequencyMask(AbsDataAug):
    def __init__(self,
                 freq_mask_width_range: Union[int, Sequence[int]] = (0, 20),
                 num_freq_mask: int = 2):
        super().__init__()
        self.freq_mask = MaskAlongAxis(
                dim="freq",
                mask_width_range=freq_mask_width_range,
                num_mask=num_freq_mask,
            )

    def transform_data(self, x, x_lengths):
        x, _ = self.freq_mask(x)
        return x, x_lengths

    def transform_data_with_label(self, x, x_lengths, label, label_lengths):
        x, _ = self.freq_mask(x)
        return x, x_lengths, label, label_lengths


class TimeShifting(AbsDataAug):
    def __init__(self,
                 mean=0,
                 std=90,
                 max_shift_frames: int = 40):
        super().__init__()
        self.max_shift_frames = max_shift_frames
        self.mean = mean
        self.std = std

    def transform_data(self, x, x_lengths):
        # shifts = random.randint(0, self.max_shift_frames)
        shifts = int(np.random.normal(self.mean, self.std))
        
        # data shifts along time axis
        x = torch.roll(x, shifts, dims=1)

        #TODO: fix sequence lengths after applied data augmentation
        return x, x_lengths

    def transform_data_with_label(self, x, x_lengths, label, label_lengths):
        # shifts = random.randint(0, self.max_shift_frames)
        shifts = int(np.random.normal(self.mean, self.std))
        
        # data shifts along time axis
        x = torch.roll(x, shifts, dims=1)
        label = torch.roll(label, shifts, dims=1)

        #TODO: fix sequence lengths after applied data augmentation
        return x, x_lengths, label, label_lengths

class Mixup(AbsDataAug):
    """Implementation of mixup.
    Reference
    """

    def __init__(self,
                 alpha: float = 0.2):
        assert 0 <= alpha <= 1.0
        self.alpha = alpha
        super().__init__()
        
    def transform_data_with_label(self, x: torch.Tensor, x_lengths: torch.Tensor, label: torch.Tensor, label_lengths: torch.Tensor):

        bs = x.size(0)
        c = np.random.beta(self.alpha, self.alpha)
        perm = torch.randperm(bs).to(x.device)

        x = c * x + (1-c) * x[perm, :]
        label = c * label + (1-c) * label[perm, :]

        #TODO: fix sequence lengths after applied data augmentation
        return x, x_lengths, label, label_lengths


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
        Example of transform: ToTensor()
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(
                self,
                x: torch.Tensor,
                x_lengths: torch.Tensor = None,
                label: torch.Tensor = None,
                label_lengths: torch.Tensor = None):
        for t in self.transforms:
            x, x_lengths, label, label_lengths = t(x, x_lengths, label, label_lengths)
        return x, x_lengths, label, label_lengths 

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string
