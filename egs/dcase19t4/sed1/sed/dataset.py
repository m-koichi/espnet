import torch
import numpy as np
from data_loader import LoadInputsAndTargetsForSED
# from utils.io_utils import pad_trunc_sequence
from torch.utils.data import Dataset
import os
import re

import ipdb
import pdb
import kaldiio
from transforms import TimeShift

class CustomConverter(object):
    """Custom batch converter for Pytorch

    :param int subsampling_factor : The subsampling factor
    """

    def __init__(self, subsampling_factor=1, preprocess_conf=None):
        self.subsampling_factor = subsampling_factor
        self.load_inputs_and_targets = LoadInputsAndTargetsForSED(
            mode='sed', load_output=True, preprocess_conf=preprocess_conf)
        self.ignore_id = -1

    def transform(self, item):
        return self.load_inputs_and_targets(item)

    def __call__(self, batch, device):
        """Transforms a batch and send it to a device

        :param list batch: The batch to transform
        :param torch.device device: The device to send to
        :return: a tuple xs_pad, ilens, ys_pad
        :rtype (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad

class SEDDataset(Dataset):
    '''Sound Event Detection
    '''
    def __init__(self, json_data, label_type, sequence_length, pooling_time_ratio=1, transforms=None,
                 time_shift=True):
        # self.batchset = batchset
        self.json_data = json_data
        self.label_type = label_type
        self.sequence_length = sequence_length
        self.data_ids = [k for k in self.json_data.keys()]
        self.transforms = transforms
        self.pooling_time_ratio = pooling_time_ratio
        self.time_shift = time_shift
        
    def __getitem__(self, index):
        data_id = self.data_ids[index]
        x = kaldiio.load_mat(self.json_data[data_id]['input'][0]['feats'])

        # x_ = np.load(os.path.join('./DCASE2019_task4/dataset/features/sr44100_win2048_hop511_mels64_nolog/features', re.sub('^.*?-', '', data_id + '.npy')))

        if self.transforms:
            for transform in self.transforms:
                # pdb.set_trace()
                x = transform(x)
        x = pad_trunc_sequence(x, self.sequence_length)
        # pdb.set_trace()

        if self.label_type == 'strong':
            output = self.json_data[data_id]['output'][0]
            y = np.zeros((output['label'][0]['shape'][1], len(output['label'])))
            for label_idx, label in enumerate(output['label']):
                y[:, label_idx] = np.array(label['tokenid'])
            # pdb.set_trace()
            y = y[self.pooling_time_ratio-1::self.pooling_time_ratio, :]
            # if self.time_shift:
                # import ipdb
                # ipdb.set_trace()
                # x, y = TimeShift()(x, y)
            # ipdb.set_trace()
            assert x.shape[0] == y.shape[0] * self.pooling_time_ratio
        elif self.label_type == 'weak':
            output = self.json_data[data_id]['output'][0]
            y = np.array(output['label']['tokenid'])
        elif self.label_type == 'unlabel':
            y = np.array([-1])
        else:
            raise ValueError(f'label_type "{self.label_type}" is not suported.')

        return torch.from_numpy(x).float().unsqueeze(0), torch.from_numpy(y).float(), data_id

    def __len__(self):
        return len(self.json_data)

    #         if len(inp['label']) != 1:
    #                     x = np.zeros((len(inp['label']), inp['label'][0]['shape'][1]))
    #                     for label_idx, label in enumerate(inp['label']):
    #                         x[label_idx] = np.array(label['tokenid'])
    #                 else:
    #                     x = np.array(label['tokenid'])
    #     y = json_data[data_id]['input'][0]['feats']

    #     return to_tensor(x), to_tensor(y)
    #     xs, ys = self.load_inputs_and_targets(self.batchset[index])
    #     pdb.set_trace()
    #     xs_new = torch.from_numpy([pad_trunc_sequence(x) for x in xs])
    #     if label_type == 'strong':
    #         ys_new = torch.from_numpy([pad_trunc_sequence(x) for y in ys])
    #     elif label_type == 'weak':
    #         ys_new = torch.from_numpy(ys)
    #     elif label_type == 'unlabel':
    #         ys_new = None
    #     return xs_new, ys_new
    # def __len__(self):
    #     return len(self.json_data)


def pad_trunc_sequence(x, max_sequence_length):
    length = x.shape[0]
    if length < max_sequence_length:
        x_new = np.pad(x, [(0, max_sequence_length - length), (0, 0)], 'constant')
    elif length > max_sequence_length:
        x_new = x[0:max_sequence_length, :]
    else:
        x_new = x
    return x_new

