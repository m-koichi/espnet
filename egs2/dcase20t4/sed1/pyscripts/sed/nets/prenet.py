import torch
import torch.nn as nn

from egs2.dcase20t4.sed1.pyscripts.sed.abs_sed import AbsSED

# dcase2020 baseline module from https://github.com/turpaultn/dcase20_task4/tree/master/baseline
class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res


class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res


class CNN(torch.nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)],
                 use_tagging_token=True,
                 ):
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                cnn.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))

        batch_norm = True
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels

        self.cnn = cnn
        self.use_tagging_token = use_tagging_token
        if use_tagging_token:
            self.token_emb = torch.nn.Linear(1, nb_filters[-1])
        self.pooling_time_ratio = 1
        for pool in pooling:
            self.pooling_time_ratio *= pool[0]

    def _load(self, filename=None, parameters=None):
        if filename is not None:
            self.cnn.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.cnn.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def _state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def _save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x, lengths=None):
        # input size : (batch_size, n_channels, n_frames, n_freq) 
        # output size : (Batch, Time, Dim)
        # conv features
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        assert x.size(-1) == 1
        x = x.squeeze(-1).permute(0 ,2, 1)
        lengths = lengths // self.pooling_time_ratio

        if self.use_tagging_token:
            token = self.token_emb(torch.ones((x.size(0), 1, 1)).to(x.device))
            x = torch.cat([token, x], dim=1)
            lengths += 1
        return x, lengths
