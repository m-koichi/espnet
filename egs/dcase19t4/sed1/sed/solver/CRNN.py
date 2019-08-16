import warnings

import torch.nn as nn
import torch

from solver.adaptive_pooling import AutoPool

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


class CNN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
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
        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels

        self.cnn = cnn

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.cnn.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.cnn.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        return x


class BidirectionalGRU(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(n_in, n_hidden, bidirectional=True, dropout=dropout, batch_first=True, num_layers=num_layers)

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut, dropout=0, num_layers=1):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden // 2, bidirectional=True, batch_first=True,
                           dropout=dropout, num_layers=num_layers)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        b, T, h = recurrent.size()
        t_rec = recurrent.contiguous().view(b * T, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(b, T, -1)
        return output

class CRNN(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(CRNN, self).__init__()
        self.attention = attention
        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)
        self.autopool = AutoPool(10)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if rnn_type == 'BGRU':
            self.rnn = BidirectionalGRU(self.cnn.nb_filters[-1],
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell*2, nclass)
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell*2, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, parameters):
        self.cnn.load(parameters)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        self.cnn.load(parameters=parameters["cnn"])
        self.rnn.load_state_dict(parameters["rnn"])
        self.dense.load_state_dict(parameters["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'rnn': self.rnn.state_dict(), 'dense': self.dense.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn("Output shape is: {}".format((bs, frames, chan * freq)))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        x = self.rnn(x)
        x = self.dropout(x)
        strong = self.dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak, alpha = self.autopool(strong)
        return strong, weak

class CRNN2(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(CRNN2, self).__init__()
        self.attention = attention
        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)
        self.autopool = AutoPool(10)
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        self.train_cnn = train_cnn
        if rnn_type == 'BGRU':
            self.rnn = BidirectionalGRU(self.cnn.nb_filters[-1],
                                        n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
            self.ead_rnn = BidirectionalGRU(self.cnn.nb_filters[-1],
                                            n_RNN_cell, dropout=dropout_recurrent, num_layers=n_layers_RNN)
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell*2, nclass)
        self.ead_dense = nn.Linear(n_RNN_cell*2, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.at_dense = nn.Linear(self.cnn.nb_filters[-1], nclass)
        self.sigmoid = nn.Sigmoid()
        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell*2, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def load_cnn(self, parameters):
        self.cnn.load(parameters)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        self.cnn.load(parameters=parameters["cnn"])
        self.rnn.load_state_dict(parameters["rnn"])
        self.dense.load_state_dict(parameters["dense"])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn_ead": self.ead_rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense_ead': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn': self.cnn.state_dict(), 'rnn': self.rnn.state_dict(), 'rnn_ead': self.rnn_ead.state_dict(),
                      'dense': self.dense.state_dict(), 'dense_ead': self.dense_ead.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn("Output shape is: {}".format((bs, frames, chan * freq)))
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        x_strong = self.rnn(x)
        x_strong = self.dropout(x_strong)
        strong = self.dense(x_strong)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)

        # Event activity detection
        x_ead = self.ead_rnn(x)
        ead = self.ead_dense(x_ead)
        ead = self.sigmoid(ead)

        # Audio tagging
        x_at = self.gap(x.permute(0, 2, 1))
        # import ipdb
        # ipdb.set_trace()
        x_at = x_at.view(x_at.size(0), -1)
        x_at = self.at_dense(x_at)
        tag = self.sigmoid(x_at)

        # if self.attention:
        #     sof = self.dense_softmax(x)  # [bs, frames, nclass]
        #     sof = self.softmax(sof)
        #     sof = torch.clamp(sof, min=1e-7, max=1)
        #     weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        # else:
        #     weak, alpha = self.autopool(strong)
        return strong, tag, ead



if __name__ == '__main__':
    CRNN(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
         pooling=[(1, 4), (1, 4), (1, 4)])
