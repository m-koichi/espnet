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

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        return x
    
    
class SubSpecCNN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0, dense=False,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(SubSpecCNN, self).__init__()
        self.nb_filters = nb_filters
        cnn_high = nn.Sequential()
        cnn_middle = nn.Sequential()
        cnn_low = nn.Sequential()

        def build_conv(model, i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            model.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                model.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                model.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                model.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                model.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                model.add_module('cg{0}'.format(i), ContextGating(nOut))
            if dropout is not None:
                model.add_module('dropout{0}'.format(i),
                               nn.Dropout(dropout))
            model.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels
            return model

        batch_norm = True
        # 128x862x64
        for i in range(len(nb_filters)):
            cnn_high = build_conv(cnn_high, i, batch_norm, conv_dropout, activ=activation)
            cnn_middle = build_conv(cnn_middle, i, batch_norm, conv_dropout, activ=activation)
            cnn_low = build_conv(cnn_low, i, batch_norm, conv_dropout, activ=activation)
            

        self.cnn_high = cnn_high
        self.cnn_middle = cnn_middle
        self.cnn_low = cnn_low
        
        self.use_dense = dense
        if self.use_dense:
            self.dense = nn.Linear(nb_filters[-1] * 3, nb_filters[-1])

    def load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")
        self.cnn_high.load_state_dict(parameters['high'])
        self.cnn_middle.load_state_dict(parameters['high'])
        self.cnn_low.load_state_dict(parameters['high'])

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn_high": self.cnn_high.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "cnn_middle": self.cnn_middle.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'cnn_low': self.cnn_low.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def save(self, filename):
        parameters = {'cnn_high': self.cnn_high.state_dict(), 'cnn_middle': self.cnn_middle.state_dict(), 'cnn_low': self.cnn_low.state_dict()}
        torch.save(parameters, filename)

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        bin_range = x.size()[-1]
        x_high = self.cnn_high(x[:,:,:,:bin_range//2])
        x_middle = self.cnn_middle(x[:,:,:,bin_range//4:bin_range//4 * 3])
        x_low = self.cnn_low(x[:,:,:,bin_range//2:])
        if self.use_dense:
            x = torch.cat([x_high, x_middle, x_low], dim=1).squeeze(-1).permute(0, 2, 1)
            x = self.dense(x)
            x = x.permute(0, 2, 1).unsqueeze(-1)
        else:
            x = x_high + x_middle + x_low
        return x

    
class CNN_adaBN(nn.Module):

    def __init__(self, n_in_channel, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(CNN_adaBN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        
        self.conv0 = nn.Conv2d(n_in_channel, nb_filters[0], kernel_size[0], stride[0], padding[0])
        self.bn0_s = nn.BatchNorm2d(nb_filters[0], eps=0.001, momentum=0.99)
        self.bn0_t = nn.BatchNorm2d(nb_filters[0], eps=0.001, momentum=0.99)
        self.glu0 = GLU(nb_filters[0])
        self.do0 = nn.Dropout(conv_dropout)
        self.pool0 = nn.AvgPool2d(pooling[0])
        
        self.conv1 = nn.Conv2d(nb_filters[0], nb_filters[1], kernel_size[1], stride[1], padding[1])
        self.bn1_s = nn.BatchNorm2d(nb_filters[1], eps=0.001, momentum=0.99)
        self.bn1_t = nn.BatchNorm2d(nb_filters[1], eps=0.001, momentum=0.99)
        self.glu1 = GLU(nb_filters[1])
        self.do1 = nn.Dropout(conv_dropout)
        self.pool1 = nn.AvgPool2d(pooling[1])
        
        self.conv2 = nn.Conv2d(nb_filters[1], nb_filters[2], kernel_size[2], stride[2], padding[2])
        self.bn2_s = nn.BatchNorm2d(nb_filters[2], eps=0.001, momentum=0.99)
        self.bn2_t = nn.BatchNorm2d(nb_filters[2], eps=0.001, momentum=0.99)
        self.glu2 = GLU(nb_filters[2])
        self.do2 = nn.Dropout(conv_dropout)
        self.pool2 = nn.AvgPool2d(pooling[2])
        
        
        
#         def conv(i, batchNormalization=False, dropout=None, activ="relu"):
#             nIn = n_in_channel if i == 0 else nb_filters[i - 1]
#             nOut = nb_filters[i]
#             cnn.add_module('conv{0}'.format(i),
#                            nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
#             if batchNormalization:
#                 cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
#             if activ.lower() == "leakyrelu":
#                 cnn.add_module('relu{0}'.format(i),
#                                nn.LeakyReLU(0.2))
#             elif activ.lower() == "relu":
#                 cnn.add_module('relu{0}'.format(i), nn.ReLU())
#             elif activ.lower() == "glu":
#                 cnn.add_module('glu{0}'.format(i), GLU(nOut))
#             elif activ.lower() == "cg":
#                 cnn.add_module('cg{0}'.format(i), ContextGating(nOut))
#             if dropout is not None:
#                 cnn.add_module('dropout{0}'.format(i),
#                                nn.Dropout(dropout))

#         batch_norm = True
#         # 128x862x64
#         for i in range(len(nb_filters)):
#             conv(i, batch_norm, conv_dropout, activ=activation)
#             cnn.add_module('pooling{0}'.format(i), nn.AvgPool2d(pooling[i]))  # bs x tframe x mels

#         self.cnn = cnn

    def load(self, filename=None, parameters=None):
        if filename is not None:
            self.load_state_dict(torch.load(filename))
        elif parameters is not None:
            self.load_state_dict(parameters)
        else:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

#     def state_dict(self, destination=None, prefix='', keep_vars=False):
#         return self.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def forward(self, x, domain):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.conv0(x)
        if domain == 'source':
            x = self.bn0_s(x)
        elif domain == 'target':
            x = self.bn0_t(x)
        else:
            raise ValueError
        x = self.glu0(x)
        x = self.do0(x)
        x = self.pool0(x)
        
        x = self.conv1(x)
        if domain == 'source':
            x = self.bn1_s(x)
        elif domain == 'target':
            x = self.bn1_t(x)
        else:
            raise ValueError
        x = self.glu1(x)
        x = self.do1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if domain == 'source':
            x = self.bn2_s(x)
        elif domain == 'target':
            x = self.bn2_t(x)
        else:
            raise ValueError
        x = self.glu2(x)
        x = self.do2(x)
        x = self.pool2(x)
        
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

    def _save(self, filename):
        torch.save(self.state_dict(), filename)

    def _load(self, filename=None, parameters=None):
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

    def _load_cnn(self, parameters):
        self.cnn.load(parameters)
        if not self.train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

    def _load(self, filename=None, parameters=None):
        if filename is not None:
            parameters = torch.load(filename)
            import ipdb
            ipdb.set_trace()
        if parameters is None:
            raise NotImplementedError("load is a filename or a list of parameters (state_dict)")

        self.cnn.load(parameters=parameters["cnn"])
        self.rnn.load_state_dict(parameters["rnn"])
        self.dense.load_state_dict(parameters["dense"])

    def _state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = {"cnn": self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      "rnn": self.rnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars),
                      'dense': self.dense.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)}
        return state_dict

    def _save(self, filename):
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
#             import ipdb
#             ipdb.set_trace()
#             attention_weight = self.logit(strong * sof)
            sof = self.softmax(sof)
#             sof = self.sigmoid(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak, alpha = self.autopool(strong)
        return strong, weak

    
class SubSpecCRNN(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(SubSpecCRNN, self).__init__()
        self.attention = attention
        self.cnn = SubSpecCNN(n_in_channel, activation, dropout, dense=True, **kwargs)
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
            import ipdb
            ipdb.set_trace()
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
#         import ipdb
#         ipdb.set_trace()
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
#             import ipdb
#             ipdb.set_trace()
#             attention_weight = self.logit(strong * sof)
            sof = self.softmax(sof)
#             sof = self.sigmoid(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak, alpha = self.autopool(strong)
        return strong, weak
    
    
class CRNN_adaBN(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(CRNN_adaBN, self).__init__()
        self.attention = attention
        self.cnn = CNN_adaBN(n_in_channel, activation, dropout, **kwargs)
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

    def forward(self, x, domain='target'):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x, domain)
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
#             import ipdb
#             ipdb.set_trace()
#             attention_weight = self.logit(strong * sof)
            sof = self.softmax(sof)
#             sof = self.sigmoid(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        else:
            weak, alpha = self.autopool(strong)
        return strong, weak

    
def logit(p):
    return torch.log(p/(1-p))

class CRNN2(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(CRNN2, self).__init__()
        self.attention = attention
        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)
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



class CRNN_Attn(nn.Module):

    def __init__(self, n_in_channel, nclass, attention=False, activation="Relu", dropout=0,
                 train_cnn=True, rnn_type='BGRU', n_RNN_cell=64, n_layers_RNN=1, dropout_recurrent=0, **kwargs):
        super(CRNN, self).__init__()
        self.attention = attention
        self.cnn = CNN(n_in_channel, activation, dropout, **kwargs)
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
#             import ipdb
#             ipdb.set_trace()
#             attention_weight = self.logit(strong * sof)
            sof = self.softmax(sof)
#             sof = self.sigmoid(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)   # [bs, nclass]
        return strong, weak

if __name__ == '__main__':
    CRNN(64, 10, kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
         pooling=[(1, 4), (1, 4), (1, 4)])
