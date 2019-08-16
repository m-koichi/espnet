import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import


class CNN(nn.Module):
    def __init__(self, in_channel, activation='Relu', dropout=0,
                kernel_size, padding, stride, nb_filters,
                pooling):
        super(CNN, self).__init__()
        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    # def conv(i, batchNorm=False, dropout=Norm, activ='relu'):
    #     n_in = 
    def forward(self, x):
        # x.shape ->  [batch, 1, 64, 864]
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn4(self.conv4(x)))
        tf_maps = F.avg_pool2d(x, kernel_size=(1, 1))
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        framewise_vector = torch.mean(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict

