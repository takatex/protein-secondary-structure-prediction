# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Sequential(
            nn.Conv1d(42, 64, 3, 1, 3 // 2),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(42, 64, 5, 1, 5 // 2),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv1d(42, 64, 3, 1, 3 // 2),
            nn.ReLU())

        # LSTM(input_size, hidden_size, num_layers, bias,
        #      batch_first, dropout, bidirectional)
        self.blstm = nn.LSTM(192, 256, 3, True, True, 0, True)

        self.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 8))

    def forward(self, x):
        # obtain multiple local contextual feature map
        conv_out = torch.cat([self.conv1(x), self.conv2(x), self.conv3(x)], dim=1)

        # Turn (batch_size x hidden_size x seq_len)
        # into (batch_size x seq_len x hidden_size)
        conv_out = conv_out.transpose(1, 2)

        # bidirectional lstm
        out, _ = self.blstm(conv_out)

        # Output shape is (batch_size x seq_len x output_size)
        out = self.fc(out)
        out = F.softmax(out, dim=2)
        return out
