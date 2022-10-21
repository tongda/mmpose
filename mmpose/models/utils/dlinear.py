# Copyright (c) OpenMMLab. All rights reserved.
# import numpy as np
import torch
import torch.nn as nn

# import torch.nn.functional as F


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    """Decomposition-Linear."""

    def __init__(self,
                 num_token,
                 in_channels,
                 out_channels,
                 sigma=6,
                 individual=True):
        super(DLinear, self).__init__()
        self.num_token = num_token
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Decompsition Kernel Size
        kernel_size = int((sigma * 20 - 7) // 3)
        kernel_size -= int((kernel_size % 2) == 0)
        # kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.num_token):
                self.Linear_Seasonal.append(
                    nn.Linear(self.in_channels, self.out_channels))
                self.Linear_Trend.append(
                    nn.Linear(self.in_channels, self.out_channels))

        else:
            self.Linear_Seasonal = nn.Linear(self.in_channels,
                                             self.out_channels)
            self.Linear_Trend = nn.Linear(self.in_channels, self.out_channels)

    def forward(self, x):
        seasonal_init, trend_init = self.decompsition(x)

        if self.individual:
            seasonal_output = torch.zeros([
                seasonal_init.size(0),
                seasonal_init.size(1), self.out_channels
            ],
                                          dtype=seasonal_init.dtype).to(
                                              seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0),
                 trend_init.size(1), self.out_channels],
                dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.num_token):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x
