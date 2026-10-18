import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    Input shape: [B, L, N].
    """
    def __init__(self, kernel_size, stride=1):
        super(moving_avg, self).__init__()
        if kernel_size <= 0:
            raise ValueError('kernel_size must be a positive integer')
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('moving_avg expects input with shape [B, L, N]')

        # padding on both ends of time series
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        if left > 0:
            front = x[:, 0:1, :].repeat(1, left, 1)
            x = torch.cat([front, x], dim=1)
        if right > 0:
            end = x[:, -1:, :].repeat(1, right, 1)
            x = torch.cat([x, end], dim=1)

        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block.
    Returns seasonal and trend components for input shape [B, L, N].
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class high_freq_decomp(nn.Module):
    """
    High-frequency decomposition block.
    Returns high-frequency and low-frequency components for input shape [B, L, N].
    """
    def __init__(self, kernel_size):
        super(high_freq_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        low_freq = self.moving_avg(x)
        high_freq = x - low_freq
        return high_freq, low_freq


class series_decomp_residual(nn.Module):
    """
    Trend-seasonal-residual decomposition block.
    Returns seasonal, trend, and residual components for input shape [B, L, N].
    """
    def __init__(self, trend_kernel_size, seasonal_kernel_size=None):
        super(series_decomp_residual, self).__init__()
        seasonal_kernel_size = seasonal_kernel_size or trend_kernel_size
        self.trend_moving_avg = moving_avg(trend_kernel_size, stride=1)
        self.seasonal_moving_avg = moving_avg(seasonal_kernel_size, stride=1)

    def forward(self, x):
        trend = self.trend_moving_avg(x)
        detrended = x - trend
        seasonal = self.seasonal_moving_avg(detrended)
        residual = detrended - seasonal
        return seasonal, trend, residual


MovingAvg = moving_avg
SeriesDecomp = series_decomp
HighFreqDecomp = high_freq_decomp
SeriesDecompResidual = series_decomp_residual


__all__ = [
    'moving_avg',
    'series_decomp',
    'high_freq_decomp',
    'series_decomp_residual',
    'MovingAvg',
    'SeriesDecomp',
    'HighFreqDecomp',
    'SeriesDecompResidual',
]
