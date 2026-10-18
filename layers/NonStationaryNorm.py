import torch
import torch.nn as nn


class AdaptiveRevIN(nn.Module):
    """
    Adaptive reversible instance normalization for forecasting tensors.

    Input tensors are expected to have shape [B, L, N], where B is batch size,
    L is sequence length, and N is the number of variables/features. Call
    ``normalize`` on the encoder/input window and ``denormalize`` on the model
    forecast. The statistics from the most recent normalize call are reused for
    denormalization, which is the usual RevIN forecasting workflow.
    """

    def __init__(
        self,
        num_features,
        hidden_dim=64,
        eps=1e-5,
        affine=True,
        detach_stats=True,
        recent_ratio=0.25,
        adaptive_scale=0.1,
    ):
        super(AdaptiveRevIN, self).__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if not 0 < recent_ratio <= 1:
            raise ValueError("recent_ratio must be in (0, 1]")

        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.detach_stats = detach_stats
        self.recent_ratio = recent_ratio
        self.adaptive_scale = adaptive_scale

        stats_dim = 10
        if affine:
            self.affine_net = nn.Sequential(
                nn.Linear(stats_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )
            nn.init.zeros_(self.affine_net[-1].weight)
            nn.init.zeros_(self.affine_net[-1].bias)
        else:
            self.affine_net = None

        self.register_buffer("_mean", torch.empty(0), persistent=False)
        self.register_buffer("_stdev", torch.empty(0), persistent=False)
        self.register_buffer("_gamma", torch.empty(0), persistent=False)
        self.register_buffer("_beta", torch.empty(0), persistent=False)

    def forward(self, x, mode="norm"):
        """
        Dispatch normalization by mode.

        Args:
            x: Tensor with shape [B, L, N].
            mode: One of "norm", "normalize", "denorm", or "denormalize".
        """
        if mode in ("norm", "normalize"):
            return self.normalize(x)
        if mode in ("denorm", "denormalize"):
            return self.denormalize(x)
        raise ValueError("mode must be one of: norm, normalize, denorm, denormalize")

    def normalize(self, x):
        """Normalize an input window and cache statistics for later denormalization."""
        self._check_input(x)

        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.eps)

        if self.detach_stats:
            stats_source = x.detach()
            mean_for_stats = mean.detach()
            stdev_for_stats = stdev.detach()
        else:
            stats_source = x
            mean_for_stats = mean
            stdev_for_stats = stdev

        gamma, beta = self._adaptive_affine(stats_source, mean_for_stats, stdev_for_stats)

        self._mean = mean_for_stats
        self._stdev = stdev_for_stats
        self._gamma = gamma
        self._beta = beta

        x = centered / stdev
        return x * gamma + beta

    def denormalize(self, x):
        """Invert the most recent normalization for a forecast tensor [B, L, N]."""
        self._check_input(x)
        if self._mean.numel() == 0 or self._stdev.numel() == 0:
            raise RuntimeError("denormalize called before normalize")

        gamma = self._gamma.clamp_min(self.eps)
        x = (x - self._beta) / gamma
        return x * self._stdev + self._mean

    def _adaptive_affine(self, x, mean, stdev):
        if not self.affine:
            ones = torch.ones_like(mean)
            zeros = torch.zeros_like(mean)
            return ones, zeros

        stats = self._window_stats(x, mean, stdev)
        gamma_raw, beta_raw = self.affine_net(stats).chunk(2, dim=-1)
        gamma = 1.0 + self.adaptive_scale * torch.tanh(gamma_raw)
        beta = self.adaptive_scale * torch.tanh(beta_raw)
        return gamma.transpose(1, 2), beta.transpose(1, 2)

    def _window_stats(self, x, mean, stdev):
        recent_len = max(1, int(x.size(1) * self.recent_ratio))
        recent = x[:, -recent_len:, :]
        diff = x[:, 1:, :] - x[:, :-1, :] if x.size(1) > 1 else torch.zeros_like(x)

        recent_mean = recent.mean(dim=1, keepdim=True)
        recent_stdev = torch.sqrt(torch.var(recent, dim=1, keepdim=True, unbiased=False) + self.eps)
        volatility = torch.sqrt(torch.var(diff, dim=1, keepdim=True, unbiased=False) + self.eps)

        stats = torch.cat(
            [
                mean,
                stdev,
                x[:, :1, :],
                x[:, -1:, :],
                recent_mean,
                recent_stdev,
                x[:, -1:, :] - x[:, :1, :],
                x.amin(dim=1, keepdim=True),
                x.amax(dim=1, keepdim=True),
                volatility,
            ],
            dim=1,
        )
        return stats.transpose(1, 2)

    def _check_input(self, x):
        if x.dim() != 3:
            raise ValueError("expected input shape [B, L, N]")
        if x.size(-1) != self.num_features:
            raise ValueError(
                "expected input with {} features, got {}".format(self.num_features, x.size(-1))
            )


NonStationaryNorm = AdaptiveRevIN
