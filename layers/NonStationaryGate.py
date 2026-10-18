import torch
import torch.nn as nn


class NonStationaryGate(nn.Module):
    """
    Shift-aware gate for fusing hidden representations.

    Args:
        d_model: Hidden representation size.
        hidden_dim: MLP hidden size. Defaults to ``2 * d_model``.
        gate_dim: Number of values emitted per token. Defaults to ``d_model`` so
            the output can gate hidden states elementwise.
        num_gates: Number of gates to emit. Use values greater than one for
            branch-fusion weights.
        recent_ratio: Fraction of the input window used as the recent segment.
        min_recent: Minimum number of recent timesteps.
        dropout: Dropout probability in the gate MLP.
        eps: Numerical stability constant for std/ratio features.
        activation: ``"sigmoid"`` for independent gates or ``"softmax"`` for
            branch-fusion weights over ``num_gates``.
        detach_stats: Detach input statistics from the forecasting backbone.
    """

    def __init__(
        self,
        d_model,
        hidden_dim=None,
        gate_dim=None,
        num_gates=1,
        recent_ratio=0.25,
        min_recent=2,
        dropout=0.1,
        eps=1e-5,
        activation="sigmoid",
        detach_stats=True,
    ):
        super(NonStationaryGate, self).__init__()

        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_gates <= 0:
            raise ValueError("num_gates must be positive")
        if not 0 < recent_ratio <= 1:
            raise ValueError("recent_ratio must be in (0, 1]")
        if min_recent <= 0:
            raise ValueError("min_recent must be positive")
        if activation not in ("sigmoid", "softmax"):
            raise ValueError("activation must be 'sigmoid' or 'softmax'")

        hidden_dim = hidden_dim or 2 * d_model
        gate_dim = gate_dim or d_model

        self.d_model = d_model
        self.gate_dim = gate_dim
        self.num_gates = num_gates
        self.recent_ratio = recent_ratio
        self.min_recent = min_recent
        self.eps = eps
        self.activation = activation
        self.detach_stats = detach_stats

        # Per variable shift features:
        # mean shift, std shift, slope, volatility, recent volatility shift,
        # range, last z-score, first z-score.
        self.stat_proj = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.hidden_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_gates * gate_dim),
        )

    def forward(self, x, hidden=None, target_tokens=None, return_features=False):
        """
        Args:
            x: Input window with shape ``[B, L, N]``.
            hidden: Optional hidden representation with shape
                ``[B, tokens, d_model]``.
            target_tokens: Token count to emit when ``hidden`` is not provided.
                Defaults to ``N``.
            return_features: Return the raw shift features with the gate.

        Returns:
            Gate tensor shaped ``[B, tokens, gate_dim]`` when ``num_gates == 1``.
            Otherwise returns ``[B, tokens, num_gates, gate_dim]``.
        """
        if x.dim() != 3:
            raise ValueError("x must have shape [B, L, N]")
        if hidden is not None and hidden.dim() != 3:
            raise ValueError("hidden must have shape [B, tokens, d_model]")
        if hidden is not None and hidden.size(-1) != self.d_model:
            raise ValueError("hidden last dimension must match d_model")

        features = self._shift_features(x)
        stat_context = self.stat_proj(features)

        if hidden is not None:
            tokens = hidden.size(1)
            stat_context = self._match_tokens(stat_context, tokens)
            gate_input = stat_context + self.hidden_proj(hidden)
        else:
            tokens = target_tokens or x.size(-1)
            stat_context = self._match_tokens(stat_context, tokens)
            gate_input = stat_context

        gate = self.out_proj(gate_input)
        gate = gate.view(x.size(0), tokens, self.num_gates, self.gate_dim)

        if self.activation == "softmax":
            gate = torch.softmax(gate, dim=2)
        else:
            gate = torch.sigmoid(gate)

        if self.num_gates == 1:
            gate = gate.squeeze(2)

        if return_features:
            return gate, features
        return gate

    def _shift_features(self, x):
        if self.detach_stats:
            x = x.detach()

        length = x.size(1)
        recent_len = max(self.min_recent, int(round(length * self.recent_ratio)))
        recent_len = min(length, recent_len)
        recent = x[:, -recent_len:, :]

        full_mean = x.mean(dim=1)
        full_std = torch.sqrt(x.var(dim=1, unbiased=False) + self.eps)
        recent_mean = recent.mean(dim=1)
        recent_std = torch.sqrt(recent.var(dim=1, unbiased=False) + self.eps)

        diff = x[:, 1:, :] - x[:, :-1, :]
        if diff.size(1) > 0:
            volatility = torch.sqrt(diff.var(dim=1, unbiased=False) + self.eps)
            recent_diff = diff[:, -max(recent_len - 1, 1):, :]
            recent_volatility = torch.sqrt(
                recent_diff.var(dim=1, unbiased=False) + self.eps
            )
        else:
            volatility = torch.zeros_like(full_std)
            recent_volatility = torch.zeros_like(full_std)

        scale = full_std + self.eps
        mean_shift = (recent_mean - full_mean) / scale
        std_shift = torch.log((recent_std + self.eps) / scale)
        slope = (x[:, -1, :] - x[:, 0, :]) / scale
        volatility_shift = torch.log(
            (recent_volatility + self.eps) / (volatility + self.eps)
        )
        volatility = volatility / scale
        value_range = (x.max(dim=1).values - x.min(dim=1).values) / scale
        last_z = (x[:, -1, :] - full_mean) / scale
        first_z = (x[:, 0, :] - full_mean) / scale

        features = torch.stack(
            [
                mean_shift,
                std_shift,
                slope,
                volatility,
                volatility_shift,
                value_range,
                last_z,
                first_z,
            ],
            dim=-1,
        )
        return torch.nan_to_num(features)

    def _match_tokens(self, stat_context, tokens):
        if stat_context.size(1) == tokens:
            return stat_context

        pooled = stat_context.mean(dim=1, keepdim=True)
        return pooled.expand(-1, tokens, -1)
