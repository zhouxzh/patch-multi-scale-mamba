import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.NonStationaryGate import NonStationaryGate
from layers.SeriesDecomp import SeriesDecomp, SeriesDecompResidual
from mamba_ssm import Mamba


def build_state_space(configs):
    """Build the configured selective state-space block for opt-in fast paths."""
    backend = str(getattr(configs, "pms_ssm_backend", "mamba")).lower()
    if backend == "mamba":
        return Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=2,
            expand=1,
        )
    if backend == "mamba2":
        from mamba_ssm.modules.mamba2 import Mamba2

        head_dim = min(64, int(configs.d_model))
        if int(configs.d_model) % head_dim:
            raise ValueError("Mamba2 requires d_model divisible by its head dimension")
        return Mamba2(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=2,
            expand=1,
            headdim=head_dim,
        )
    raise ValueError("pms_ssm_backend must be mamba or mamba2")


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, dropout=0.0):
        super(FlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return self.dropout(x)


class PatchMambaBranch(nn.Module):
    def __init__(self, configs, input_len, patch_len=None, stride=None):
        super(PatchMambaBranch, self).__init__()
        self.patch_len = int(patch_len if patch_len is not None else configs.patch_len)
        self.stride = int(stride if stride is not None else configs.stride)
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            self.patch_len,
            self.stride,
            self.stride,
            configs.dropout,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=2,
                        expand=1,
                    ),
                    Mamba(
                        d_model=configs.d_model,
                        d_state=configs.d_state,
                        d_conv=2,
                        expand=1,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        patch_num = max(1, int((input_len - self.patch_len) / self.stride + 2))
        self.head = FlattenHead(
            configs.d_model * patch_num,
            configs.pred_len,
            dropout=configs.dropout,
        )

    def forward(self, x):
        # x: [B, L, C]
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        return self.head(enc_out)


class FastPatchMambaBranch(nn.Module):
    """Single-direction patch Mamba branch for latency-constrained forecasting."""

    def __init__(self, configs, input_len, patch_len=None, stride=None):
        super().__init__()
        self.patch_len = int(patch_len if patch_len is not None else configs.patch_len)
        self.stride = int(stride if stride is not None else configs.stride)
        self.patch_embedding = PatchEmbedding(
            configs.d_model,
            self.patch_len,
            self.stride,
            self.stride,
            configs.dropout,
        )
        self.mamba = build_state_space(configs)
        self.norm = nn.LayerNorm(configs.d_model)
        patch_num = max(1, int((input_len - self.patch_len) / self.stride + 2))
        self.head = FlattenHead(
            configs.d_model * patch_num,
            configs.pred_len,
            dropout=configs.dropout,
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x)
        enc_out = self.norm(enc_out + self.mamba(enc_out))
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        return self.head(enc_out.permute(0, 1, 3, 2))


class TemporalMambaBranch(nn.Module):
    """One full-resolution temporal Mamba scan shared across all channels."""

    def __init__(self, configs):
        super().__init__()
        self.input_projection = nn.Linear(1, configs.d_model)
        self.mamba = build_state_space(configs)
        self.norm = nn.LayerNorm(configs.d_model)
        self.head = FlattenHead(
            configs.d_model * configs.seq_len,
            configs.pred_len,
            dropout=configs.dropout,
        )

    def forward(self, x):
        batch_size, length, channels = x.shape
        tokens = x.permute(0, 2, 1).reshape(batch_size * channels, length, 1)
        encoded = self.input_projection(tokens)
        encoded = self.norm(encoded + self.mamba(encoded))
        encoded = encoded.reshape(batch_size, channels, length, encoded.size(-1))
        return self.head(encoded.permute(0, 1, 3, 2))


class InvertedMambaBranch(nn.Module):
    """One-direction Mamba over variable tokens, with full histories embedded per token."""

    def __init__(self, configs):
        super().__init__()
        self.embedding = nn.Linear(configs.seq_len, configs.d_model)
        self.mamba = Mamba(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_conv=2,
            expand=1,
        )
        self.norm = nn.LayerNorm(configs.d_model)
        self.head = nn.Linear(configs.d_model, configs.pred_len)

    def forward(self, x):
        tokens = self.embedding(x.permute(0, 2, 1))
        tokens = self.norm(tokens + self.mamba(tokens))
        return self.head(tokens)


class FusedDualPatchMamba(nn.Module):
    """Dual patch paths sharing one bidirectional Mamba scan per encoder layer."""

    def __init__(self, configs, fine_patch_len, fine_stride, coarse_patch_len, coarse_stride):
        super().__init__()
        self.fine_embedding = PatchEmbedding(
            configs.d_model, fine_patch_len, fine_stride, fine_stride, configs.dropout
        )
        self.coarse_embedding = PatchEmbedding(
            configs.d_model, coarse_patch_len, coarse_stride, coarse_stride, configs.dropout
        )
        self.layers = nn.ModuleList()
        for _ in range(configs.e_layers):
            self.layers.append(nn.ModuleDict({
                "mamba": Mamba(
                    d_model=configs.d_model,
                    d_state=configs.d_state,
                    d_conv=2,
                    expand=1,
                ),
                "norm1": nn.LayerNorm(configs.d_model),
                "conv1": nn.Conv1d(configs.d_model, configs.d_ff, kernel_size=1),
                "conv2": nn.Conv1d(configs.d_ff, configs.d_model, kernel_size=1),
                "norm2": nn.LayerNorm(configs.d_model),
            }))
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        fine_tokens = max(1, int((configs.seq_len - fine_patch_len) / fine_stride + 2))
        coarse_tokens = max(1, int((configs.seq_len - coarse_patch_len) / coarse_stride + 2))
        self.fine_head = FlattenHead(configs.d_model * fine_tokens, configs.pred_len, configs.dropout)
        self.coarse_head = FlattenHead(configs.d_model * coarse_tokens, configs.pred_len, configs.dropout)

    @staticmethod
    def _pad(x, length):
        if x.size(1) == length:
            return x
        return F.pad(x, (0, 0, 0, length - x.size(1)))

    def _encode(self, fine, coarse):
        fine_len, coarse_len = fine.size(1), coarse.size(1)
        max_len = max(fine_len, coarse_len)
        for layer in self.layers:
            merged = torch.cat((
                self._pad(fine, max_len),
                self._pad(coarse, max_len),
                self._pad(fine.flip(dims=[1]), max_len),
                self._pad(coarse.flip(dims=[1]), max_len),
            ), dim=0)
            encoded = layer["mamba"](merged)
            count = fine.size(0)
            forward_fine = encoded[:count, :fine_len]
            forward_coarse = encoded[count:2 * count, :coarse_len]
            reverse_fine = encoded[2 * count:3 * count, :fine_len].flip(dims=[1])
            reverse_coarse = encoded[3 * count:, :coarse_len].flip(dims=[1])
            fine = layer["norm1"](fine + forward_fine + reverse_fine)
            coarse = layer["norm1"](coarse + forward_coarse + reverse_coarse)
            fine_ff = self.dropout(self.activation(layer["conv1"](fine.transpose(-1, 1))))
            fine_ff = self.dropout(layer["conv2"](fine_ff).transpose(-1, 1))
            coarse_ff = self.dropout(self.activation(layer["conv1"](coarse.transpose(-1, 1))))
            coarse_ff = self.dropout(layer["conv2"](coarse_ff).transpose(-1, 1))
            fine = layer["norm2"](fine + fine_ff)
            coarse = layer["norm2"](coarse + coarse_ff)
        return fine, coarse

    def forward(self, x):
        x = x.permute(0, 2, 1)
        fine, n_vars = self.fine_embedding(x)
        coarse, coarse_vars = self.coarse_embedding(x)
        if n_vars != coarse_vars:
            raise RuntimeError("dual patch embeddings produced inconsistent variable counts")
        fine, coarse = self._encode(fine, coarse)
        fine = fine.reshape(-1, n_vars, fine.size(1), fine.size(2)).permute(0, 1, 3, 2)
        coarse = coarse.reshape(-1, n_vars, coarse.size(1), coarse.size(2)).permute(0, 1, 3, 2)
        return self.fine_head(fine), self.coarse_head(coarse)


class SharedMultiScalePatchMamba(nn.Module):
    """Multi-scale patch encoder with one shared bidirectional Mamba scan.

    The standard Regime-Gated Mamba implementation has independent forward and reverse Mamba modules
    for every scale.  This variant keeps scale-specific patch embeddings and
    forecasting heads, but batches every scale and direction through a shared
    encoder cell.  Padding is only appended to the end of each causal scan, so
    retained tokens are unaffected.
    """

    def __init__(self, configs, input_lengths):
        super().__init__()
        self.embeddings = nn.ModuleList([
            PatchEmbedding(
                configs.d_model,
                configs.patch_len,
                configs.stride,
                configs.stride,
                configs.dropout,
            )
            for _ in input_lengths
        ])
        self.layers = nn.ModuleList()
        for _ in range(configs.e_layers):
            self.layers.append(nn.ModuleDict({
                "mamba": Mamba(
                    d_model=configs.d_model,
                    d_state=configs.d_state,
                    d_conv=2,
                    expand=1,
                ),
                "norm1": nn.LayerNorm(configs.d_model),
                "conv1": nn.Conv1d(configs.d_model, configs.d_ff, kernel_size=1),
                "conv2": nn.Conv1d(configs.d_ff, configs.d_model, kernel_size=1),
                "norm2": nn.LayerNorm(configs.d_model),
            }))
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu if configs.activation == "relu" else F.gelu
        self.final_norm = nn.LayerNorm(configs.d_model)
        self.heads = nn.ModuleList([
            FlattenHead(
                configs.d_model * max(1, int((length - configs.patch_len) / configs.stride + 2)),
                configs.pred_len,
                dropout=configs.dropout,
            )
            for length in input_lengths
        ])

    @staticmethod
    def _pad(tokens, length):
        if tokens.size(1) == length:
            return tokens
        return F.pad(tokens, (0, 0, 0, length - tokens.size(1)))

    def _encode(self, tokens):
        lengths = [item.size(1) for item in tokens]
        max_len = max(lengths)
        for layer in self.layers:
            # Concatenating forward and reversed paths makes this one Mamba
            # launch per layer. The padded suffix is discarded after the scan.
            merged = torch.cat(
                [self._pad(item, max_len) for item in tokens]
                + [self._pad(item.flip(dims=[1]), max_len) for item in tokens],
                dim=0,
            )
            encoded = layer["mamba"](merged)
            batch_sizes = [item.size(0) for item in tokens]
            if len(set(batch_sizes)) != 1:
                raise RuntimeError("multi-scale patch batches must have the same size")
            batch_size = batch_sizes[0]
            forward = []
            reverse = []
            offset = 0
            for length in lengths:
                forward.append(encoded[offset:offset + batch_size, :length])
                offset += batch_size
            for length in lengths:
                reverse.append(encoded[offset:offset + batch_size, :length].flip(dims=[1]))
                offset += batch_size
            updated = []
            for item, fwd, rev in zip(tokens, forward, reverse):
                item = layer["norm1"](item + fwd + rev)
                feed_forward = self.dropout(self.activation(layer["conv1"](item.transpose(-1, 1))))
                feed_forward = self.dropout(layer["conv2"](feed_forward).transpose(-1, 1))
                updated.append(layer["norm2"](item + feed_forward))
            tokens = updated
        return [self.final_norm(item) for item in tokens]

    def forward(self, scale_inputs):
        if len(scale_inputs) != len(self.embeddings):
            raise ValueError("expected one input per configured scale")
        embedded = []
        num_vars = None
        for embedding, item in zip(self.embeddings, scale_inputs):
            out, item_vars = embedding(item.permute(0, 2, 1))
            if num_vars is None:
                num_vars = item_vars
            elif num_vars != item_vars:
                raise RuntimeError("multi-scale patch embeddings produced inconsistent variable counts")
            embedded.append(out)
        encoded = self._encode(embedded)
        outputs = []
        for item, head in zip(encoded, self.heads):
            item = item.reshape(-1, num_vars, item.size(1), item.size(2))
            outputs.append(head(item.permute(0, 1, 3, 2)))
        return outputs


class PersistenceChangeGate(nn.Module):
    """Encoder-only gate between persistence and a predicted change path."""

    def __init__(self, pred_len, hidden_dim, recent_ratio=0.25, dropout=0.0):
        super().__init__()
        self.pred_len = int(pred_len)
        self.recent_ratio = float(recent_ratio)
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: [B, L, N]. Every statistic is available at the forecast origin.
        _, length, _ = x.shape
        eps = torch.finfo(x.dtype).eps
        centered = x - x.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1).sqrt().clamp_min(eps)
        differences = x[:, 1:, :] - x[:, :-1, :]
        full_vol = differences.square().mean(dim=1).sqrt().clamp_min(eps)
        recent_count = max(1, int(round(max(1, differences.size(1)) * self.recent_ratio)))
        recent_vol = differences[:, -recent_count:, :].square().mean(dim=1).sqrt()
        volatility_ratio = recent_vol / full_vol

        time_index = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)
        slope = (centered * time_index.view(1, -1, 1)).sum(dim=1)
        slope = slope / time_index.square().sum().clamp_min(eps)
        normalized_slope = slope * float(length) / scale
        last_zscore = centered[:, -1, :] / scale

        horizon = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype)
        horizon = horizon / float(self.pred_len)
        base = torch.stack((volatility_ratio, normalized_slope, last_zscore), dim=-1)
        base = base.unsqueeze(2).expand(-1, -1, self.pred_len, -1)
        horizon = horizon.view(1, 1, -1, 1).expand(x.size(0), x.size(2), -1, -1)
        features = torch.cat((base, horizon), dim=-1)
        return torch.sigmoid(self.network(features).squeeze(-1))


class ChannelScaleGate(nn.Module):
    """Cheap scalar channel gate whose output is broadcast over the horizon."""

    def __init__(self, pred_len, recent_ratio=0.25):
        super().__init__()
        self.pred_len = int(pred_len)
        self.recent_ratio = float(recent_ratio)
        self.network = nn.Linear(3, 1)

    def forward(self, x):
        _, length, _ = x.shape
        eps = torch.finfo(x.dtype).eps
        centered = x - x.mean(dim=1, keepdim=True)
        scale = centered.square().mean(dim=1).sqrt().clamp_min(eps)
        differences = x[:, 1:] - x[:, :-1]
        full_vol = differences.square().mean(dim=1).sqrt().clamp_min(eps)
        recent_count = max(1, int(round(max(1, differences.size(1)) * self.recent_ratio)))
        recent_vol = differences[:, -recent_count:].square().mean(dim=1).sqrt()
        time_index = torch.linspace(-1.0, 1.0, length, device=x.device, dtype=x.dtype)
        slope = (centered * time_index.view(1, -1, 1)).sum(dim=1)
        slope = slope / time_index.square().sum().clamp_min(eps)
        features = torch.stack(
            (recent_vol / full_vol, slope * float(length) / scale,
             centered[:, -1, :] / scale),
            dim=-1,
        )
        gate = torch.sigmoid(self.network(features)).squeeze(-1)
        return gate.unsqueeze(-1).expand(-1, -1, self.pred_len)


class ChannelwiseLinearForecast(nn.Module):
    """NLinear-style forecast with an independent affine map per channel."""

    def __init__(self, channels, input_len, pred_len):
        super().__init__()
        self.channels = int(channels)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.weight = nn.Parameter(torch.full(
            (self.channels, self.pred_len, self.input_len),
            1.0 / self.input_len,
        ))
        self.bias = nn.Parameter(torch.zeros(self.channels, self.pred_len))

    def forward(self, x):
        # x: [B, L, C] -> [B, H, C]
        if x.size(-1) != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, received {x.size(-1)}"
            )
        return torch.einsum("blc,cpl->bpc", x, self.weight) + self.bias.t().unsqueeze(0)


class MultiScaleChannelwiseLinearForecast(nn.Module):
    """One channelwise affine projection over concatenated temporal scales."""

    def __init__(self, channels, input_lengths, pred_len):
        super().__init__()
        self.channels = int(channels)
        self.input_lengths = [int(length) for length in input_lengths]
        self.total_length = sum(self.input_lengths)
        self.pred_len = int(pred_len)
        self.weight = nn.Parameter(torch.full(
            (self.channels, self.pred_len, self.total_length),
            1.0 / self.total_length,
        ))
        self.bias = nn.Parameter(torch.zeros(self.channels, self.pred_len))

    def forward(self, scale_inputs):
        if len(scale_inputs) != len(self.input_lengths):
            raise ValueError("expected one input for every configured temporal scale")
        if any(item.size(-1) != self.channels for item in scale_inputs):
            raise ValueError("all multiscale inputs must have the configured channel count")
        if [item.size(1) for item in scale_inputs] != self.input_lengths:
            raise ValueError("multiscale input lengths differ from the configured lengths")
        x = torch.cat(scale_inputs, dim=1)
        return torch.einsum("blc,cpl->bpc", x, self.weight) + self.bias.t().unsqueeze(0)


class Model(nn.Module):
    """
    Patch multi-scale Mamba forecaster.

    This model targets the gap with PatchTST by making Mamba operate over
    temporal patch sequences instead of only over inverted variable tokens.
    It keeps the GLD-friendly reversible normalization and can add the same
    trend/seasonal/residual component split used by Multi-Scale Decomposition Mamba.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = bool(configs.use_norm)
        self.residual_target = bool(getattr(configs, "residual_target", 0))
        requested_head = str(getattr(configs, "forecast_head", "auto")).strip().lower()
        self.forecast_head = (
            "residual" if requested_head == "auto" and self.residual_target
            else "level" if requested_head == "auto"
            else requested_head
        )
        if self.forecast_head not in ("level", "residual", "cumulative", "persistence_change"):
            raise ValueError(
                "forecast_head must be one of: auto, level, residual, cumulative, persistence_change"
            )
        self.increment_loss_weight = float(getattr(configs, "increment_loss_weight", 0.1))
        self.change_gate_regularization = float(
            getattr(configs, "change_gate_regularization", 0.01)
        )
        self.pms_change_scale = float(getattr(configs, "pms_change_scale", 1.0))
        self.pms_branch_mode = str(getattr(configs, "pms_branch_mode", "multiscale"))
        if self.pms_branch_mode not in (
            "multiscale", "shared_multiscale", "dual_resolution", "fast_single", "temporal_single", "regime_linear", "inverted_single", "fused_dual"
        ):
            raise ValueError(
                "pms_branch_mode must be multiscale, shared_multiscale, dual_resolution, fast_single, temporal_single, regime_linear, inverted_single, or fused_dual"
            )
        self.ms_use_decomp = bool(getattr(configs, "ms_use_decomp", 1))
        self.ms_fusion = getattr(configs, "ms_fusion", "gate")
        self.ms_scales = self._parse_scales(getattr(configs, "ms_scales", [1, 2, 4]))
        self.component_count = 3 if self.ms_use_decomp else 1

        if self.ms_fusion not in ("gate", "learned", "mean"):
            raise ValueError("ms_fusion must be one of: gate, learned, mean")

        self.decomp = SeriesDecompResidual(configs.moving_avg) if self.ms_use_decomp else None
        if getattr(self, "pms_branch_mode", "multiscale") == "regime_linear":
            self.component_count = 1
            self.decomp = None
            self.branch_scales = []
            self.branches = nn.ModuleList()
            self.regime_input = nn.Linear(4, configs.d_model)
            self.regime_mamba = Mamba(
                d_model=configs.d_model,
                d_state=configs.d_state,
                d_conv=2,
                expand=1,
            )
            self.regime_norm = nn.LayerNorm(configs.d_model)
            self.regime_head = nn.Linear(configs.d_model, configs.pred_len)
            self.regime_gate_bias = float(getattr(configs, "pms_regime_gate_bias", 2.0))
            self.regime_gate_stride = max(
                1, int(getattr(configs, "pms_regime_gate_stride", 1))
            )
            self.regime_channel_gate = (
                ChannelScaleGate(
                    pred_len=configs.pred_len,
                    recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                )
                if bool(getattr(configs, "pms_regime_channel_gate", 1))
                else None
            )
            self.regime_linear = nn.Linear(configs.seq_len, configs.pred_len)
            self.regime_linear.weight = nn.Parameter(
                (1.0 / configs.seq_len)
                * torch.ones([configs.pred_len, configs.seq_len])
            )
            self.regime_individual_linear = bool(
                getattr(configs, "pms_regime_individual_linear", 0)
            )
            self.regime_channel_linear = (
                ChannelwiseLinearForecast(
                    channels=getattr(configs, "enc_in", 1),
                    input_len=configs.seq_len,
                    pred_len=configs.pred_len,
                )
                if self.regime_individual_linear
                else None
            )
            self.regime_multiscale_linear = bool(
                getattr(configs, "pms_regime_multiscale_linear", 0)
            )
            self.regime_linear_scales = (
                self._parse_scales(getattr(configs, "pms_regime_linear_scales", [1, 2, 4]))
                if self.regime_multiscale_linear
                else [1]
            )
            if self.regime_multiscale_linear and not self.regime_individual_linear:
                raise ValueError(
                    "pms_regime_multiscale_linear requires pms_regime_individual_linear=1"
                )
            self.regime_scale_linear = (
                MultiScaleChannelwiseLinearForecast(
                    channels=getattr(configs, "enc_in", 1),
                    input_lengths=[
                        self._scaled_length(configs.seq_len, scale)
                        for scale in self.regime_linear_scales
                    ],
                    pred_len=configs.pred_len,
                )
                if self.regime_multiscale_linear
                else None
            )
            self.regime_use_decomp = bool(getattr(configs, "pms_regime_decomp", 0))
            if self.regime_use_decomp:
                self.regime_decomp = SeriesDecomp(configs.moving_avg)
                self.regime_trend_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self.regime_seasonal_linear = nn.Linear(configs.seq_len, configs.pred_len)
                for linear in (self.regime_trend_linear, self.regime_seasonal_linear):
                    linear.weight = nn.Parameter(
                        (1.0 / configs.seq_len)
                        * torch.ones([configs.pred_len, configs.seq_len])
                    )
            else:
                self.regime_decomp = None
                self.regime_trend_linear = None
                self.regime_seasonal_linear = None
            self.scale_gate = None
            self.register_parameter("scale_logits", None)
            self.change_gate = None
            self._last_change_gate = None
            self._last_regime_gate = None
            return
        if self.pms_branch_mode == "inverted_single":
            self.inverted_use_decomp = bool(getattr(configs, "pms_inverted_decomp", 0))
            self.component_count = 3 if self.inverted_use_decomp else 1
            self.decomp = (
                SeriesDecompResidual(configs.moving_avg)
                if self.inverted_use_decomp else None
            )
            self.branch_scales = []
            self.branches = nn.ModuleList()
            self.inverted_branch = InvertedMambaBranch(configs)
            self.scale_gate = None
            self.register_parameter("scale_logits", None)
            self.change_gate = None
            self._last_change_gate = None
            return
        if self.pms_branch_mode == "fused_dual":
            self.branch_scales = [1, 1]
            self.fused_dual = FusedDualPatchMamba(
                configs,
                int(getattr(configs, "pms_fine_patch_len", 8)),
                int(getattr(configs, "pms_fine_stride", 4)),
                int(getattr(configs, "pms_coarse_patch_len", 32)),
                int(getattr(configs, "pms_coarse_stride", 16)),
            )
            self.branches = nn.ModuleList()
        elif self.pms_branch_mode == "shared_multiscale":
            self.branch_scales = list(self.ms_scales)
            self.shared_multiscale = SharedMultiScalePatchMamba(
                configs,
                [self._scaled_length(configs.seq_len, scale) for scale in self.ms_scales],
            )
            self.branches = nn.ModuleList()
        else:
            self.fused_dual = None
        if self.pms_branch_mode == "temporal_single":
            self.branch_scales = [1]
            self.branches = nn.ModuleList([TemporalMambaBranch(configs)])
            self.fast_linear_skip = bool(getattr(configs, "pms_fast_linear_skip", 0))
            if self.fast_linear_skip:
                self.fast_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self.fast_linear_logit = nn.Parameter(torch.zeros(()))
            else:
                self.fast_linear = None
                self.register_parameter("fast_linear_logit", None)
        elif self.pms_branch_mode == "fast_single":
            self.branch_scales = [1]
            self.branches = nn.ModuleList([
                FastPatchMambaBranch(
                    configs,
                    configs.seq_len,
                    patch_len=int(getattr(configs, "pms_fast_patch_len", 16)),
                    stride=int(getattr(configs, "pms_fast_stride", 8)),
                )
            ])
            self.fast_linear_skip = bool(getattr(configs, "pms_fast_linear_skip", 0))
            if self.fast_linear_skip:
                self.fast_linear = nn.Linear(configs.seq_len, configs.pred_len)
                self.fast_linear_logit = nn.Parameter(torch.zeros(()))
            else:
                self.fast_linear = None
                self.register_parameter("fast_linear_logit", None)
        elif self.pms_branch_mode == "dual_resolution":
            branch_specs = (
                (
                    int(getattr(configs, "pms_fine_patch_len", 8)),
                    int(getattr(configs, "pms_fine_stride", 4)),
                ),
                (
                    int(getattr(configs, "pms_coarse_patch_len", 32)),
                    int(getattr(configs, "pms_coarse_stride", 16)),
                ),
            )
            self.branch_scales = [1, 1]
            self.branches = nn.ModuleList(
                [
                    PatchMambaBranch(
                        configs,
                        configs.seq_len,
                        patch_len=patch_len,
                        stride=stride,
                    )
                    for patch_len, stride in branch_specs
                ]
            )
            self.fast_linear_skip = False
            self.fast_linear = None
            self.register_parameter("fast_linear_logit", None)
        elif self.pms_branch_mode not in ("fused_dual", "shared_multiscale"):
            self.branch_scales = list(self.ms_scales)
            self.branches = nn.ModuleList(
                [
                    PatchMambaBranch(configs, self._scaled_length(configs.seq_len, scale))
                    for scale in self.ms_scales
                ]
            )
            self.fast_linear_skip = False
            self.fast_linear = None
            self.register_parameter("fast_linear_logit", None)

        self.change_gate = (
            PersistenceChangeGate(
                pred_len=configs.pred_len,
                hidden_dim=max(8, configs.d_model // 2),
                recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                dropout=configs.dropout,
            )
            if self.forecast_head == "persistence_change"
            else None
        )
        self._last_change_gate = None

        fusion_branches = (
            2 if self.pms_branch_mode == "fused_dual"
            else len(self.ms_scales) if self.pms_branch_mode == "shared_multiscale"
            else len(self.branches)
        )

        if self.ms_fusion == "gate":
            self.scale_gate = NonStationaryGate(
                d_model=configs.d_model,
                hidden_dim=2 * configs.d_model,
                gate_dim=configs.pred_len,
                num_gates=fusion_branches,
                recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                dropout=configs.dropout,
                activation="softmax",
            )
            self.scale_logits = None
        elif self.ms_fusion == "learned":
            self.scale_gate = None
            self.scale_logits = nn.Parameter(torch.zeros(fusion_branches))
        else:
            self.scale_gate = None
            self.register_parameter("scale_logits", None)

    def _parse_scales(self, scales):
        if isinstance(scales, int):
            scales = [scales]
        parsed = []
        for scale in scales:
            scale = int(scale)
            if scale <= 0:
                raise ValueError("all ms_scales must be positive integers")
            if scale not in parsed:
                parsed.append(scale)
        return parsed

    def _scaled_length(self, length, scale):
        return max(1, int(math.ceil(length / scale)))

    def _standard_normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return centered / stdev, means, stdev

    def _standard_denormalize(self, x, means, stdev):
        x = x * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return x + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

    def _downsample(self, x, scale):
        if scale == 1:
            return x
        if scale >= x.size(1):
            return x.mean(dim=1, keepdim=True)
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, kernel_size=scale, stride=scale, ceil_mode=True)
        return x.permute(0, 2, 1)

    def _component_tokens(self, x):
        if self.decomp is None:
            return x
        seasonal, trend, residual = self.decomp(x)
        return torch.cat([trend, seasonal, residual], dim=-1)

    def _branch_forecast(self, branch, x, num_vars):
        return self._branch_forecast_from_output(branch(x), num_vars)

    def _branch_forecast_from_output(self, out, num_vars):
        if self.component_count == 1:
            return out[:, :num_vars, :]

        out = out.reshape(out.size(0), self.component_count, num_vars, self.pred_len)
        return out.sum(dim=1)

    def _fuse_scales(self, scale_outputs, raw_x, num_vars):
        stack = torch.stack(scale_outputs, dim=2)
        if self.scale_gate is not None:
            weights = self.scale_gate(raw_x, target_tokens=num_vars)
            return (weights * stack).sum(dim=2)
        if self.scale_logits is not None:
            weights = torch.softmax(self.scale_logits, dim=0).view(1, 1, -1, 1)
            return (weights * stack).sum(dim=2)
        return stack.mean(dim=2)

    def _scale_delta(self, value, stdev):
        return value * stdev[:, 0, :].unsqueeze(1)

    def _apply_forecast_head(self, encoded_path, raw_x, means, stdev):
        forecast_head = getattr(
            self,
            "forecast_head",
            "residual" if getattr(self, "residual_target", False) else "level",
        )
        last = raw_x[:, -1:, :]
        if forecast_head == "level":
            if self.use_norm:
                return self._standard_denormalize(encoded_path, means, stdev)
            return encoded_path

        change = encoded_path
        if forecast_head == "cumulative":
            change = torch.cumsum(change, dim=1)
        elif forecast_head == "persistence_change":
            gate = self.change_gate(raw_x).permute(0, 2, 1)
            self._last_change_gate = gate
            change = self.pms_change_scale * gate * change

        if self.use_norm:
            change = self._scale_delta(change, stdev)
        return last + change

    def auxiliary_loss(self, prediction, target, batch_x):
        loss = prediction.new_zeros(())
        forecast_head = getattr(
            self,
            "forecast_head",
            "residual" if getattr(self, "residual_target", False) else "level",
        )
        target_count = prediction.size(-1)
        last = batch_x[:, -1:, -target_count:]
        if forecast_head == "cumulative" and self.increment_loss_weight > 0:
            predicted_previous = torch.cat((last, prediction[:, :-1, :]), dim=1)
            target_previous = torch.cat((last, target[:, :-1, :]), dim=1)
            loss = loss + self.increment_loss_weight * F.mse_loss(
                prediction - predicted_previous,
                target - target_previous,
            )
        if (
            forecast_head == "persistence_change"
            and self.change_gate_regularization > 0
            and self._last_change_gate is not None
        ):
            horizon = torch.arange(
                1,
                self.pred_len + 1,
                device=prediction.device,
                dtype=prediction.dtype,
            )
            prior = torch.sqrt(horizon / float(self.pred_len)).view(1, -1, 1)
            loss = loss + self.change_gate_regularization * F.mse_loss(
                self._last_change_gate,
                prior.expand_as(self._last_change_gate),
            )
        return loss

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raw_x = x_enc
        _, _, num_vars = x_enc.shape
        if getattr(self, "pms_branch_mode", "multiscale") == "regime_linear":
            return self._regime_linear_forecast(raw_x)
        if getattr(self, "pms_branch_mode", "multiscale") == "inverted_single":
            return self._inverted_single_forecast(raw_x)
        means = None
        stdev = None

        if self.use_norm:
            x_enc, means, stdev = self._standard_normalize(x_enc)

        x_tokens = self._component_tokens(x_enc)
        scale_outputs = []
        if getattr(self, "fused_dual", None) is not None:
            scale_outputs.extend(self._branch_forecast_from_output(output, num_vars) for output in self.fused_dual(x_tokens))
        elif getattr(self, "pms_branch_mode", "multiscale") == "shared_multiscale":
            scale_inputs = [
                self._downsample(x_tokens, scale)
                for scale in self.branch_scales
            ]
            scale_outputs.extend(
                self._branch_forecast_from_output(output, num_vars)
                for output in self.shared_multiscale(scale_inputs)
            )
        else:
            branch_scales = getattr(self, "branch_scales", self.ms_scales)
            for scale, branch in zip(branch_scales, self.branches):
                scale_x = self._downsample(x_tokens, scale)
                scale_outputs.append(self._branch_forecast(branch, scale_x, num_vars))

        encoded_path = self._fuse_scales(scale_outputs, raw_x, num_vars).permute(0, 2, 1)
        fast_linear = getattr(self, "fast_linear", None)
        if fast_linear is not None:
            linear_path = fast_linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
            linear_change = linear_path - x_enc[:, -1:, :]
            linear_weight = torch.sigmoid(self.fast_linear_logit)
            encoded_path = (
                linear_weight * linear_change + (1.0 - linear_weight) * encoded_path
            )
        self._last_change_gate = None
        dec_out = self._apply_forecast_head(encoded_path, raw_x, means, stdev)

        return dec_out, None

    def _regime_linear_forecast(self, raw_x):
        means = None
        stdev = None
        x_enc = raw_x
        if self.use_norm:
            x_enc, means, stdev = self._standard_normalize(raw_x)

        differences = torch.zeros_like(x_enc)
        if x_enc.size(1) > 1:
            differences[:, 1:] = x_enc[:, 1:] - x_enc[:, :-1]
        features = torch.stack(
            (
                x_enc.mean(dim=-1),
                x_enc.std(dim=-1, unbiased=False),
                differences.mean(dim=-1),
                differences.std(dim=-1, unbiased=False),
            ),
            dim=-1,
        )
        if self.regime_gate_stride > 1:
            features = F.avg_pool1d(
                features.transpose(1, 2),
                kernel_size=self.regime_gate_stride,
                stride=self.regime_gate_stride,
                ceil_mode=True,
            ).transpose(1, 2)
        encoded = self.regime_input(features)
        encoded = self.regime_norm(encoded + self.regime_mamba(encoded))
        gate_logits = self.regime_head(encoded[:, -1, :]) + self.regime_gate_bias
        gate = torch.sigmoid(gate_logits).unsqueeze(-1)
        if self.regime_channel_gate is not None:
            channel_gate = self.regime_channel_gate(raw_x).permute(0, 2, 1)
            gate = gate * (0.5 + channel_gate)
        self._last_regime_gate = gate

        seq_last = x_enc[:, -1:, :]
        if self.regime_decomp is None:
            centered = x_enc - seq_last
            if self.regime_scale_linear is not None:
                linear_delta = self.regime_scale_linear([
                    self._downsample(centered, scale)
                    for scale in self.regime_linear_scales
                ])
            elif self.regime_channel_linear is None:
                linear_delta = self.regime_linear(centered.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                linear_delta = self.regime_channel_linear(centered)
        else:
            seasonal, trend = self.regime_decomp(x_enc)
            trend_last = trend[:, -1:, :]
            trend_future = trend_last + self.regime_trend_linear(
                (trend - trend_last).permute(0, 2, 1)
            ).permute(0, 2, 1)
            seasonal_future = self.regime_seasonal_linear(
                seasonal.permute(0, 2, 1)
            ).permute(0, 2, 1)
            linear_delta = trend_future + seasonal_future - seq_last
        change = gate * linear_delta
        if self.use_norm:
            change = self._scale_delta(change, stdev)
        return raw_x[:, -1:, :] + change, None

    def _inverted_single_forecast(self, raw_x):
        x_enc = raw_x
        means = None
        stdev = None
        if self.use_norm:
            x_enc, means, stdev = self._standard_normalize(raw_x)
        if self.decomp is None:
            token_input = x_enc
            component_count = 1
        else:
            seasonal, trend, residual = self.decomp(x_enc)
            token_input = torch.cat([trend, seasonal, residual], dim=-1)
            component_count = 3
        encoded = self.inverted_branch(token_input)
        if component_count == 1:
            encoded_path = encoded.permute(0, 2, 1)
        else:
            encoded = encoded.reshape(
                encoded.size(0), component_count, raw_x.size(-1), self.pred_len
            )
            encoded_path = encoded.sum(dim=1).permute(0, 2, 1)
        if self.forecast_head == "level":
            if self.use_norm:
                return self._standard_denormalize(encoded_path, means, stdev), None
            return encoded_path, None
        if self.use_norm:
            encoded_path = self._scale_delta(encoded_path, stdev)
        return raw_x[:, -1:, :] + encoded_path, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.output_attention:
            return dec_out, attns
        return dec_out
