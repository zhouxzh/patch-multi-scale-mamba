import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.NonStationaryGate import NonStationaryGate
from layers.SeriesDecomp import SeriesDecompResidual
from mamba_ssm import Mamba


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


class Model(nn.Module):
    """
    Patch multi-scale Mamba forecaster.

    This model targets the gap with PatchTST by making Mamba operate over
    temporal patch sequences instead of only over inverted variable tokens.
    It keeps the GLD-friendly reversible normalization and can add the same
    trend/seasonal/residual component split used by MS_Mamba.
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
        self.pms_branch_mode = str(getattr(configs, "pms_branch_mode", "multiscale"))
        if self.pms_branch_mode not in ("multiscale", "dual_resolution"):
            raise ValueError("pms_branch_mode must be multiscale or dual_resolution")
        self.ms_use_decomp = bool(getattr(configs, "ms_use_decomp", 1))
        self.ms_fusion = getattr(configs, "ms_fusion", "gate")
        self.ms_scales = self._parse_scales(getattr(configs, "ms_scales", [1, 2, 4]))
        self.component_count = 3 if self.ms_use_decomp else 1

        if self.ms_fusion not in ("gate", "learned", "mean"):
            raise ValueError("ms_fusion must be one of: gate, learned, mean")

        self.decomp = SeriesDecompResidual(configs.moving_avg) if self.ms_use_decomp else None
        if self.pms_branch_mode == "dual_resolution":
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
        else:
            self.branch_scales = list(self.ms_scales)
            self.branches = nn.ModuleList(
                [
                    PatchMambaBranch(configs, self._scaled_length(configs.seq_len, scale))
                    for scale in self.ms_scales
                ]
            )

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

        if self.ms_fusion == "gate":
            self.scale_gate = NonStationaryGate(
                d_model=configs.d_model,
                hidden_dim=2 * configs.d_model,
                gate_dim=configs.pred_len,
                num_gates=len(self.branches),
                recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                dropout=configs.dropout,
                activation="softmax",
            )
            self.scale_logits = None
        elif self.ms_fusion == "learned":
            self.scale_gate = None
            self.scale_logits = nn.Parameter(torch.zeros(len(self.branches)))
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
        out = branch(x)
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
            change = gate * change

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
        means = None
        stdev = None

        if self.use_norm:
            x_enc, means, stdev = self._standard_normalize(x_enc)

        x_tokens = self._component_tokens(x_enc)
        scale_outputs = []
        branch_scales = getattr(self, "branch_scales", self.ms_scales)
        for scale, branch in zip(branch_scales, self.branches):
            scale_x = self._downsample(x_tokens, scale)
            scale_outputs.append(self._branch_forecast(branch, scale_x, num_vars))

        encoded_path = self._fuse_scales(scale_outputs, raw_x, num_vars).permute(0, 2, 1)
        self._last_change_gate = None
        dec_out = self._apply_forecast_head(encoded_path, raw_x, means, stdev)

        return dec_out, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.output_attention:
            return dec_out, attns
        return dec_out
