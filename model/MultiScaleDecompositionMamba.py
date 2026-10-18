import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.NonStationaryGate import NonStationaryGate
from layers.SeriesDecomp import SeriesDecompResidual
from mamba_ssm import Mamba


class _ScaleMambaBranch(nn.Module):
    def __init__(self, configs, input_len):
        super(_ScaleMambaBranch, self).__init__()
        self.embedding = DataEmbedding_inverted(
            input_len,
            configs.d_model,
            configs.embed,
            configs.freq,
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
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x, x_mark):
        enc_out = self.embedding(x, x_mark)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        return self.projector(enc_out)


class Model(nn.Module):
    """
    Multi-scale Mamba forecaster.

    The model keeps the inverted-token Inverted Mamba interface, but creates several
    temporal resolutions before embedding. Each scale is encoded by an
    independent bidirectional Mamba encoder. Scale forecasts are fused by a
    shift-aware gate, learned weights, or a simple mean.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = bool(configs.use_norm)
        self.enc_in = configs.enc_in
        self.ms_use_decomp = bool(getattr(configs, "ms_use_decomp", 1))
        self.ms_use_linear_skip = bool(getattr(configs, "ms_use_linear_skip", 0))
        self.ms_fusion = getattr(configs, "ms_fusion", "gate")
        self.ms_scales = self._parse_scales(getattr(configs, "ms_scales", [1, 2, 4]))

        if self.ms_fusion not in ("gate", "learned", "mean"):
            raise ValueError("ms_fusion must be one of: gate, learned, mean")

        self.component_count = 3 if self.ms_use_decomp else 1
        self.decomp = SeriesDecompResidual(configs.moving_avg) if self.ms_use_decomp else None

        self.branches = nn.ModuleList(
            [
                _ScaleMambaBranch(configs, self._scaled_length(configs.seq_len, scale))
                for scale in self.ms_scales
            ]
        )

        if self.ms_fusion == "gate":
            self.scale_gate = NonStationaryGate(
                d_model=configs.d_model,
                hidden_dim=2 * configs.d_model,
                gate_dim=configs.pred_len,
                num_gates=len(self.ms_scales),
                recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                dropout=configs.dropout,
                activation="softmax",
            )
            self.scale_logits = None
        elif self.ms_fusion == "learned":
            self.scale_gate = None
            self.scale_logits = nn.Parameter(torch.zeros(len(self.ms_scales)))
        else:
            self.scale_gate = None
            self.register_parameter("scale_logits", None)

        if self.ms_use_linear_skip:
            self.linear_skip = nn.Linear(configs.seq_len, configs.pred_len)
            self.linear_skip.weight = nn.Parameter(
                (1 / configs.seq_len) * torch.ones([configs.pred_len, configs.seq_len])
            )
            self.linear_skip_mix = nn.Parameter(torch.tensor(0.0))
        else:
            self.linear_skip = None
            self.register_parameter("linear_skip_mix", None)

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

    def _branch_forecast(self, branch, x, x_mark, num_vars):
        token_count = num_vars * self.component_count
        out = branch(x, x_mark)[:, :token_count, :]
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

    def _linear_skip_forecast(self, x):
        last = x[:, -1:, :].detach()
        residual = (x - last).permute(0, 2, 1)
        skip = self.linear_skip(residual)
        return skip + last[:, 0, :].unsqueeze(-1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raw_x = x_enc
        _, _, num_vars = x_enc.shape
        means = None
        stdev = None

        if self.use_norm:
            x_enc, means, stdev = self._standard_normalize(x_enc)

        x_tokens = self._component_tokens(x_enc)
        scale_outputs = []
        for scale, branch in zip(self.ms_scales, self.branches):
            scale_x = self._downsample(x_tokens, scale)
            scale_mark = None if x_mark_enc is None else self._downsample(x_mark_enc, scale)
            scale_outputs.append(self._branch_forecast(branch, scale_x, scale_mark, num_vars))

        dec_tokens = self._fuse_scales(scale_outputs, raw_x, num_vars)
        if self.linear_skip is not None:
            skip_tokens = self._linear_skip_forecast(x_enc)
            mix = torch.sigmoid(self.linear_skip_mix)
            dec_tokens = mix * dec_tokens + (1.0 - mix) * skip_tokens

        dec_out = dec_tokens.permute(0, 2, 1)

        if self.use_norm:
            dec_out = self._standard_denormalize(dec_out, means, stdev)

        return dec_out, None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.output_attention:
            return dec_out, attns
        return dec_out
