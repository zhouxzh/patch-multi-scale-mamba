import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.NonStationaryGate import NonStationaryGate
from layers.NonStationaryNorm import AdaptiveRevIN
from layers.SeriesDecomp import SeriesDecompResidual
from mamba_ssm import Mamba


class Model(nn.Module):
    """
    Nonstationarity-aware Inverted Mamba variant.

    Ablation flags:
      - use_adaptive_norm: replace fixed mean/std normalization with AdaptiveRevIN
      - use_decomp: split input into trend, seasonal, and residual branches
      - use_shift_gate: use shift-aware hidden gating or branch fusion
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = bool(configs.use_norm)
        self.use_adaptive_norm = bool(getattr(configs, "use_adaptive_norm", 1))
        self.use_decomp = bool(getattr(configs, "use_decomp", 0))
        self.use_shift_gate = bool(getattr(configs, "use_shift_gate", 0))
        self.class_strategy = configs.class_strategy

        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        if self.use_adaptive_norm:
            self.adaptive_norm = AdaptiveRevIN(
                num_features=configs.enc_in,
                hidden_dim=configs.d_model,
                recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                adaptive_scale=getattr(configs, "ns_adaptive_scale", 0.1),
            )
        else:
            self.adaptive_norm = None

        if self.use_decomp:
            self.decomp = SeriesDecompResidual(configs.moving_avg)
            self.branch_embeddings = nn.ModuleDict(
                {
                    name: DataEmbedding_inverted(
                        configs.seq_len,
                        configs.d_model,
                        configs.embed,
                        configs.freq,
                        configs.dropout,
                    )
                    for name in ("trend", "seasonal", "residual")
                }
            )
            self.branch_encoders = nn.ModuleDict(
                {name: self._build_encoder(configs) for name in ("trend", "seasonal", "residual")}
            )
            self.branch_projectors = nn.ModuleDict(
                {
                    name: nn.Linear(configs.d_model, configs.pred_len, bias=True)
                    for name in ("trend", "seasonal", "residual")
                }
            )
            if self.use_shift_gate:
                self.branch_gate = NonStationaryGate(
                    d_model=configs.d_model,
                    hidden_dim=2 * configs.d_model,
                    gate_dim=configs.pred_len,
                    num_gates=3,
                    recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                    dropout=configs.dropout,
                    activation="softmax",
                )
            else:
                self.branch_gate = None
        else:
            self.enc_embedding = DataEmbedding_inverted(
                configs.seq_len,
                configs.d_model,
                configs.embed,
                configs.freq,
                configs.dropout,
            )
            self.encoder = self._build_encoder(configs)
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
            if self.use_shift_gate:
                self.hidden_gate = NonStationaryGate(
                    d_model=configs.d_model,
                    hidden_dim=2 * configs.d_model,
                    gate_dim=configs.d_model,
                    num_gates=1,
                    recent_ratio=getattr(configs, "ns_recent_ratio", 0.25),
                    dropout=configs.dropout,
                    activation="sigmoid",
                )
            else:
                self.hidden_gate = None

    def _build_encoder(self, configs):
        return Encoder(
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

    def _standard_normalize(self, x):
        means = x.mean(1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return centered / stdev, means, stdev

    def _standard_denormalize(self, x, means, stdev):
        x = x * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return x + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

    def _branch_forecast(self, name, x, x_mark, num_vars):
        enc_out = self.branch_embeddings[name](x, x_mark)
        enc_out, _ = self.branch_encoders[name](enc_out, attn_mask=None)
        return self.branch_projectors[name](enc_out)[:, :num_vars, :]

    def _forecast_single_branch(self, x_enc, x_mark_enc, num_vars):
        enc_in = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_in, attn_mask=None)
        if self.hidden_gate is not None:
            gate = self.hidden_gate(x_enc, hidden=enc_out)
            enc_out = gate * enc_out + (1.0 - gate) * enc_in
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :num_vars]
        return dec_out, attns

    def _forecast_decomp(self, x_enc, x_mark_enc, raw_x, num_vars):
        seasonal, trend, residual = self.decomp(x_enc)
        branch_tokens = [
            self._branch_forecast("trend", trend, x_mark_enc, num_vars),
            self._branch_forecast("seasonal", seasonal, x_mark_enc, num_vars),
            self._branch_forecast("residual", residual, x_mark_enc, num_vars),
        ]
        branch_stack = torch.stack(branch_tokens, dim=2)
        if self.branch_gate is not None:
            gate = self.branch_gate(raw_x, target_tokens=num_vars)
            branch_tokens = (gate * branch_stack).sum(dim=2)
        else:
            branch_tokens = branch_stack.sum(dim=2)
        return branch_tokens.permute(0, 2, 1), None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        raw_x = x_enc
        _, _, num_vars = x_enc.shape
        means = None
        stdev = None

        if self.use_norm:
            if self.adaptive_norm is not None:
                x_enc = self.adaptive_norm.normalize(x_enc)
            else:
                x_enc, means, stdev = self._standard_normalize(x_enc)

        if self.use_decomp:
            dec_out, attns = self._forecast_decomp(x_enc, x_mark_enc, raw_x, num_vars)
        else:
            dec_out, attns = self._forecast_single_branch(x_enc, x_mark_enc, num_vars)

        if self.use_norm:
            if self.adaptive_norm is not None:
                dec_out = self.adaptive_norm.denormalize(dec_out)
            else:
                dec_out = self._standard_denormalize(dec_out, means, stdev)

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        dec_out = dec_out[:, -self.pred_len:, :]
        if self.output_attention:
            return dec_out, attns
        return dec_out
