import torch
import torch.nn as nn


class Model(nn.Module):
    """
    NLinear baseline adapted for this repo.
    Paper: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = getattr(configs, "task_name", "long_term_forecast")
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.individual = bool(getattr(configs, "individual", False))
        self.channels = configs.enc_in

        if self.individual:
            self.Linear = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
            for linear in self.Linear:
                linear.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Linear.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forecast(self, x_enc):
        seq_last = x_enc[:, -1:, :].detach()
        x = (x_enc - seq_last).permute(0, 2, 1)

        if self.individual:
            output = torch.zeros(
                [x.size(0), x.size(1), self.pred_len],
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.channels):
                output[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            output = self.Linear(x)

        return output.permute(0, 2, 1) + seq_last

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)[:, -self.pred_len:, :]
        if self.output_attention:
            return dec_out, None
        return dec_out
