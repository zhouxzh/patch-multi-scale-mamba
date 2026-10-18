import os

import torch
from model import DLinear, NLinear, PatchTST, TimesNet, iTransformer

try:
    from model import (
        DecompositionMamba,
        InvertedMamba,
        MultiScaleDecompositionMamba,
        MultiScalePatchMamba,
        RegimeGatedMamba,
    )
except (ImportError, RuntimeError):
    DecompositionMamba = None
    InvertedMamba = None
    MultiScaleDecompositionMamba = None
    MultiScalePatchMamba = None
    RegimeGatedMamba = None


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.writer = None
        self.model_dict = {
            'iTransformer': iTransformer,
            'InvertedMamba': InvertedMamba,
            'DecompositionMamba': DecompositionMamba,
            'MultiScaleDecompositionMamba': MultiScaleDecompositionMamba,
            'MultiScalePatchMamba': MultiScalePatchMamba,
            'RegimeGatedMamba': RegimeGatedMamba,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'PatchTST': PatchTST,
            'TimesNet': TimesNet,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def _get_writer(self, setting):
        if getattr(self.args, 'disable_tensorboard', False):
            self.writer = None
            return None
        if self.writer is not None:
            return self.writer
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print('TensorBoard is not available in the current environment.')
            self.writer = None
            return None

        log_dir = os.path.join(self.args.tensorboard_dir, setting)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        return self.writer
