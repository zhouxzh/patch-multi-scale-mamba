import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
import json

if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Nonstationary time-series forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='RegimeGatedMamba',
                        help='model name: RegimeGatedMamba, MultiScalePatchMamba, InvertedMamba, DecompositionMamba, MultiScaleDecompositionMamba, DLinear, NLinear, PatchTST, TimesNet, or iTransformer')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name for imported TSLib-style models')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results_root', type=str, default='./results/', help='location of saved result artifacts')
    parser.add_argument('--split_config', type=str, default='', help='JSON file containing explicit chronological split dates')
    parser.add_argument('--evaluation_split', choices=['val', 'test'], default='test',
                        help='split evaluated after training; use val for smoke tests and selection')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--individual', action='store_true', default=False,
                        help='use individual linear layers per variable for DLinear/NLinear')
    parser.add_argument('--top_k', type=int, default=5, help='top periods for TimesNet')
    parser.add_argument('--num_kernels', type=int, default=6, help='number of Inception kernels for TimesNet')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length for PatchTST')
    parser.add_argument('--stride', type=int, default=8, help='patch stride for PatchTST')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--seed', type=int, default=2023, help='training random seed')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--horizon_weighting', type=str, choices=['balanced', 'long'], default='balanced',
                        help='training-step weighting: uniform or increasing weight for later forecast steps')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard/', help='tensorboard log directory')
    parser.add_argument('--thermal_guard_temp', type=int, default=0,
                        help='pause between train/validation phases at or above this GPU temperature; 0 disables')
    parser.add_argument('--thermal_resume_temp', type=int, default=75,
                        help='resume after a thermal pause at or below this GPU temperature')
    parser.add_argument('--thermal_poll_seconds', type=float, default=5.0,
                        help='temperature polling interval during a thermal pause')
    parser.add_argument('--thermal_check_batches', type=int, default=0,
                        help='check the thermal guard every N batches; 0 disables batch checks')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--require_cuda', action='store_true',
                        help='abort instead of falling back to CPU when CUDA is unavailable')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--inverse', dest='inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--no-inverse', dest='inverse', action='store_false',
                        help='keep output data in scaled space')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions for Nonstationary Transformer projectors')
    parser.add_argument('--p_hidden_layers', type=int, default=2,
                        help='number of hidden layers for Nonstationary Transformer projectors')
    parser.add_argument('--use_adaptive_norm', type=int, default=1,
                        help='use AdaptiveRevIN in NS_Mamba')
    parser.add_argument('--use_decomp', type=int, default=0,
                        help='use trend/seasonal/residual decomposition in NS_Mamba')
    parser.add_argument('--use_shift_gate', type=int, default=0,
                        help='use distribution-shift-aware gating in NS_Mamba')
    parser.add_argument('--ns_adaptive_scale', type=float, default=0.1,
                        help='adaptive affine scale for NS_Mamba AdaptiveRevIN')
    parser.add_argument('--ns_recent_ratio', type=float, default=0.25,
                        help='recent-window ratio for NS_Mamba shift statistics')
    parser.add_argument('--ms_scales', type=int, nargs='+', default=[1, 2, 4],
                        help='temporal downsampling scales for MS_Mamba')
    parser.add_argument('--ms_fusion', type=str, default='gate',
                        help='scale fusion for MS_Mamba: gate, learned, or mean')
    parser.add_argument('--ms_use_decomp', type=int, default=1,
                        help='use trend/seasonal/residual component tokens in MS_Mamba')
    parser.add_argument('--ms_use_linear_skip', type=int, default=0,
                        help='blend an NLinear-style last-value skip forecast into MS_Mamba')
    parser.add_argument('--residual_target', type=int, default=0,
                        help='anchor PMS-Mamba forecasts around the last observed value')
    parser.add_argument('--forecast_head', type=str, default='auto',
                        choices=['auto', 'level', 'residual', 'cumulative', 'persistence_change'],
                        help='PMS-Mamba output parameterization; auto preserves legacy residual_target behavior')
    parser.add_argument('--increment_loss_weight', type=float, default=0.1,
                        help='auxiliary increment-MSE coefficient for the cumulative PMS head')
    parser.add_argument('--change_gate_regularization', type=float, default=0.01,
                        help='horizon-prior regularization for the PMS persistence-change gate')
    parser.add_argument('--pms_change_scale', type=float, default=1.0,
                        help='fixed multiplier for the PMS predicted change path')
    parser.add_argument('--pms_branch_mode', type=str, default='regime_linear',
                        choices=['multiscale', 'regime_linear'],
                        help='selected Regime-Gated Mamba or Multi-Scale Patch Mamba branch layout')
    parser.add_argument('--pms_fine_patch_len', type=int, default=8)
    parser.add_argument('--pms_fine_stride', type=int, default=4)
    parser.add_argument('--pms_coarse_patch_len', type=int, default=32)
    parser.add_argument('--pms_coarse_stride', type=int, default=16)
    parser.add_argument('--pms_fast_patch_len', type=int, default=16)
    parser.add_argument('--pms_fast_stride', type=int, default=8)
    parser.add_argument('--pms_fast_linear_skip', type=int, default=0,
                        help='blend a last-value-anchored linear change path into fast PMS-Mamba')
    parser.add_argument('--pms_ssm_backend', type=str, default='mamba', choices=['mamba', 'mamba2'],
                        help='selective state-space backend for opt-in one-scan PMS branches')
    parser.add_argument('--pms_regime_channel_gate', type=int, default=1,
                        help='use an encoder-only per-channel persistence gate in regime_linear mode')
    parser.add_argument('--pms_regime_gate_bias', type=float, default=2.0,
                        help='initial logit bias for the regime_linear change gate')
    parser.add_argument('--pms_regime_gate_stride', type=int, default=1,
                        help='temporal pooling stride before the regime Mamba gate')
    parser.add_argument('--pms_regime_decomp', type=int, default=0,
                        help='use trend/seasonal linear paths in regime_linear mode')
    parser.add_argument('--pms_regime_individual_linear', type=int, default=0,
                        help='use an independent NLinear-style forecast map for each input channel')
    parser.add_argument('--pms_regime_multiscale_linear', type=int, default=0,
                        help='fuse channelwise raw/coarse linear paths with one Mamba-derived horizon gate')
    parser.add_argument('--pms_regime_linear_scales', type=int, nargs='+', default=[1, 2, 4],
                        help='temporal scales for the regime multiscale linear head')
    parser.add_argument('--pms_inverted_decomp', type=int, default=0,
                        help='use trend/seasonal/residual variable tokens in inverted_single mode')
    parser.add_argument('--pms_min_patch_tokens', type=int, default=1,
                        help='minimum patch-token count accepted by V3 parameter validation')
    parser.add_argument('--fixed_train_epochs', action='store_true',
                        help='train for exactly train_epochs without reading validation data')

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.split_config:
        with open(args.split_config) as split_handle:
            split_payload = json.load(split_handle)
        args.split_dates = split_payload.get('splits', split_payload)
    else:
        args.split_dates = None
    args.channel_independence = False
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.require_cuda and not args.use_gpu:
        raise SystemExit('CUDA is required for this run but is unavailable')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')

    if args.is_training==1:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>evaluating {} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
                args.evaluation_split, setting))
            if args.evaluation_split == 'test':
                exp.test(setting)
            else:
                exp.evaluate(
                    setting=setting,
                    save_artifacts=True,
                    save_visuals=False,
                    write_result_log=False,
                    write_tensorboard=False,
                    flag=args.evaluation_split,
                )

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    elif args.is_training == 2:
        print(11111)
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)
            exp = Exp(args)
            exp.get_input(setting)
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
