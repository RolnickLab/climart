"""
Author: Salva Rühling Cachay
"""

import argparse
import time

import wandb

from climart.models.GNs.constants import GLOBALS
from climart.models.interface import is_gnn, is_cnn, is_graph_net
from climart.utils.utils import set_gpu, get_name


def get_argparser(jupyter_mode=False):
    parser = argparse.ArgumentParser(description='PyTorch RT emulation')
    parser.add_argument('--resume_training_file', type=str, help='A .pkl file to resume training from')
    parser.add_argument('--resume_ID', type=str, help='A wandb run ID to resume training from')

    parser.add_argument('--exp_type', type=str, default='pristine', help='Pristine, clear-sky')
    parser.add_argument('--target_type', type=str, default='shortwave', help='Long- or short-wave')
    parser.add_argument('--target_variable', type=str, default='fluxes', help='Fluxes or Heating-rate')

    parser.add_argument('--expID', type=str, default='', help='A special ID for the experiment if so desired')
    parser.add_argument("--wandb_mode", type=str, default='disabled', help="Use 'disabled' to turn wandb off")
    parser.add_argument('--model_dir', type=str, default='./out/', help='')

    parser.add_argument('--model', type=str, default='MLP',
                        help='Which ML model to use (MLP, GCN, GN, CNN, CNNMS)')

    parser.add_argument('--gpu_id', default=0, help="Please give a value for gpu device")
    parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for loading data')
    parser.add_argument('--load_train_into_mem', action='store_true', help='Load training h5`s into RAM?')
    parser.add_argument('--load_val_into_mem', action='store_true', help='Load val h5 into RAM?')

    parser.add_argument('--test_ood_1991', action='store_true', help='Load 1991 Mt. Pinatubo OOD test data?')
    parser.add_argument('--test_ood_historic', action='store_true', help='Load & Test on Historic (1850-52) data?')
    parser.add_argument('--test_ood_future', action='store_true', help='Load & Test on Future (2097-99) data?')


    parser.add_argument('--epochs', type=int, default=100, help='')
    parser.add_argument('--additional_epochs', type=int, default=0, help='Additional epochs when resuming training.')
    parser.add_argument('--loss', default="MSE", help="Please give a value for the loss function")
    parser.add_argument('--train_on_raw_targets', action='store_true', help='Compute loss *after* denormalizing preds')
    parser.add_argument('--gradient_clipping', default="", type=str,
                        help='One of {Norm, Value} to use clipping, otherwise no clipping is performed')
    parser.add_argument('--clip', default=1.0, type=float,
                        help='How to clip the gradients. Only used when using gradient clipping')

    parser.add_argument('--act', '--activation_function', default="GeLU",
                        help="Please give a value for the activation function")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--optim', default='Adam', help="Please give a value for optimizer")
    parser.add_argument('--lr', default=2e-4, type=float, help="Please give a value for learning rate")
    parser.add_argument('--nesterov', default='True', type=str, help="Please give a value for learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight decay rate')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--shuffle', default='True', help='shuffle training batches?')

    parser.add_argument('--seed', default=7, type=int, help="Please give a value for seed")
    parser.add_argument('--hidden_dims', nargs='*', type=int, default=[256, 256, 256, 256, 256],
                        help="Hidden layer dimensions, pass multiple values.")
    parser.add_argument('--out_dim', default=100, type=int, help="Please give a value for out_dim")
    parser.add_argument('--net_norm', '--net_normalization', default='none',
                        help="batch_norm, layer_norm, inst_norm or none")

    # ------------------- Preprocessing/Projection of hetereoogenous variables (with different shapes):
    parser.add_argument('--preprocessing', type=str, default='padding', help='none, padding, duplication, MLP')
    parser.add_argument('--projector_n_layers', type=int, default=1, help='#layers of MLP, if preprocessing == MLP')
    parser.add_argument('--projector_net_normalization', type=str, default='layer_norm',
                        help='none, layer_norm, batch_norm')
    parser.add_argument('--keep_node_encoding', action='store_true',
                        help='When duplicating global feats and only layer nodes are kept')
    parser.add_argument('--drop_level_features', action='store_true', help='Ignore levels?')
    parser.add_argument('--drop_last_level', action='store_true', help='none, layer_norm, batch_norm')

    # ----------------------------------

    # ------------------- CNN Arguments
    parser.add_argument('--dilation', default=1, type=int,
                        help="The dilation rate used in CNN's to increase the receptive field size")
    parser.add_argument('--channels_list', nargs='*', type=int, default=[100, 200, 400, 100],
                        help="Increase in the number of channels of CNN, pass multiple values.")
    parser.add_argument('--gap', action='store_true', help='Use global average pooling in place of linear head in CNN')
    parser.add_argument('--use_act', action='store_true', help='Use activation in multiscale module')
    parser.add_argument('--se_block', action='store_true', help='Use Squeeze and Excitation in CNN')
    parser.add_argument('--dual_inputs', action='store_true', help='Use multiple inputs for CNN')
    # ----------------------------------

    # ------------------- GNN Arguments
    # --> Preprocessing features/edges:
    parser.add_argument('--node_encoding', action='store_false', help='GNN Node one-hot encodings for node types')
    parser.add_argument('--learn_edge_structure', action='store_true', help='Learn adjacency matrix?')
    parser.add_argument('--degree_normalized_adj', action='store_true', help='')
    parser.add_argument('--improved_self_loops', action='store_true', help='')
    parser.add_argument('--pyg_builtin', action='store_true', help='Use Pytorch geometric GCN implementation')
    # ------------------- GN Arguments
    parser.add_argument('--update_mlp_n_layers', type=int, default=1, help='Number of layers of the update MLPs')
    parser.add_argument('--aggregator_func', type=str, default='sum', help='Aggregator function for edges, node,...')
    parser.add_argument('--readout_which_output', type=str, default='nodes', help='edges, nodes, globals, graph')

    # ------------------- GN and GNN Arguments
    parser.add_argument('--residual', action='store_true', help='GNN residual')
    parser.add_argument('--graph_pooling', type=str, default='flatten', help='flatten, mean, sum')

    # ----------------------------------

    # data preprocessing
    parser.add_argument('--in_normalize', default='Z', type=str,
                        help="Please give a value for how to normalize data (Z, min_max, none)")
    parser.add_argument('--out_normalize', type=str,
                        help="Please give a value for how to normalize data (Z, min_max, none)")
    parser.add_argument('--spatial_normalization_in', action='store_true',
                        help='Whether to normalize inputs layer/level-wise')
    parser.add_argument('--spatial_normalization_out', action='store_true',
                        help='Whether to normalize outputs layer/level-wise')
    parser.add_argument('--log_scaling', action='store_true', help='Log-scale pressure+height vars?')

    parser.add_argument('--scheduler', default="none", help="Please give a value for using a schedule")
    parser.add_argument('--val_metric', type=str, default='RMSE',
                        help='Which metric to use for early-stopping (lower->better needed)')
    ########
    parser.add_argument('--train_years', type=str, default="1979-83", help='Training set years')
    parser.add_argument('--validation_years', type=str, default="2005", help='Validation set years')
    parser.add_argument('--save_model_to_wandb', type=str, default="True", help='Save best checkpoint to wandb?')
    args = parser.parse_args(args=[] if jupyter_mode else None)

    # parameters
    params = dict()
    params['model'] = args.model
    params['exp_type'] = args.exp_type.lower()
    assert params['exp_type'].replace('_sky', '') in ['pristine', 'clear']
    params['target_type'] = args.target_type
    params['target_variable'] = args.target_variable
    params['model_dir'] = args.model_dir

    params['in_normalize'] = args.in_normalize.lower()
    params['spatial_normalization_in'] = args.spatial_normalization_in
    params['spatial_normalization_out'] = args.spatial_normalization_out
    params['log_scaling'] = args.log_scaling

    if args.out_normalize is None:
        params['out_normalize'] = None
    else:
        params['out_normalize'] = args.out_normalize.lower()

    params["train_years"] = args.train_years
    params["validation_years"] = args.validation_years

    params['nesterov'] = True if args.nesterov.lower() == 'true' else False
    params['shuffle'] = True if args.shuffle.lower() == 'true' else False
    params['loss'] = args.loss
    params['train_on_raw_targets'] = args.train_on_raw_targets
    params['gradient_clipping'] = args.gradient_clipping.lower()
    params['clip'] = args.clip

    params['seed'] = int(args.seed)
    params['epochs'] = int(args.epochs)
    params['batch_size'] = int(args.batch_size)
    params['workers'] = int(args.workers)
    params['load_train_into_mem'] = args.load_train_into_mem
    params['load_val_into_mem'] = args.load_val_into_mem

    params['lr'] = float(args.lr)
    params['weight_decay'] = float(args.weight_decay)
    params['optim'] = args.optim or 'Adam'
    params['scheduler'] = args.scheduler
    params['val_metric'] = args.val_metric
    params['device'] = args.device
    # params['add_coords'] = args.add_coords

    # network parameters
    net_params = dict()
    net_params['activation_function'] = args.act

    if 'L' in args and args.L is not None:
        net_params['L'] = int(args.L)
    else:
        net_params['L'] = len(args.hidden_dims)
    net_params['hidden_dims'] = args.hidden_dims
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)

    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    net_params['net_normalization'] = args.net_norm.lower()
    assert net_params['net_normalization'] in ['none', 'batch_norm', 'layer_norm', 'inst_norm', 'instance_norm',
                                               'group_norm']

    # MODEL SPECIFIC PARAMS
    if is_gnn(params['model']):
        params['node_encoding'] = args.node_encoding
        net_params['learn_edge_structure'] = args.learn_edge_structure
        net_params['improved_self_loops'] = args.improved_self_loops
        net_params['degree_normalized_adj'] = args.degree_normalized_adj
        net_params['pyg_builtin'] = args.pyg_builtin

    if is_graph_net(params['model']):
        net_params['update_mlp_n_layers'] = args.update_mlp_n_layers
        net_params['aggregator_func'] = args.aggregator_func
        net_params['readout_which_output'] = args.readout_which_output

    if is_gnn(params['model']) or is_graph_net(params['model']):
        net_params['residual'] = args.residual
        if 'readout' in params['model'].lower() and \
                (not is_graph_net(params['model']) or net_params['readout_which_output'] != GLOBALS):
            net_params['graph_pooling'] = args.graph_pooling

    if is_cnn(params['model']):
        params['dual_inputs'] = args.dual_inputs
        net_params['dilation'] = args.dilation
        net_params['channels_list'] = args.channels_list
        net_params['gap'] = args.gap
        net_params['use_act'] = args.use_act
        net_params['se_block'] = args.se_block

    params['preprocessing_dict'] = {
        'preprocessing': args.preprocessing.lower(),
        'use_level_features': not args.drop_level_features,
    }
    if 'mlp' in params['preprocessing_dict']['preprocessing']:
        params['preprocessing_dict'] = {
            **params['preprocessing_dict'],
            'projector_n_layers': args.projector_n_layers,
            'projector_net_normalization': args.projector_net_normalization.lower(),
        }
    elif params['preprocessing_dict']['preprocessing'] == 'duplication':
        params['preprocessing_dict'] = {
            **params['preprocessing_dict'],
            'drop_node_encoding': not args.keep_node_encoding,
            'drop_last_level': args.drop_last_level
        }

    def prefix():
        """ This is a prefix for naming the runs for a more agreeable logging."""
        s = args.expID
        if 'clear' in params['exp_type']:
            s += '_CS'
        s += f"_{params['train_years']}train" + f"_{params['validation_years']}val"
        s = s.lstrip('_')
        s += f"_{net_params['L']}L_"
        if args.dropout > 0:
            s += f"{args.dropout}dout_"
        # s += f"{args.readout.upper()}_"
        s += net_params['activation_function'] + 'act_'
        s += args.optim + 'Optim_'
        s += args.scheduler + 'Sched_'

        s += f"{params['batch_size']}bs_"
        s += f"{params['lr']}lr_"
        if args.weight_decay > 0:
            s += f"{args.weight_decay}wd_"

        if all([h == net_params['hidden_dims'][0] for h in net_params['hidden_dims']]):
            hdims = f"{net_params['hidden_dims'][0]}x{len(net_params['hidden_dims'])}"
        else:
            hdims = str(net_params['hidden_dims'])
        s += f"{hdims}h&{net_params['out_dim']}oDim"
        if not params['shuffle']:
            s += 'noShuffle_'
        s += f'{args.seed}seed'

        return s

    set_gpu(args.gpu_id)
    exp_id = prefix()
    params['ID'] = exp_id if len(exp_id) < 128 else exp_id[:128]
    params['wandb_ID'] = wandb.util.generate_id()
    name = get_name(params) + '_' + time.strftime('%Hh%Mm_on_%b_%d') + '_' + params['wandb_ID']
    params['wandb_name'] = name
    return params, net_params, args
