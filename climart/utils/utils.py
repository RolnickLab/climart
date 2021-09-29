"""
Author: Salva Rühling Cachay
"""
import logging
import math
import os
from functools import wraps
from typing import Union, Sequence, List, Dict, Optional, Callable

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from climart.data_wrangling import constants, data_variables


def get_activation_function(name: str, functional: bool = False, num: int = 1):
    name = name.lower().strip()

    def get_functional(s: str) -> Optional[Callable]:
        return {"softmax": F.softmax, "relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
                "identity": nn.Identity(),
                None: None, 'swish': F.silu, 'silu': F.silu, 'elu': F.elu, 'gelu': F.gelu, 'prelu': nn.PReLU(),
                }[s]

    def get_nn(s: str) -> Optional[Callable]:
        return {"softmax": nn.Softmax(dim=1), "relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(),
                "identity": nn.Identity(), 'silu': nn.SiLU(), 'elu': nn.ELU(), 'prelu': nn.PReLU(),
                'swish': nn.SiLU(), 'gelu': nn.GELU(),
                }[s]

    if num == 1:
        return get_functional(name) if functional else get_nn(name)
    else:
        return [get_nn(name) for _ in range(num)]


def get_normalization_layer(name, dims, num_groups=None, device='cpu'):
    if not isinstance(name, str) or name.lower() == 'none':
        return None
    elif 'batch' in name:
        return nn.BatchNorm1d(num_features=dims).to(device)
    elif 'layer' in name:
        return nn.LayerNorm(dims).to(device)
    elif 'inst' in name:
        return nn.InstanceNorm1d(num_features=dims).to(device)
    elif 'group' in name:
        if num_groups is None:
            num_groups = int(dims / 10)
        return nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)


def identity(X):
    return X


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def _get_rank() -> int:
    rank_keys = ('RANK', 'SLURM_PROCID', 'LOCAL_RANK')
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, 'rank', _get_rank())


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def adj_to_edge_indices(adj: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Args:
        adj: a (N, N) adjacency matrix, where N is the number of nodes
    Returns:
        A (2, E) array, edge_idxs, where E is the number of edges,
                and edge_idxs[0],  edge_idxs[1] are the source & destination nodes, respectively.
    """
    edge_tuples = torch.nonzero(adj, as_tuple=True) if torch.is_tensor(adj) else np.nonzero(adj)
    edge_src = edge_tuples[0].unsqueeze(0) if torch.is_tensor(adj) else np.expand_dims(edge_tuples[0], axis=0)
    edge_dest = edge_tuples[1].unsqueeze(0) if torch.is_tensor(adj) else np.expand_dims(edge_tuples[1], axis=0)
    if torch.is_tensor(adj):
        edge_idxs = torch.cat((edge_src, edge_dest), dim=0)
    else:
        edge_idxs = np.concatenate((edge_src, edge_dest), axis=0)
    return edge_idxs


def normalize_adjacency_matrix_torch(adj: Tensor, improved: bool = True, add_self_loops: bool = False):
    if add_self_loops:
        fill_value = 2. if improved else 1.
        adj = adj.fill_diagonal_(fill_value)
    deg: Tensor = torch.sum(adj, dim=1)
    deg_inv_sqrt: Tensor = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = torch.mul(adj, deg_inv_sqrt.view(-1, 1))
    adj_t = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


def normalize_adjacency_matrix(adj: np.ndarray, improved: bool = True, add_self_loops: bool = True):
    if add_self_loops:
        fill_value = 2. if improved else 1.
        np.fill_diagonal(adj, fill_value)
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.

    deg_inv_sqrt_matrix = np.diag(deg_inv_sqrt)
    adj_normed = deg_inv_sqrt_matrix @ adj @ deg_inv_sqrt_matrix
    return adj_normed


def set_gpu(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_seed(seed, device='cuda'):
    import random, torch
    # setting seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_name(params):
    ID = params['model'].upper()
    if 'clear' in params['exp_type']:
        ID += '_CS'
    ID += f"_{params['train_years']}train_{params['validation_years']}val"
    ID += f"_{params['in_normalize'].upper()}"
    if params['spatial_normalization_in'] and params['spatial_normalization_out']:
        ID += '+spatialNormed'
    elif params['spatial_normalization_in']:
        ID += '+spatialInNormed'
    elif params['spatial_normalization_out']:
        ID += '+spatialOutNormed'

    ID += '_' + str(params['seed']) + 'seed'
    return ID


def stem_word(word):
    return word.lower().strip().replace('-', '').replace('&', '').replace('+', '').replace('_', '')


# CanAM specific functions to find out the year corresponding to CanAM snapshots/time steps
def canam_file_id_to_year_fraction(canam_filename: str) -> float:
    if '/' in canam_filename:
        canam_filename = canam_filename.split('/')[-1]
    ID = canam_filename.replace('CanAM_snapshot_', '').replace('.nc', '')
    ID = int(ID)
    year = (ID / (365 * 24 * 4)) + 1
    return year


def get_year_to_canam_files_dict(canam_filenames: Sequence[str]) -> Dict[int, List[str]]:
    years = [
        int(math.floor(canam_file_id_to_year_fraction(fname))) for fname in canam_filenames
    ]
    mapping = dict()
    for fname, year in zip(canam_filenames, years):
        if year not in mapping.keys():
            mapping[year] = []
        mapping[year].append(fname)
    return mapping


def year_string_to_list(year_string: str):
    """
    Args:
        year_string (str): must only contain {digits, '-', '+'}.
    Examples:
        '1988-90' will return [1988, 1989, 1990]
        '1988-1990+2001-2004' will return [1988, 1989, 1990, 2001, 2002, 2003, 2004]
    """
    if not isinstance(year_string, str):
        return year_string

    def year_string_to_full_year(year_string: str):
        if len(year_string) == 4:
            return int(year_string)
        assert len(year_string) == 2, f'Year {year_string} had an unexpected length.'
        if int(year_string[0]) < 3:
            return int('20' + year_string)
        else:
            return int('19' + year_string)

    def update_years(year_list: List[int], year_start, year_end):
        if not isinstance(year_start, int):
            year_start = year_string_to_full_year(year_start)
        if year_end == '':
            year_end = year_start
        else:
            year_end = year_string_to_full_year(year_end)
        year_list += list(range(year_start, year_end + 1))
        return year_list, '', ''

    years = []
    cur_year_start = cur_year_end = ''
    for char in year_string:
        if char == '-':
            cur_year_start = year_string_to_full_year(cur_year_start)
        elif char == '+':
            years, cur_year_start, cur_year_end = update_years(years, cur_year_start, cur_year_end)
        else:
            if isinstance(cur_year_start, int):
                cur_year_end += char
            else:
                cur_year_start += char
    years, _, _ = update_years(years, cur_year_start, cur_year_end)
    return years


def compute_absolute_level_height(dz_layer_heights: xr.DataArray) -> xr.DataArray:
    """ Call with dz_layer_heights=YourDataset['dz'] """
    # layers=slice(None, None, -1) or levels=slice(None, None, -1) will simply reverse the data along that dim
    # Since levels=0 corresponds to TOA, this is needed, so that cumsum correctly accumulates from surface -> TOA
    surface_to_toa = dz_layer_heights.pad(layers=(0, 1), constant_values=0).sel(layers=slice(None, None, -1))
    # surface_to_toa[column = i] = [0, d_height_layer1, ..., d_height_lastLayer]
    level_abs_heights = surface_to_toa.cumsum(dim='layers').rename({'layers': 'levels'})
    toa_to_surface = level_abs_heights.sel(levels=slice(None, None, -1))  # reverse back to the existing format
    return toa_to_surface


def compute_temperature_diff(level_temps: xr.DataArray) -> xr.DataArray:
    """
    Usage:
        Call with level_temps=YourDataset['tfrow'], assuming that 'tfrow' is the temperature var. at the levels

    Returns:
        A xr.DataArray with same dimensions as level_temps, except for `levels` being replaced by `layer`.
        In the layer dimension, it will hold that:
            layer_i_tempDiff = level_i+1_temp - level_i_temp
        Note: This means that the temperature at *spatially higher* layers is subtracted from its adjacent lower layer.
            E.g., the layer next to the surface will get surface - level_one_above_surface
    """
    layer_temp_diffs = level_temps.diff(dim='levels', n=1).rename({'levels': 'layers'})
    return layer_temp_diffs


def get_target_types(target_type: Union[str, List[str]]) -> List[str]:
    if isinstance(target_type, list):
        assert all([t in [constants.SHORTWAVE, constants.LONGWAVE] for t in target_type])
        return target_type
    target_type2 = target_type.lower().replace('&', '+').replace('-', '')
    if target_type2 in ['sw+lw', 'lw+sw', 'shortwave+longwave', 'longwave+shortwave']:
        return [constants.SHORTWAVE, constants.LONGWAVE]
    elif target_type2 in ['sw', 'shortwave']:
        return [constants.SHORTWAVE]
    elif target_type2 in ['lw', 'longwave']:
        return [constants.LONGWAVE]
    else:
        raise ValueError(f"Target type `{target_type}` must be one of shortwave, longwave or shortwave+longwave")


def get_target_variable_names(target_types: Union[str, List[str]],
                              target_variable: Union[str, List[str]]) -> List[str]:
    out_vars = data_variables.OUT_SHORTWAVE_NOCLOUDS + data_variables.OUT_LONGWAVE_NOCLOUDS \
               + data_variables.OUT_HEATING_RATE_NOCLOUDS
    if isinstance(target_variable, list):
        if len(target_variable) == 1:
            target_variable = target_variable[0]
        else:
            err_msg = f"Each target var must be in {out_vars}, but got {target_variable}"
            assert all([t.lower() in out_vars for t in target_variable]), err_msg
            return target_variable

    target_types = get_target_types(target_types)
    target_variable2 = target_variable.lower().replace('&', '+').replace('-', '').replace('_', '')
    target_variable2 = target_variable2.replace('fluxes', 'flux').replace('heatingrate', 'hr')
    target_vars: List[str] = []
    if constants.LONGWAVE in target_types:
        if 'flux' in target_variable2:
            target_vars += data_variables.OUT_LONGWAVE_NOCLOUDS
        if 'hr' in target_variable2:
            target_vars += [data_variables.LW_HEATING_RATE]
    if constants.SHORTWAVE in target_types:
        if 'flux' in target_variable2:
            target_vars += data_variables.OUT_SHORTWAVE_NOCLOUDS
        if 'hr' in target_variable2:
            target_vars += [data_variables.SW_HEATING_RATE]

    if len(target_vars) == 0:
        raise ValueError(f"Target var `{target_variable2}` must be one of fluxes, heating_rate.")
    return target_vars


def get_target_variable(target_variable: Union[str, List[str]]) -> List[str]:
    if isinstance(target_variable, list):
        if len(target_variable) == 1 and 'flux' in target_variable[0]:
            target_variable = target_variable[0]
        else:
            return target_variable
    target_variable2 = target_variable.lower().replace('&', '+').replace('-', '').replace('_', '')
    target_variable2 = target_variable2.replace('fluxes', 'flux').replace('heatingrate', 'hr')
    target_vars: List[str] = []
    if target_variable2 == 'hr':
        return [constants.SURFACE_FLUXES, constants.TOA_FLUXES, constants.HEATING_RATES]
    else:
        if 'flux' in target_variable2:
            target_vars += [constants.FLUXES]
        if 'hr' in target_variable2:
            target_vars += [constants.HEATING_RATES]

    if len(target_vars) == 0:
        raise ValueError(f"Target var `{target_variable2}` must be one of fluxes, heating_rate.")
    return target_vars


def get_exp_ID(exp_type: str, target_types: Union[str, List[str]], target_variables: Union[str, List[str]]):
    s = f"{exp_type.upper()} conditions, with {' '.join(target_types)} x {' '.join(target_variables)} targets"
    return s
