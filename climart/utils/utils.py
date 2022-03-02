"""
Author: Salva Rühling Cachay
"""
import logging
import math
import os
import warnings
import time
import wandb
from functools import wraps
from types import SimpleNamespace
from typing import Union, Sequence, List, Dict, Optional, Callable

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from climart.data_loading import constants, data_variables
from climart.utils.naming import get_group_name, get_detailed_name


def no_op(*args, **kwargs):
    pass


def get_identity_callable(*args, **kwargs) -> Callable:
    return identity


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


def print_config(
        config,
        fields: Union[str, Sequence[str]] = (
                "datamodule",
                "end_model",
                "trainer",
                # "callbacks",
                # "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    import importlib
    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        return
    from omegaconf import DictConfig, OmegaConf
    import rich.syntax
    import rich.tree

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)
    if isinstance(fields, str):
        if fields.lower() == 'all':
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def extras(config) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Modifies DictConfig in place.
    """

    log = get_logger()
    check_config_values(config)

    # Create working dir if it does not exist yet
    if config.get('work_dir'):
        os.makedirs(name=config.get("work_dir"), exist_ok=True)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    if "logger" in config.keys() and config.logger.get("wandb"):
        import wandb
        wandb_id = wandb.util.generate_id()
        config.logger.wandb.id = wandb_id
        group_name = get_group_name(config)
        config.logger.wandb.group = group_name if len(group_name) < 128 else group_name[:128]
        config.logger.wandb.name = get_detailed_name(config) + '_' + time.strftime('%Hh%Mm_on_%b_%d') + '_' + wandb_id


def check_config_values(config: DictConfig):
    exp_type = config.datamodule.exp_type.lower()
    config.datamodule.exp_type = exp_type
    if exp_type not in ["clear_sky", "pristine"]:
        raise ValueError(f"Arg `exp_type` should be one of clear_sky or pristine, but got {exp_type}")

    if "net_normalization" in config.model.keys():
        if config.model.net_normalization is None:
            config.model.net_normalization = "none"
        config.model.net_normalization = config.model.net_normalization.lower()


def get_all_instantiable_hydra_modules(config, module_name: str):
    from hydra.utils import instantiate as hydra_instantiate
    modules = []
    if module_name in config:
        for _, module_config in config[module_name].items():
            if "_target_" in module_config:
                # if 'wandb' in module_config._target_:
                #     modules.append(hydra_instantiate(module_config, settings=wandb.Settings(start_method='fork')))
                modules.append(hydra_instantiate(module_config))
    return modules


def log_hyperparameters(
        config,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Additionally saves:
        - number of {total, trainable, non-trainable} model parameters
    """

    def copy_and_ignore_keys(dictionary, *keys_to_ignore):
        new_dict = dict()
        for k in dictionary.keys():
            if k not in keys_to_ignore:
                new_dict[k] = dictionary[k]
        return new_dict

    params = dict()
    if 'seed' in config:
        params['seed'] = config['seed']
    if 'model' in config:
        params['model'] = config['model']

    # Remove redundant keys or those that are not important to know after training -- feel free to edit this!
    params["datamodule"] = copy_and_ignore_keys(config["datamodule"], 'pin_memory', 'num_workers')
    params['model'] = config['model']
    params["trainer"] = copy_and_ignore_keys(config["trainer"])
    # encoder, optims, and scheduler as separate top-level key
    params['optim'] = config['model']['optim']
    params['scheduler'] = config['model']['scheduler'] if 'scheduler' in config['model'] else None

    if "callbacks" in config:
        if 'model_checkpoint' in config['callbacks']:
            params["model_checkpoint"] = copy_and_ignore_keys(
                config["callbacks"]['model_checkpoint'], 'save_top_k'
            )

    # save number of model parameters
    params["model/params_total"] = sum(p.numel() for p in model.parameters())
    params["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(params)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = no_op


def to_dict(obj: Optional[Union[dict, SimpleNamespace]]):
    if obj is None:
        return dict()
    elif isinstance(obj, dict):
        return obj
    else:
        return vars(obj)


def to_DictConfig(obj: Optional[Union[List, Dict]]):
    from omegaconf import OmegaConf, DictConfig

    if isinstance(obj, DictConfig):
        return obj

    if isinstance(obj, list):
        try:
            dict_config = OmegaConf.from_dotlist(obj)
        except ValueError as e:
            dict_config = OmegaConf.create(obj)

    elif isinstance(obj, dict):
        dict_config = OmegaConf.create(obj)

    else:
        dict_config = OmegaConf.create()  # empty

    return dict_config


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


def year_string_to_list(year_string: str) -> List[int]:
    """
    Args:
        year_string (str): must only contain {digits, '-', '+'}.
    Examples:
        '1988-90' will return [1988, 1989, 1990]
        '1988-1990+2001-2004' will return [1988, 1989, 1990, 2001, 2002, 2003, 2004]
    """
    if not isinstance(year_string, str):
        year_string = str(year_string)

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
        target_vars += data_variables.OUT_LONGWAVE_NOCLOUDS + [data_variables.LW_HEATING_RATE]
    if constants.SHORTWAVE in target_types:
        target_vars += data_variables.OUT_SHORTWAVE_NOCLOUDS + [data_variables.SW_HEATING_RATE]

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


def pressure_from_level_array(levels_array):
    PRESSURE_IDX = 2
    return levels_array[..., PRESSURE_IDX]


def fluxes_to_heating_rates(upwelling_flux: Union[np.ndarray, Tensor],
                            downwelling_flux: Union[np.ndarray, Tensor],
                            pressure: Union[np.ndarray, Tensor],
                            c: float = 9.761357e-03
                            ) -> Union[np.ndarray, Tensor]:
    """
    N - the batch/data dimension size
    L - the number of levels (= number of layers + 1)

    Args:
     upwelling_flux: a (N, L) array
     downwelling_flux: a (N, L) array
     pressure: a (N, L) array representing the levels pressure
            or a (N, L, D-lev) array containing *all* level variables (including pressure)

    Returns:
        A (N, L-1) array representing the heating rates at each of the L-1 layers
    """
    assert upwelling_flux.shape == downwelling_flux.shape
    if len(pressure.shape) <= 2:
        err_msg = f"pressure arg has not the expected shape (N, #levels), but has shape {pressure.shape}"
        assert downwelling_flux.shape == pressure.shape, err_msg
    else:
        err_msg = "pressure argument is not the expected levels array of shape (N, #levels, #level-vars)"
        assert len(pressure.shape) == 3, err_msg
        pressure = pressure_from_level_array(pressure)

    c = 9.761357e-03  # 9.76 * 1e-5
    # c_p = 1004.98322108, g = 9.81, c = g/c_p
    # 3D radiative effects paper uses 8.91/1004 = 0.00977091633
    net_level_flux = upwelling_flux - downwelling_flux
    net_layer_flux = net_level_flux[:, 1:] - net_level_flux[:, :-1]
    pressure_diff = pressure[:, 1:] - pressure[:, :-1]
    heating_rate = c * net_layer_flux / pressure_diff

    assert tuple(heating_rate.shape) == (pressure.shape[0], pressure.shape[1] - 1)

    return heating_rate
