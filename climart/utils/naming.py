from typing import Union, List
from omegaconf import DictConfig


def _shared_prefix(config: DictConfig, init_prefix: str = "") -> str:
    s = init_prefix if isinstance(init_prefix, str) else ""
    if 'clear' in config.datamodule.get('exp_type'):
        s += '_CS'
    s += f"_{config.datamodule.get('train_years')}train" + f"_{config.datamodule.get('validation_years')}val"
    if config.normalizer.get('input_normalization'):
        s += f"_{config.normalizer.get('input_normalization').upper()}"
    if config.normalizer.get('output_normalization'):
        s += f"_{config.normalizer.get('output_normalization').upper()}"
    return s.lstrip('_')


def get_name_for_hydra_config_class(config: DictConfig) -> str:
    if 'name' in config and config.get('name') is not None:
        return config.get('name')
    elif '_target_' in config:
        return config._target_.split('.')[-1]
    return "$"


def get_detailed_name(config) -> str:
    """ This is a prefix for naming the runs for a more agreeable logging."""
    s = config.get("name", '')
    s = _shared_prefix(config, init_prefix=s) + '_'
    if config.model.get('dropout') > 0:
        s += f"{config.model.get('dropout')}dout_"

    s += config.model.get('activation_function') + '_'
    s += get_name_for_hydra_config_class(config.model.optimizer) + '_'
    s += get_name_for_hydra_config_class(config.model.scheduler) + '_'

    s += f"{config.datamodule.get('batch_size')}bs_"
    s += f"{config.model.optimizer.get('lr')}lr_"
    if config.model.optimizer.get('weight_decay') > 0:
        s += f"{config.model.optimizer.get('weight_decay')}wd_"

    hdims = config.model.get('hidden_dims')
    if all([h == hdims[0] for h in hdims]):
        hdims = f"{hdims[0]}x{len(hdims)}"
    else:
        hdims = str(hdims)
    s += f"{hdims}h"  # &{net_params['out_dim']}oDim"
    # if not params['shuffle']:
    #    s += 'noShuffle_'
    s += f"{config.get('seed')}seed"

    return s.replace('None', '')


def get_model_name(name: str) -> str:
    if 'CNN' in name:
        return 'CNN'
    elif 'MLP' in name:
        return 'MLP'
    elif 'GCN' in name:
        return 'GCN'
    elif 'GN' in name:
        return 'GraphNet'
    else:
        raise ValueError(name)


def get_group_name(config) -> str:
    s = get_name_for_hydra_config_class(config.model)
    s = s.lower().replace('net', '').replace('_', '').replace("climart", "").replace("with", "+").upper()
    s = _shared_prefix(config, init_prefix=s)

    if config.normalizer.get('spatial_normalization_in') and config.normalizer.get('spatial_normalization_out'):
        s += '+spatialNormed'
    elif config.normalizer.get('spatial_normalization_in'):
        s += '+spatialInNormed'
    elif config.normalizer.get('spatial_normalization_out'):
        s += '+spatialOutNormed'

    return s


def stem_word(word: str) -> str:
    return word.lower().strip().replace('-', '').replace('&', '').replace('+', '').replace('_', '')


def get_exp_ID(exp_type: str, target_types: Union[str, List[str]], target_variables: Union[str, List[str]]):
    s = f"{exp_type.upper()} conditions, with {' '.join(target_types)} x {' '.join(target_variables)} targets"
    return s
