import logging
from abc import ABC
from typing import Optional, Union, Dict, Iterable, Sequence, List, Callable
from omegaconf import DictConfig
import numpy as np
import torch
from torch import Tensor
from climart.data_loading.constants import LEVELS, LAYERS, GLOBALS, get_metadata, get_statistics
from climart.data_loading import constants
from climart.utils.callbacks import PredictionPostProcessCallback
from climart.utils.utils import get_logger, get_target_variable_names, get_identity_callable, identity

NP_ARRAY_MAPPING = Callable[[np.ndarray], np.ndarray]

log = get_logger(__name__)


def in_degree_normalization(adj: np.ndarray):
    """
    For Graph networks
    :param adj: A N x N adjacency matrix
    :return: In-degree normalized matrix (Row-normalized matrix)
    """
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv @ adj
    return mx



class NormalizationMethod(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def normalize(self, data: np.ndarray, axis=0, *args, **kwargs):
        return data

    def inverse_normalize(self, normalized_data: np.ndarray):
        return normalized_data

    def stored_values(self):
        return dict()

    def __copy__(self):
        return type(self)(**self.stored_values())

    def copy(self):
        return self.__copy__()

    def change_input_type(self, new_type):
        for attribute, value in self.__dict__.items():
            if new_type in [torch.Tensor, torch.TensorType]:
                if isinstance(value, np.ndarray):
                    setattr(self, attribute, torch.from_numpy(value))
            elif new_type == np.ndarray:
                if torch.is_tensor(value):
                    setattr(self, attribute, value.numpy().cpu())
            else:
                setattr(self, attribute, new_type(value))

    def apply_torch_func(self, fn):
        """
        Function to be called to apply a torch function to all tensors of this class, e.g. apply .to(), .cuda(), ...,
        Just call this function from within the model's nn.Module._apply()
        """
        for attribute, value in self.__dict__.items():
            if torch.is_tensor(value):
                setattr(self, attribute, fn(value))


class Z_Normalizer(NormalizationMethod):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def normalize(self, data, axis=None, compute_stats=True, *args, **kwargs):
        if compute_stats or self.mean is None:
            self.mean = np.mean(data, axis=axis)
            self.std = np.std(data, axis=axis)
        return self(data)

    def inverse_normalize(self, normalized_data):
        data = normalized_data * self.std + self.mean
        return data

    def stored_values(self):
        return {'mean': self.mean, 'std': self.std}

    def __call__(self, data):
        return (data - self.mean) / self.std


class MinMax_Normalizer(NormalizationMethod):
    def __init__(self, min=None, max_minus_min=None, max=None, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        if min:
            assert max_minus_min or max
            self.max_minus_min = max_minus_min or max - min

    def normalize(self, data, axis=None, *args, **kwargs):
        self.min = np.min(data, axis=axis)
        self.max_minus_min = (np.max(data, axis=axis) - self.min)
        return self(data)

    def inverse_normalize(self, normalized_data):
        shapes = normalized_data.shape
        if len(shapes) >= 2:
            normalized_data = normalized_data.reshape(normalized_data.shape[0], -1)
        data = normalized_data * self.max_minus_min + self.min
        if len(shapes) >= 2:
            data = data.reshape(shapes)
        return data

    def stored_values(self):
        return {'min': self.min, 'max_minus_min': self.max_minus_min}

    def __call__(self, data):
        return (data - self.min) / self.max_minus_min


class LogNormalizer(NormalizationMethod):
    def normalize(self, data, *args, **kwargs):
        normalized_data = self(data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = np.exp(normalized_data)
        return data

    def __call__(self, data: np.ndarray, *args, **kwargs):
        return np.log(data)


class LogZ_Normalizer(NormalizationMethod):
    def __init__(self, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.z_normalizer = Z_Normalizer(mean, std)

    def normalize(self, data, *args, **kwargs):
        normalized_data = np.log(data + 1e-5)
        normalized_data = self.z_normalizer.normalize(normalized_data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = self.z_normalizer.inverse_normalize(normalized_data)
        data = np.exp(data) - 1e-5
        return data

    def stored_values(self):
        return self.z_normalizer.stored_values()

    def change_input_type(self, new_type):
        self.z_normalizer.change_input_type(new_type)

    def apply_torch_func(self, fn):
        self.z_normalizer.apply_torch_func(fn)

    def __call__(self, data, *args, **kwargs):
        normalized_data = np.log(data + 1e-5)
        return self.z_normalizer(normalized_data)


class MinMax_LogNormalizer(NormalizationMethod):
    def __init__(self, min=None, max_minus_min=None, **kwargs):
        super().__init__(**kwargs)
        self.min_max_normalizer = MinMax_Normalizer(min, max_minus_min)

    def normalize(self, data, *args, **kwargs):
        normalized_data = self.min_max_normalizer.normalize(data)
        normalized_data = np.log(normalized_data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = np.exp(normalized_data)
        data = self.min_max_normalizer.inverse_normalize(data)
        return data

    def stored_values(self):
        return self.min_max_normalizer.stored_values()

    def change_input_type(self, new_type):
        self.min_max_normalizer.change_input_type(new_type)

    def apply_torch_func(self, fn):
        self.min_max_normalizer.apply_torch_func(fn)


def get_normalizer(normalizer='z', *args, **kwargs) -> NormalizationMethod:
    normalizer = normalizer.lower().strip().replace('-', '_').replace('&', '+')
    supported_normalizers = ['z',
                             'min_max',
                             'min_max+log', 'min_max_log',
                             'log_z',
                             'log',
                             'none']
    assert normalizer in supported_normalizers, f"Unsupported Normalization {normalizer} not in {str(supported_normalizers)}"
    if normalizer == 'z':
        return Z_Normalizer(*args, **kwargs)
    elif normalizer == 'min_max':
        return MinMax_Normalizer(*args, **kwargs)
    elif normalizer in ['min_max+log', 'min_max_log']:
        return MinMax_LogNormalizer(*args, **kwargs)
    elif normalizer in ['logz', 'log_z']:
        return LogZ_Normalizer(*args, **kwargs)
    elif normalizer == 'log':
        return LogNormalizer(*args, **kwargs)
    else:
        return NormalizationMethod(*args, **kwargs)  # like no normalizer


class Normalizer:
    def __init__(
            self,
            datamodule_config: DictConfig,
            input_normalization: Optional[str] = None,
            output_normalization: Optional[str] = None,
            spatial_normalization_in: bool = False,
            spatial_normalization_out: bool = False,
            log_scaling: Union[bool, List[str]] = False,
            data_dir: Optional[str] = None,
            verbose: bool = True
    ):
        """
        input_normalization (str): "z" for z-scaling (zero mean and unit standard deviation)
        """
        if not verbose:
            log.setLevel(logging.WARNING)

        if data_dir is None:
            data_dir = constants.DATA_DIR
        exp_type = datamodule_config.get("exp_type")
        target_type = datamodule_config.get("target_type")
        target_variable = datamodule_config.get("target_variable")

        self._layer_mask = 45 if exp_type == constants.CLEAR_SKY else 14
        self._recover_meta_info(data_dir)
        self._input_normalizer: Dict[str, Callable] = dict()
        self._output_normalizer = None

        self._target_variables = get_target_variable_names(target_type, target_variable)
        print(datamodule_config, exp_type, "!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        self.output_variable_splitter = PredictionPostProcessCallback(
            variable_to_channel=self.feature_by_var[f"outputs_{exp_type}"], variables=self._target_variables,
        )

        if input_normalization:
            norma_type = '_spatial' if spatial_normalization_in else ''
            info_msg = f" Applying {norma_type.lstrip('_')} {input_normalization} normalization to input data," \
                       f" based on pre-computed stats."
            log.info(info_msg)

            precomputed_stats = get_statistics(data_dir)
            precomputed_stats = {k: precomputed_stats[k] for k in precomputed_stats.keys() if
                                 (('spatial' in k and spatial_normalization_in) or ('spatial' not in k))}
            if isinstance(log_scaling, list) or log_scaling:
                log.info(' Log scaling pressure and height variables! (no other normalization is applied to them)')
                post_log_vals = dict(pressg=(11.473797, 0.10938317),
                                     layer_pressure=(9.29207, 2.097411),
                                     dz=(6.5363674, 1.044927),
                                     layer_thickness=(6.953938313568889, 1.3751644503732554),
                                     level_pressure=(9.252319, 2.1721559))
                vars_to_log_scale = ['pressg', 'layer_pressure', 'dz', 'layer_thickness', 'level_pressure']
                self._layer_log_mask = torch.tensor([2, 5, 12])
                for var in vars_to_log_scale:
                    dtype = self._variables[var]['data_type']
                    s, e = self.feature_by_var[dtype][var]['start'], self.feature_by_var[dtype][var]['end']
                    # precomputed_stats[f'{dtype}{prefix}_mean'][..., s:e] = 0
                    # precomputed_stats[f'{dtype}{prefix}_std'][..., s:e] = 1
                    precomputed_stats[f'{dtype}{norma_type}_mean'][..., s:e] = post_log_vals[var][0]
                    precomputed_stats[f'{dtype}{norma_type}_std'][..., s:e] = post_log_vals[var][1]

                def log_scaler(X: Dict[str, Tensor]) -> Dict[str, Tensor]:
                    # layer_log_mask = torch.tensor([2, 5, 12])
                    X[GLOBALS][2] = torch.log(X[GLOBALS][2])
                    X[LEVELS][..., 2] = torch.log(X[LEVELS][..., 2])
                    X[LAYERS][..., self._layer_log_mask] = torch.log(X[LAYERS][..., self._layer_log_mask])
                    return X

                self._log_scaler_func = log_scaler
            else:
                self._log_scaler_func = identity

            for data_type in [GLOBALS, LEVELS, LAYERS]:
                if input_normalization is not None:
                    normer_kwargs = dict(
                        mean=precomputed_stats[data_type + f'{norma_type}_mean'],
                        std=precomputed_stats[data_type + f'{norma_type}_std'],
                        min=precomputed_stats[data_type + f'{norma_type}_min'],
                        max=precomputed_stats[data_type + f'{norma_type}_max'],
                    )
                    if data_type == LAYERS:
                        for k, v in normer_kwargs.items():
                            normer_kwargs[k] = v[..., :self._layer_mask]
                    normalizer = get_normalizer(
                        input_normalization,
                        **normer_kwargs,
                        variable_to_channel=self.feature_by_var[data_type]
                    )
                else:
                    normalizer = identity
                self._input_normalizer[data_type] = normalizer

    def _recover_meta_info(self, data_dir: str):
        meta_info = get_metadata(data_dir)
        self._variables = meta_info['variables']
        self._vars_used_or_not = list(self._variables.keys())
        self._feature_by_var = meta_info['feature_by_var']

    @property
    def feature_by_var(self):
        return self._feature_by_var

    def get_normalizer(self, data_type: str) -> Union[NP_ARRAY_MAPPING, NormalizationMethod]:
        return self._input_normalizer[data_type]

    def get_normalizers(self) -> Dict[str, Union[NP_ARRAY_MAPPING, NormalizationMethod]]:
        return {
            data_type: self.get_normalizer(data_type)
            for data_type in constants.INPUT_TYPES
        }

    def set_normalizer(self, data_type: str, new_normalizer: Optional[NP_ARRAY_MAPPING]):
        if new_normalizer is None:
            new_normalizer = identity
        if data_type in constants.INPUT_TYPES:
            self._input_normalizer[data_type] = new_normalizer
        else:
            self._output_normalizer = new_normalizer

    def set_input_normalizers(self, new_normalizer: Optional[NP_ARRAY_MAPPING]):
        for data_type in constants.INPUT_TYPES:
            self.set_normalizer(data_type, new_normalizer)

    @property
    def output_normalizer(self):
        if self._output_normalizer is None:
            return None
        return self._output_normalizer

    def normalize(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for input_type, rawX in X.items():
            X[input_type] = self._input_normalizer[input_type](rawX)

        X = self._log_scaler_func(X)
        return X

    def __call__(self, X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return self.normalize(X)