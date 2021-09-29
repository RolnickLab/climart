from abc import ABC
from typing import Optional, Union, Dict, Iterable, Sequence, List

import numpy as np
import torch



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

class Normalizer(ABC):
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


class Z_Normalizer(Normalizer):
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


class MinMax_Normalizer(Normalizer):
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


class LogNormalizer(Normalizer):
    def normalize(self, data, *args, **kwargs):
        normalized_data = self(data)
        return normalized_data

    def inverse_normalize(self, normalized_data):
        data = np.exp(normalized_data)
        return data

    def __call__(self, data: np.ndarray, *args, **kwargs):
        return np.log(data)


class LogZ_Normalizer(Normalizer):
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


class MinMax_LogNormalizer(Normalizer):
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


def get_normalizer(normalizer='z', *args, **kwargs) -> Normalizer:
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
        return Normalizer(*args, **kwargs)  # like no normalizer
