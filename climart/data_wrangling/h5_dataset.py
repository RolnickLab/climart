# |----------------------- HDF5 DATASET -----------------------|
import json
import logging
import os
import shutil
from typing import Dict, Optional, List, Callable, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor

from climart.data_wrangling.constants import LEVELS, LAYERS, GLOBALS, EXP_TYPES, get_metadata, get_statistics
from climart.data_wrangling import constants
from climart.utils.callbacks import PredictionPostProcessCallback
from climart.utils.preprocessing import get_normalizer, Normalizer
from climart.utils.utils import get_logger, identity, get_target_variable_names

NP_ARRAY_MAPPING = Callable[[np.ndarray], np.ndarray]

log = get_logger(__name__)


def base_output_transform(raw_Ys: Dict[str, np.ndarray]):
    Y = np.concatenate([y.reshape((-1,)) for y in raw_Ys.values()], axis=0)
    return Y


def base_output_transform_batched(raw_Ys: Dict[str, np.ndarray]):
    Y = np.concatenate([y.reshape((y.shape[0], -1)) for y in raw_Ys.values()], axis=1)
    return Y


def get_basic_output_transform(batched: bool):
    return base_output_transform_batched if batched else base_output_transform


def get_basic_input_transform(batched: bool):
    return identity


class ClimART_HdF5_Dataset(torch.utils.data.Dataset):
    """Class for working with multiple HDF5 datasets"""

    def __init__(
            self,
            years: List[int],
            name: str = "",
            data_dir: Optional[str] = None,
            exp_type: str = 'pristine',
            target_type: Union[str, List[str]] = 'shortwave',
            target_variable: Union[str, List[str]] = 'fluxes',
            input_normalization: Optional[str] = None,
            input_transform: Callable[[bool], Callable] = None,
            output_normalization: Optional[str] = None,
            output_transform: Callable[[bool], Callable] = None,
            spatial_normalization_in: bool = False,
            spatial_normalization_out: bool = False,
            log_scaling: Union[bool, List[str]] = False,
            load_h5_into_mem: bool = False,
            verbose: bool = True
    ):
        """
        """
        self._name = str(name)
        if not verbose:
            log.setLevel(logging.WARNING)
        if input_transform is None:
            input_transform = get_basic_input_transform
        if output_transform is None:
            output_transform = get_basic_output_transform

        if isinstance(years, int):
            years = [years]
        print(years)
        assert all([y in constants.ALL_YEARS for y in years]), 'All years must be within 1979-2014.'
        msg = 'must return a batched transform if its arg is True and otherwise a non-batched transform.'
        assert callable(input_transform), f"input_transform {msg}"
        assert callable(output_transform), f"output_transform {msg}"
        self.years = years
        self.n_years = len(years)
        self._load_h5_into_mem = load_h5_into_mem

        if data_dir is None:
            data_dir = constants.DATA_DIR
        self._recover_meta_info(data_dir)

        self._layer_mask = 45 if exp_type == constants.CLEAR_SKY else 14

        self._output_normalizer = None
        if input_normalization:
            normalization = input_normalization
            prefix = '_spatial' if spatial_normalization_in else ''
            info_msg = f" {self.name}: Applying {prefix.lstrip('_')} {normalization} normalization to input data," \
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
                    precomputed_stats[f'{dtype}{prefix}_mean'][..., s:e] = post_log_vals[var][0]
                    precomputed_stats[f'{dtype}{prefix}_std'][..., s:e] = post_log_vals[var][1]

                def log_scaler(X: Dict[str, Tensor]):
                    # layer_log_mask = torch.tensor([2, 5, 12])
                    X[GLOBALS][2] = torch.log(X[GLOBALS][2])
                    X[LEVELS][..., 2] = torch.log(X[LEVELS][..., 2])
                    X[LAYERS][..., self._layer_log_mask] = torch.log(X[LAYERS][..., self._layer_log_mask])
                    return X

                self._log_scaler_func = log_scaler
            else:
                self._log_scaler_func = identity

            for data_type in [GLOBALS, LEVELS, LAYERS]:

                if normalization is not None:
                    n_kwargs = dict(
                        mean=precomputed_stats[data_type + f'{prefix}_mean'],
                        std=precomputed_stats[data_type + f'{prefix}_std'],
                        min=precomputed_stats[data_type + f'{prefix}_min'],
                        max=precomputed_stats[data_type + f'{prefix}_max'],
                    )
                    if data_type == LAYERS:
                        for k, v in n_kwargs.items():
                            n_kwargs[k] = v[..., :self._layer_mask]
                    normalizer = get_normalizer(
                        normalization,
                        **n_kwargs,
                        variable_to_channel=self.feature_by_var[data_type]
                    )
                else:
                    normalizer = None
                setattr(self, f'{data_type}_normalizer', normalizer)

        self._target_variables = get_target_variable_names(target_type, target_variable)
        # print(self._target_variables, target_type, target_variable)
        self.output_variable_splitter = PredictionPostProcessCallback(
            variable_to_channel=self.feature_by_var[f"outputs_{exp_type}"], variables=self._target_variables,
        )

        self.dataset_index_to_sub_dataset: Dict[int, Tuple[int, int]] = dict()

        dataset_size = 0
        dset_kwargs = dict(
            data_dir=data_dir,
            exp_type=exp_type,
            target_type=target_type,
            target_variable=target_variable
        )
        if load_h5_into_mem:
            log.info(f" {self.name} Loading the H5 into RAM & pre-processing it!")
            dset_class = RT_HdF5_FastSingleDataset
            dset_kwargs['input_normalizer'] = self.get_normalizers()
            self.set_input_normalizers(new_normalizer=None)  # normalization is done before starting training

            input_transform = input_transform(batched=True)
            if not isinstance(input_transform, torch.nn.Module):
                dset_kwargs['input_transform'] = input_transform  # batched
            self._input_transform = identity

            dset_kwargs['output_transform'] = output_transform(batched=True)
            self._output_transform = identity
        else:
            dset_class = RT_HdF5_SingleDataset
            self._input_transform = input_transform(batched=False)  # not batched
            if isinstance(self._input_transform, torch.nn.Module):
                self._input_transform = identity

            self._output_transform = output_transform(batched=False)

        self.h5_dsets: List[dset_class] = []
        for file_num, year in enumerate(years):
            year_h5_dset = dset_class(filename=f"{str(year)}.h5", **dset_kwargs)
            n_samples_of_year = len(year_h5_dset)
            for h5_file_idx in range(n_samples_of_year):
                self.dataset_index_to_sub_dataset[h5_file_idx + dataset_size] = (file_num, h5_file_idx)
            dataset_size += n_samples_of_year
            self.h5_dsets.append(year_h5_dset)
        self.dataset_size = dataset_size

        for dset in self.h5_dsets:
            dset.copy_to_slurm_tmp_dir()

        log.info(self)

    @property
    def name(self):
        return self._name.upper()

    @property
    def output_normalizer(self):
        if self._output_normalizer is None:
            return None
            self._output_normalizer = self.h5_dsets[0].get_preprocesser(data_type=OUTPUT)
        return self._output_normalizer

    def set_input_transform(self, new_transform: Callable):
        self._input_transform = new_transform

    def __str__(self):
        s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
        return s

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item) -> (Dict[str, Tensor], Tensor):  # Dict[str, Tensor]):
        which_h5, h5_index = self.dataset_index_to_sub_dataset[item]
        raw_Xs, raw_Ys = self.h5_dsets[which_h5][h5_index]
        if self._load_h5_into_mem:
            Xs = raw_Xs
            Y = raw_Ys
        else:
            Xs = dict()
            for input_type, rawX in raw_Xs.items():
                Xs[input_type] = torch.from_numpy(
                    self.get_normalizer(input_type)(rawX)
                ).float()

            Xs = self._log_scaler_func(Xs)
            Xs = self._input_transform(Xs)

            Y = self._output_transform(raw_Ys)
            Y = torch.from_numpy(np.array(Y)).float()
        # Ys = dict()
        # for output_type, rawY in raw_Ys.items():
        #    Ys[output_type] = torch.from_numpy(raw_Ys[output_type]).float()
        return Xs, Y

    def _recover_meta_info(self, data_dir: str):
        meta_info = get_metadata(data_dir)
        if meta_info is None:
            self._variables = self._vars_used_or_not = self._feature_by_var = None
        else:
            self._variables = meta_info['variables']
            self._vars_used_or_not = list(self._variables.keys())
            self._feature_by_var = meta_info['feature_by_var']

    @property
    def feature_by_var(self):
        return self._feature_by_var  # self.h5_dsets[0].feature_by_var

    @property
    def spatial_dim(self) -> Dict[str, int]:
        return self.h5_dsets[0].spatial_dim

    @property
    def input_dim(self) -> Dict[str, int]:
        return self.h5_dsets[0].input_dim

    @property
    def output_dim(self) -> Dict[str, int]:
        return 100  # self.h5_dsets[0].output_dim

    def get_normalizer(self, data_type: str) -> Union[NP_ARRAY_MAPPING, Normalizer]:
        return getattr(self, f"{str(data_type)}_normalizer")

    def get_normalizers(self) -> Dict[str, Union[NP_ARRAY_MAPPING, Normalizer]]:
        return {
            data_type: self.get_normalizer(data_type)
            for data_type in constants.INPUT_TYPES
        }

    def set_normalizer(self, data_type: str, new_normalizer: Optional[NP_ARRAY_MAPPING]):
        if new_normalizer is None:
            new_normalizer = identity
        setattr(self, f'{data_type}_normalizer', new_normalizer)

    def set_input_normalizers(self, new_normalizer: Optional[NP_ARRAY_MAPPING]):
        for data_type in constants.INPUT_TYPES:
            self.set_normalizer(data_type, new_normalizer)

    def close(self):
        for dset in self.h5_dsets:
            dset.close()


class RT_HdF5_SingleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filename: str,
                 data_dir: str,
                 exp_type: str = 'pristine',
                 target_type: Union[str, List[str]] = 'shortwave',
                 target_variable: Union[str, List[str]] = 'fluxes'
                 ):
        assert exp_type in EXP_TYPES, f"arg `exp_type`= {exp_type} is not one of {str(EXP_TYPES)}!"
        self.exp_type = exp_type
        self._filename = filename
        self._name = self._filename.replace('.h5', '')
        self._target_variables = get_target_variable_names(target_type, target_variable)
        self._layer_mask = 45 if exp_type == constants.CLEAR_SKY else 14

        dirs = constants.get_data_subdirs(data_dir)
        in_dir, out_dir = dirs[constants.INPUTS], dirs[exp_type]
        self._in_path = os.path.join(in_dir, filename)
        self._out_path = os.path.join(out_dir, filename)

        with h5py.File(self._in_path, 'r') as h5f:
            self._num_examples = h5f[LAYERS].shape[0]
            n_levels = h5f[LEVELS].shape[1]
            n_layers = h5f[LAYERS].shape[1]
            self._spatial_dim = {GLOBALS: 0, LEVELS: n_levels, LAYERS: n_layers}

            n_glob_feats = h5f[GLOBALS].shape[1]
            n_level_feats = h5f[LEVELS].shape[2]
            n_layer_feats = h5f[LAYERS][..., :self._layer_mask].shape[2]

            self._in_dim = {GLOBALS: n_glob_feats, LEVELS: n_level_feats, LAYERS: n_layer_feats}

        with h5py.File(self._out_path, 'r') as h5f:
            self._out_dim = {
                key: h5f[key].shape[-1] for key in self._target_variables
            }

    def copy_to_slurm_tmp_dir(self):
        # todo: move this logic up, only keep copy-file logic here (&support copy+normalization!)
        if 'SLURM_TMPDIR' in os.environ:
            log.info(f' Copying {self.name} h5 file to SLURM_TMPDIR')
            h5_path_new_in = os.environ['SLURM_TMPDIR'] + '/input_' + self._filename
            shutil.copyfile(self._in_path, h5_path_new_in)
            self._in_path = h5_path_new_in

            h5_path_new_out = os.environ['SLURM_TMPDIR'] + '/output_' + self._filename
            shutil.copyfile(self._out_path, h5_path_new_out)
            self._out_path = h5_path_new_out

    # def __str__(self):
    #     return get_exp_ID(self.exp_type, self.target_types, self.target_variables)

    @property
    def name(self):
        return self._name

    @property
    def input_path(self):
        return self._in_path

    @property
    def output_path(self):
        return self._out_path

    def __len__(self):
        return self._num_examples

    def __getitem__(self, index) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        with h5py.File(self.input_path, 'r') as h5f:
            X = {
                LAYERS: np.array(h5f[LAYERS][index, :, :self._layer_mask]),
                LEVELS: np.array(h5f[LEVELS][index]),
                GLOBALS: np.array(h5f[GLOBALS][index])
            }
        with h5py.File(self.output_path, 'r') as h5f:
            Y = {
                output_var: np.array(h5f[output_var][index])
                for output_var in self._target_variables
            }

        return X, Y

    def get_raw_input_data(self) -> Dict[str, np.ndarray]:
        with h5py.File(self.input_path, 'r') as h5f:
            raw_data = {
                LAYERS: np.array(h5f[LAYERS][..., :self._layer_mask]),
                LEVELS: np.array(h5f[LEVELS]),
                GLOBALS: np.array(h5f[GLOBALS])
            }
        return raw_data

    def get_dataset_statistics(self, statistics: List[str] = None) -> Dict[str, np.ndarray]:
        if statistics is None:
            statistics = ['mean', 'std']
        raw_data = self.get_raw_input_data()
        stats = dict()
        axes_to_norm = {GLOBALS: 0, LEVELS: (0, 1), LAYERS: (0, 1)}
        for data_type, data in raw_data.items():
            if 'mean' in statistics:
                stats[data_type + '_mean'] = np.mean(data, axis=axes_to_norm[data_type], dtype=np.float64)
            if 'std' in statistics:
                stats[data_type + '_std'] = np.std(data, axis=axes_to_norm[data_type], dtype=np.float64)
            if 'min' in statistics:
                stats[data_type + '_min'] = np.min(data, axis=axes_to_norm[data_type])
            if 'max' in statistics:
                stats[data_type + '_max'] = np.max(data, axis=axes_to_norm[data_type])

        return stats

    @property
    def spatial_dim(self) -> Dict[str, int]:
        return self._spatial_dim

    @property
    def input_dim(self) -> Dict[str, int]:
        return self._in_dim

    @property
    def output_dim(self) -> Dict[str, int]:
        return self._out_dim

    def close(self):
        pass


class RT_HdF5_FastSingleDatasetTwo(RT_HdF5_SingleDataset):
    def __init__(self, input_preprocesser: Optional[Dict[str, Callable]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = h5py.File(self.input_path, 'r')
        self.output_data = h5py.File(self.output_path, 'r')

    def close(self):
        self.input_data.close()
        self.output_data.close()

    def __getitem__(self, index) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        X = {
            LAYERS: np.array(self.input_data[LAYERS][index, :, :self._layer_mask]),
            LEVELS: np.array(self.input_data[LEVELS][index]),
            GLOBALS: np.array(self.input_data[GLOBALS][index])
        }
        Y = {
            output_var: np.array(self.output_data[output_var][index])
            for output_var in self._target_variables
        }

        return X, Y


def get_processed_fname(h5_name: str, ending='.npz', **kwargs):
    processed = h5_name.replace('.h5', '').replace('hdf5/', 'numpy/')
    for k, v in kwargs.items():
        if v is None:
            name = 'No'
        elif isinstance(v, str):
            name = v.upper()
        elif isinstance(v, Normalizer):
            name = v.__class__.__name__
        else:
            name = v.__qualname__.replace('.', '_')
        processed += f'_{name}k'  # {k}'
    return processed + f'{ending}'


class RT_HdF5_FastSingleDataset(RT_HdF5_SingleDataset):
    def __init__(self,
                 input_normalizer: Optional[Dict[str, Callable]] = None,
                 input_transform: Callable[[Dict[str, np.ndarray]], np.ndarray] = None,
                 output_transform: Callable[[Dict[str, np.ndarray]], np.ndarray] = None,
                 write_data: bool = True,
                 reload_if_exists: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        pkwargs = dict(
            in_normalizer=input_normalizer[GLOBALS],
            in_transform=input_transform,
            out_transform=output_transform
        )
        ending = '.npz' if self.exp_type == 'pristine' else f"_{self.exp_type}.npz"
        input_processed_fname = get_processed_fname(self.input_path, **pkwargs, ending=ending)
        output_processed_fname = get_processed_fname(self.output_path, **pkwargs, ending='.npz')
        if os.path.isfile(input_processed_fname) and os.path.isfile(output_processed_fname) and reload_if_exists:
            self.input_data = self._reload_data(input_processed_fname)
            self.output_data = self._reload_data(output_processed_fname)

        elif os.path.isfile(input_processed_fname) and reload_if_exists:
            self.input_data = self._reload_data(input_processed_fname)
            self._preprocess_h5data(input_normalizer, input_transform, output_transform, inputs=False)
            if write_data:
                self._write_data(output_processed_fname, self.output_data)

        else:
            self._preprocess_h5data(input_normalizer, input_transform, output_transform)
            if write_data:
                self._write_data(input_processed_fname, self.input_data)
                self._write_data(output_processed_fname, self.output_data)

        if isinstance(self.input_data, dict) and 'edges' in self.input_data and self.input_data['edges'].shape[-1] > 14 and self.exp_type == 'pristine':
            log.info('overwriting', input_processed_fname)
            self._preprocess_h5data(input_normalizer, input_transform, output_transform, outputs=False)
            self._write_data(input_processed_fname, self.input_data)

    def _reload_data(self, fname):
        log.info(f' Reloading from {fname}')
        in_data = np.load(fname)
        in_files = in_data.files
        if len(in_files) == 1:
            return in_data[in_files[0]]
        else:
            return {k: in_data[k] for k in in_files}

    def _write_data(self, fname, data):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if isinstance(data, dict):
            np.savez_compressed(fname, **data)
        elif isinstance(data, np.ndarray):
            np.savez_compressed(fname, data)
        else:
            raise ValueError(f"Data has type {type(data)}")

    def _preprocess_h5data(self,
                           input_normalizer: Optional[Dict[str, Callable]] = None,
                           input_transform: Callable[[Dict[str, np.ndarray]], np.ndarray] = None,
                           output_transform: Callable[[Dict[str, np.ndarray]], np.ndarray] = None,
                           inputs: bool = True,
                           outputs: bool = True,
                           ):

        if inputs:
            input_data_h5 = h5py.File(self.input_path, 'r')
            input_data = {
                LAYERS: np.array(input_data_h5[LAYERS][:, :, :self._layer_mask]),
                LEVELS: np.array(input_data_h5[LEVELS]),
                GLOBALS: np.array(input_data_h5[GLOBALS])
            }

            if input_normalizer is not None:
                input_data = {k: input_normalizer[k](v) for k, v in input_data.items()}
            if input_transform is not None:
                self.input_data = input_transform(input_data)
            else:
                self.input_data = input_data
            input_data_h5.close()
        if outputs:
            output_data_h5 = h5py.File(self.output_path, 'r')
            output_data = {
                output_var: np.array(output_data_h5[output_var])
                for output_var in self._target_variables
            }
            if output_transform is not None:
                self.output_data = output_transform(output_data)
            else:
                self.output_data = output_data
            output_data_h5.close()

    def copy_to_slurm_tmp_dir(self):
        pass

    def __getitem__(self, index) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        if isinstance(self.input_data, np.ndarray):
            X = torch.from_numpy(self.input_data[index]).float()
        else:
            X = {k: torch.from_numpy(v[index]).float() for k, v in self.input_data.items()}

        if isinstance(self.output_data, np.ndarray):
            Y = torch.from_numpy(self.output_data[index]).float()
        else:
            Y = {k: torch.from_numpy(v[index]).float() for k, v in self.output_data.items()}

        return X, Y
