# |----------------------- HDF5 DATASET -----------------------|
import copy
import logging
import os
import pickle
import shutil
import zipfile
from typing import Dict, Optional, List, Callable, Tuple, Union

import h5py
import numpy as np
import torch
from torch import Tensor

from climart.data_loading.constants import LEVELS, LAYERS, GLOBALS, EXP_TYPES
from climart.data_loading import constants
from climart.data_transform.normalization import Normalizer
from climart.data_transform.normalization import NormalizationMethod
from climart.data_transform.transforms import AbstractTransform, IdentityTransform
from climart.utils.utils import get_logger, get_target_variable_names, pressure_from_level_array

log = get_logger(__name__)


class ClimART_HdF5_Dataset(torch.utils.data.Dataset):
    """Class for working with multiple HDF5 datasets"""

    def __init__(
            self,
            years: List[int],
            name: str = "",
            data_dir: Optional[str] = None,
            exp_type: str = 'pristine',
            target_type: Union[str, List[str]] = 'shortwave',
            target_variable: Union[str, List[str]] = 'fluxes+hr',
            normalizer: Optional[Normalizer] = None,
            input_transform: Optional[AbstractTransform] = None,
            output_transform: Optional[AbstractTransform] = None,
            load_h5_into_mem: bool = False,
            verbose: bool = True
    ):
        """
        """
        self._name = str(name)
        self.normalizer = copy.deepcopy(normalizer)
        if not verbose:
            log.setLevel(logging.WARNING)

        if isinstance(years, int):
            years = [years]
        assert all([y in constants.ALL_YEARS for y in years]), 'All years must be within 1979-2014.'

        if not isinstance(input_transform, AbstractTransform):
            input_transform = IdentityTransform(exp_type)
        if not isinstance(output_transform, AbstractTransform):
            output_transform = IdentityTransform(exp_type)

        self.years = years
        self.n_years = len(years)
        self._load_h5_into_mem = load_h5_into_mem
        self._layer_mask = 45 if exp_type == constants.CLEAR_SKY else 14

        if data_dir is None:
            log.info(" No data_dir was specified, defaulting to climart.data_loading.constants.DATA_DIR")
            data_dir = constants.DATA_DIR

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
            # normalization is done before starting training
            dset_kwargs['input_normalizer'] = self.normalizer.get_normalizers()
            self.normalizer.set_input_normalizers(new_normalizer=None)

            dset_kwargs['input_transform'] = input_transform  # batched
            dset_kwargs['output_transform'] = output_transform

            self._input_transform = IdentityTransform(exp_type)
            self._output_transform = IdentityTransform(exp_type)
        else:
            dset_class = RT_HdF5_SingleDataset
            self._input_transform: AbstractTransform = input_transform  # not batched
            self._output_transform: AbstractTransform = output_transform

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
            X = raw_Xs
            Y = raw_Ys
        else:
            Xs = self.normalizer(raw_Xs['data'])
            Xs = self._input_transform.transform(Xs)
            X = {'data': Xs,
                 'level_pressure_profile': raw_Xs['level_pressure_profile']}

            Y = self._output_transform.transform(raw_Ys)
            Y = {k: torch.from_numpy(v).float() for k, v in Y.items()}

        return X, Y

    @property
    def spatial_dim(self) -> Dict[str, int]:
        return self.h5_dsets[0].spatial_dim

    @property
    def input_dim(self) -> Dict[str, int]:
        return self.h5_dsets[0].input_dim

    @property
    def output_dim(self) -> Dict[str, int]:
        return 100  # self.h5_dsets[0].output_dim

    def close(self):
        for dset in self.h5_dsets:
            dset.close()


class RT_HdF5_SingleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 filename: str,
                 data_dir: str,
                 exp_type: str = 'pristine',
                 target_type: Union[str, List[str]] = 'shortwave',
                 target_variable: Union[str, List[str]] = 'fluxes+hr'
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
                GLOBALS: np.array(h5f[GLOBALS][index]),
            }
            X = {'data': X, 'level_pressure_profile': pressure_from_level_array(X[LEVELS])}
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
        X = {'data': X,
             'level_pressure_profile': pressure_from_level_array(X[LEVELS])
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
        elif isinstance(v, NormalizationMethod) or isinstance(v, AbstractTransform):
            name = v.__class__.__name__
        else:
            try:
                name = v.__qualname__.replace('.', '_')
            except Exception as e:
                print(e)
                name = "?"
        processed += f'_{name}{k}'
    return processed + f'{ending}'


class RT_HdF5_FastSingleDataset(RT_HdF5_SingleDataset):
    def __init__(self,
                 input_normalizer: Optional[Dict[str, Callable]] = None,
                 input_transform: Optional[AbstractTransform] = None,
                 output_transform: Optional[AbstractTransform] = None,
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
        # if os.path.isfile(input_processed_fname) and reload_if_exists:
        #    self.input_data = self._reload_data(input_processed_fname)
        # if os.path.isfile(output_processed_fname) and reload_if_exists:
        #    self.output_data = self._reload_data(output_processed_fname)

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

        if isinstance(self.input_data, dict) and 'edges' in self.input_data and self.input_data['edges'].shape[
            -1] > 14 and self.exp_type == 'pristine':
            log.info('overwriting', input_processed_fname)
            self._preprocess_h5data(input_normalizer, input_transform, output_transform, outputs=False)
            self._write_data(input_processed_fname, self.input_data)

    def _reload_data(self, fname):
        log.info(f' Reloading from {fname}')
        try:
            in_data = np.load(fname, allow_pickle=True)
        except zipfile.BadZipFile as e:
            log.warning(f"{fname} was not properly saved or has been corrupted.")
            raise e
        try:
            in_files = in_data.files
        except AttributeError:
            return in_data

        if len(in_files) == 1:
            return in_data[in_files[0]]
        else:
            return {k: in_data[k] for k in in_files}

    def _write_data(self, fname, data):
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        if isinstance(data, dict) or isinstance(data, np.ndarray):
            try:
                np.savez_compressed(fname, **data) if isinstance(data, dict) else np.savez_compressed(fname, data)
            except OverflowError:
                with open(fname, "wb") as fp:
                    pickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)
            except PermissionError as e:
                log.warning(f" Tried to cache data to {fname} but got error {e}. "
                            f"Consider adjusting the permissions to cache the data and make training more efficient.")
        else:
            raise ValueError(f"Data has type {type(data)}")

    def _preprocess_h5data(self,
                           input_normalizer: Optional[Dict[str, Callable]] = None,
                           input_transform: AbstractTransform = None,
                           output_transform: AbstractTransform = None,
                           inputs: bool = True,
                           outputs: bool = True,
                           ):
        if inputs:
            with h5py.File(self.input_path, 'r') as input_data_h5:
                input_data = {
                    LAYERS: np.array(input_data_h5[LAYERS][:, :, :self._layer_mask]),
                    LEVELS: np.array(input_data_h5[LEVELS]),
                    GLOBALS: np.array(input_data_h5[GLOBALS])
                }
                level_pressure_profile = pressure_from_level_array(input_data[LEVELS])

            if input_normalizer is not None:
                for k in constants.INPUT_TYPES:
                    input_data[k] = input_normalizer[k](input_data[k])

            if input_transform is not None:
                input_data = input_transform.batched_transform(input_data)
            self.input_data = {
                'data': input_data,
                'level_pressure_profile': level_pressure_profile
            }

        if outputs:
            with h5py.File(self.output_path, 'r') as output_data_h5:
                output_data = {
                    output_var: np.array(output_data_h5[output_var])
                    for output_var in self._target_variables
                }

            if output_transform is not None:
                output_data = output_transform.batched_transform(X=output_data)
            self.output_data = output_data

    def copy_to_slurm_tmp_dir(self):
        pass

    def __getitem__(self, index) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        if isinstance(self.input_data['data'], np.ndarray) and len(self.input_data['data'].shape) == 0:
            # Recover a nested dictionary of np arrays back, i.e. transform np.ndarray of object dtype into dict
            self.input_data['data'] = self.input_data['data'].item()

        if isinstance(self.input_data['data'], np.ndarray):
            X = torch.from_numpy(self.input_data['data'][index]).float()
        else:
            X = {k: torch.from_numpy(v[index]).float() for k, v in self.input_data['data'].items()}

        X = {'data': X,
             'level_pressure_profile': torch.from_numpy(self.input_data['level_pressure_profile'][index]).float()
             }

        if isinstance(self.output_data, np.ndarray):
            Y = torch.from_numpy(self.output_data[index]).float()
        else:
            Y = {k: torch.from_numpy(v[index]).float() for k, v in self.output_data.items()}

        return X, Y
