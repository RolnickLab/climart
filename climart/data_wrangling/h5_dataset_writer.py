# |----------------------- HDF5 writing/ dataset collection from netCDF -----------------------|
import json
import logging
import os
from typing import Dict, Optional, List, Callable, Tuple, Union

import h5py
import numpy as np
import xarray as xr

from climart.data_wrangling.data_variables import reorder_input_variables
from climart.data_wrangling.constants import LEVELS, LAYERS, GLOBALS, EXP_TYPES
from climart.utils.utils import get_logger

NP_ARRAY_MAPPING = Callable[[np.ndarray], np.ndarray]
log = get_logger(__name__)


class ClimART_GeneralHdF5_Writer:
    def __init__(self,
                 input_data: xr.Dataset,
                 output_data: Dict[str, xr.Dataset],
                 save_name: str,
                 save_dir: str = '/miniscratch/salva.ruhling-cachay/ECC_data/snapshots/1979-2014/hdf5',
                 *args, **kwargs):
        self.save_dir = save_dir
        self.filename = save_name
        if not self.filename.endswith('.h5'):
            self.filename += '.h5'
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(self.input_h5_dir, exist_ok=True)
        for exp_type in EXP_TYPES:
            os.makedirs(os.path.join(save_dir, f'outputs_{exp_type}'), exist_ok=True)

        self.vars_used_or_not = {}
        self.feature_by_var = {GLOBALS: dict(), LEVELS: dict(), LAYERS: dict(),
                               **{'outputs_' + exp_type: dict() for exp_type in EXP_TYPES}}

        self._n_lev, self._n_lay = input_data.sizes['levels'], input_data.sizes['layers']
        self._create_input_dataset(input_data)
        del input_data
        self._create_output_dataset(output_data)
        del output_data

        self._write_meta_info_to_json()

    @property
    def input_h5_dir(self) -> str:
        fpath = os.path.join(self.save_dir, 'inputs')
        return fpath

    def h5_filepaths(self) -> Dict[str, str]:
        return {
            'inputs': os.path.join(self.input_h5_dir, self.filename),
            **{
                f'outputs_{exp_type}': os.path.join(self.save_dir, f'outputs_{exp_type}/{self.filename}')
                for exp_type in EXP_TYPES
            }
        }

    def _create_input_dataset(self, dataset: xr.Dataset, *args, **kwargs) -> Dict[str, np.ndarray]:
        if os.path.isfile(self.h5_filepaths()['inputs']):
            log.info(f" {self.h5_filepaths()['inputs']} already exists.")
            return dict()
        var_names = list(dataset.keys())
        dataset = dataset.transpose('columns', ...)  # bring spatial dim to the front
        dataset_size = dataset.sizes['columns']

        data_temp = {GLOBALS: None, LEVELS: None, LAYERS: None}
        # Order equivalent vars to same channel for different components:
        ordered_var_names = reorder_input_variables(var_names)
        for var_name in ordered_var_names:
            self.vars_used_or_not[var_name] = True

            var_data = dataset[var_name].values
            if len(np.unique(var_data)) <= 1:
                print(f'{var_name} only has one value: {np.unique(var_data)[0]}!!!!')

            # GLOBALS
            dim_names = dataset[var_name].dims
            if len(var_data.shape) <= 1 or ('layers' not in dim_names and 'levels' not in dim_names):
                var_type = GLOBALS
                var_data = var_data.reshape(dataset_size, -1)  # num_features = var_data.shape[1]
            # LEVEL VAR
            elif 'levels' in dim_names:
                var_type = LEVELS
                var_data = var_data.reshape((dataset_size, self._n_lev, -1))  # num_features = var_data.shape[2]
            # LAYER VAR
            elif 'layers' in dim_names:
                var_type = LAYERS
                var_data = var_data.reshape((dataset_size, self._n_lay, -1))  # num_features = var_data.shape[2]
            else:
                raise ValueError()

            # Concatenate variable to its corresponding np array in the data_temp dict
            if data_temp[var_type] is None:
                data_temp[var_type] = var_data
            else:
                data_temp[var_type] = np.concatenate((data_temp[var_type], var_data), axis=-1)
            n_feats_var = var_data.shape[-1]
            self._build_feature_by_var_dict(var_name,
                                            var_type,
                                            n_features_var=n_feats_var,
                                            n_features_without_var=data_temp[var_type].shape[-1] - n_feats_var)

        with h5py.File(self.h5_filepaths()['inputs'], 'w') as h5f:
            for input_key, input_array in data_temp.items():
                dset = h5f.create_dataset(input_key, data=input_array, compression=None, dtype='float32')
        log.info(f" {self.filename} HDF5 created at: {self.h5_filepaths()['inputs']}")
        return data_temp

    def _create_output_dataset(self, datasets: Dict[str, xr.Dataset], *args, **kwargs) -> None:
        for exp_type, dataset in datasets.items():
            var_names = list(dataset.keys())
            dataset = dataset.transpose('columns', ...)  # bring spatial dim to the front
            data_temp = dict()
            for var_name in var_names:
                self.vars_used_or_not[var_name] = True
                var_data = dataset[var_name].values
                if len(np.unique(var_data)) <= 1:
                    print(f'{var_name} only has one value: {np.unique(var_data)[0]}!!!!')

                data_temp[f"{var_name.lower()}"] = var_data
                n_feats_var = var_data.shape[1]
                self._build_feature_by_var_dict(var_name,
                                                var_type='outputs_' + exp_type,
                                                n_features_var=n_feats_var,
                                                n_features_without_var=0)

            with h5py.File(self.h5_filepaths()[f'outputs_{exp_type}'], 'w') as h5f:
                for output_key, output_array in data_temp.items():
                    h5f.create_dataset(output_key, data=output_array, compression=None, dtype='float32')
            log.info(f" {self.filename} created at {self.h5_filepaths()[f'outputs_{exp_type}']}")

    def _build_feature_by_var_dict(self,
                                   var_name: str,
                                   var_type: str,
                                   n_features_var: int,
                                   n_features_without_var
                                   ):
        self.feature_by_var[var_type][var_name] = {
            'start': n_features_without_var,
            'end': n_features_without_var + n_features_var,
        }

    def get_meta_info_json_path(self) -> str:
        return os.path.join(self.save_dir, 'META_INFO.json')

    def _write_meta_info_to_json(self):
        meta_info_path = self.get_meta_info_json_path()
        if os.path.isfile(meta_info_path):
            log.info(f" {meta_info_path} already exists. Not overwriting it.")
            return
        meta_info = {
            'used_vars': self.vars_used_or_not,
            'feature_by_var': self.feature_by_var,
            'input_dims': {
                exp: {} for exp in EXP_TYPES
            },
            'output_dims': {
                exp: {} for exp in EXP_TYPES
            },
        }
        with open(meta_info_path, 'w') as fp:
            json.dump(meta_info, fp)
        log.info(f'Meta information has been written to {meta_info_path}')
