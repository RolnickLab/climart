import json
import os
from typing import Optional, Sequence, List

import h5py
import numpy as np

from climart.data_wrangling.constants import INPUT_TYPES, LAYERS, TRAIN_YEARS, DATA_DIR, SPATIAL_DATA_TYPES, DATA_TYPE_DIMS
from climart.data_wrangling.data_variables import DONT_NORMALIZE
from climart.utils.utils import get_logger

log = get_logger(__name__)


def pre_compute_dataset_statistics_on_h5(
        training_years: Optional[List[int]] = None,
        compute_spatial_stats_too: bool = False
):
    if training_years is None:
        # default 1979-2005, without 1991-93 for pinatubo OOD
        training_years = TRAIN_YEARS

    with open(f'{DATA_DIR}/META_INFO.json', 'r') as fp:
        meta_info_all = json.load(fp)
    meta_info = meta_info_all['feature_by_var']
    data_types = list(meta_info.keys())

    def get_all_datatype_data(direc: str, h5_array: str, start: int = 0, upto: int = None) -> np.ndarray:
        data = None
        for year in training_years:
            h5file_year = os.path.join(direc, f'{year}.h5')
            with h5py.File(h5file_year, 'r') as h5f:
                new_data = np.array(h5f[h5_array])
                if isinstance(upto, int) and upto > 0:
                    new_data = new_data[..., start:upto]
                if data is None:
                    data = new_data
                else:
                    data = np.concatenate([data, new_data], axis=0)
        return data

    statistics = ['mean', 'std', 'min', 'max']
    to_save = {
        data_type + '_' + stat: np.zeros(shape=DATA_TYPE_DIMS[data_type], dtype='float32')
        for data_type in INPUT_TYPES for stat in statistics
    }

    for data_type in data_types:
        if data_type in INPUT_TYPES:
            in_out_name = 'inputs'
            data_arrays = [data_type]
        else:
            in_out_name = data_type
            data_arrays = meta_info[data_type].keys()
        direc = os.path.join(DATA_DIR, in_out_name)
        log.info(f' Normalizing {data_type}: {data_arrays}')
        for array in data_arrays:
            name = data_type if data_type in INPUT_TYPES else f"{data_type}_{array}"
            variables = meta_info[data_type]  # if data_type in INPUT_TYPES else meta_info[data_type][array]
            # all_data = get_all_datatype_data(direc, h5_array=array)
            if compute_spatial_stats_too and data_type in SPATIAL_DATA_TYPES:
                spatial_shape = 49 if array in [LAYERS, 'hrlc', 'hrsc', 'hrl', 'hrs'] else 50
                for stat in statistics:
                    to_save[name + '_spatial_' + stat] = np.zeros((spatial_shape, DATA_TYPE_DIMS[data_type]),
                                                                  dtype='float32').squeeze()
            for var_name in variables.keys():
                log.info(f' Computing the statistics for variable {var_name}.')
                start, end = variables[var_name]['start'], variables[var_name]['end']
                all_data = get_all_datatype_data(direc, h5_array=array, start=start, upto=end)
                if var_name in DONT_NORMALIZE:
                    scalar_mean, scalar_std = 0, 1  # z-scaling will be like identity
                else:
                    # ... will make sure that we select all prior dimensions, and index the feature dimension with start-end
                    # scalar_mean = np.mean(all_data[..., start:end], dtype=np.float64)
                    # scalar_std = np.std(all_data[..., start:end], dtype=np.float64)
                    scalar_mean = np.mean(all_data, dtype=np.float64)
                    scalar_std = np.std(all_data, dtype=np.float64)

                stat_to_value = [('mean', scalar_mean), ('std', scalar_std),
                                 ('min', np.min(all_data)),
                                 ('max', np.max(all_data))]
                for stat, val in stat_to_value:
                    if data_type in INPUT_TYPES:
                        to_save[name + f'_{stat}'][start:end] = val
                    else:
                        to_save[name + f'_{stat}'] = val

                if compute_spatial_stats_too and data_type in SPATIAL_DATA_TYPES:
                    stat_to_value = [
                        ('mean', 0 if var_name in DONT_NORMALIZE else np.mean(all_data, dtype=np.float64, axis=0)),
                        ('std', 1 if var_name in DONT_NORMALIZE else np.std(all_data, dtype=np.float64, axis=0)),
                        ('min', np.min(all_data, axis=0)),
                        ('max', np.max(all_data, axis=0))
                    ]
                    for stat, val in stat_to_value:
                        if data_type in INPUT_TYPES:
                            to_save[name + f'_spatial_{stat}'][..., start:end] = val
                        else:
                            to_save[name + f'_spatial_{stat}'] = val

            del all_data
            log.info(f'Statistics for {data_type}-{array} computed!')
    output_file = os.path.join(DATA_DIR, 'statistics.npz')
    np.savez(output_file, **to_save)
    log.info(f"Statistics have been successfully saved to {output_file}")
    return output_file
