import os
import glob

import math
from functools import partial
from multiprocessing import Pool
from typing import Union, Optional, List, Tuple, Dict

import xarray as xr
import numpy as np
from tqdm import tqdm

from climart.data_wrangling.constants import INPUT_RAW_SUBDIR, DATA_CREATION_SEED
from climart.data_wrangling.data_variables import get_all_vars, input_variables_to_drop_for_exp, \
    output_variables_to_drop_for_exp, EXP_TYPES, _ALL_INPUT_VARS
from climart.data_wrangling.h5_dataset_writer import ClimART_GeneralHdF5_Writer
from climart.utils.utils import compute_absolute_level_height, get_year_to_canam_files_dict, \
    get_logger, compute_temperature_diff

log = get_logger(__name__)
CACHE = {}


def get_single_var_complete_xarray(input_filepaths: List[str],
                                   variable: str,
                                   exp_type: str = 'pristine'
                                   ) -> xr.DataArray:
    var_type = 'input' if variable in _ALL_INPUT_VARS else 'output'
    if var_type == 'output':
        input_filepaths = [f.replace('input_raw', f'output_{exp_type.lower()}') for f in input_filepaths]
    vars_to_drop = ['iseed'] + get_all_vars(var_type=var_type, exp_type='all')
    if variable == 'layer_pressure':
        vars_to_keep = ['shj', 'pressg']
    elif variable == 'level_pressure':
        vars_to_keep = ['shtj', 'pressg']
    elif variable == 'layer_thickness':
        vars_to_keep = ['dshj', 'pressg']
    elif variable == 'temp_diff':
        vars_to_keep = ['tfrow']
    elif variable == 'height':
        vars_to_keep = ['dz']
    else:
        vars_to_keep = [variable]

    for var_to_keep in vars_to_keep:
        vars_to_drop.remove(var_to_keep)

    open_dset_kwargs = dict(
        paths=input_filepaths,
        preprocess=lambda d: d.stack(columns=['latitude', 'longitude']),
        concat_dim='columns',
        combine="nested",
        data_vars=vars_to_keep,
        drop_variables=vars_to_drop,
    )
    dset = xr.open_mfdataset(**open_dset_kwargs)[vars_to_keep]

    if variable == 'layer_pressure':
        dset = dset['shj'] * dset['pressg']
    elif variable == 'level_pressure':
        dset = dset['shtj'] * dset['pressg']
    elif variable == 'layer_thickness':
        dset = dset['dshj'] * dset['pressg']
    elif variable == 'temp_diff':
        dset = compute_temperature_diff(dset['tfrow'])
    elif variable == 'height':
        dset = compute_absolute_level_height(dset['dz'])
    else:
        dset = dset[variable]
    return dset


def get_single_dataset(input_filename: str,
                       add_pressure: bool = True,
                       add_temp_diff: bool = True,
                       add_layer_thickness: bool = True,
                       add_absolute_level_height: bool = True,
                       add_coords: bool = True,
                       use_cache_for_coords: bool = True,
                       verbose=False, **kwargs) -> (xr.Dataset, xr.Dataset):
    remove_vars_in = input_variables_to_drop_for_exp['clear_sky']
    input_dset = xr.open_dataset(input_filename,
                                 drop_variables=remove_vars_in,
                                 chunks={"longitude": 10, "latitude": 10})
    output_dsets = dict()
    for exp_type in EXP_TYPES:
        output_filename = input_filename.replace(INPUT_RAW_SUBDIR, f"output_{exp_type}")
        remove_vars_out = output_variables_to_drop_for_exp[exp_type]
        output_dset = xr.open_dataset(output_filename,
                                      drop_variables=remove_vars_out,
                                      chunks={"longitude": 10, "latitude": 10})
        output_dsets[exp_type] = output_dset

    if add_coords:
        # Convert lat/lon to points on a unit sphere: https://datascience.stackexchange.com/a/13575
        if use_cache_for_coords and 'x_cord' in CACHE.keys():
            input_dset['x_cord'] = CACHE['x_cord']
            input_dset['y_cord'] = CACHE['y_cord']
            input_dset['z_cord'] = CACHE['z_cord']
        else:
            lat = list(input_dset.get_index('latitude'))
            lon = list(input_dset.get_index('longitude'))
            x_cord, y_cord, z_cord = [], [], []

            for i in lat:
                for j in lon:
                    x = math.cos(i) * math.cos(j)
                    y = math.cos(i) * math.sin(j)
                    z = math.sin(i)
                    x_cord.append(x)
                    y_cord.append(y)
                    z_cord.append(z)
            x_cord, y_cord, z_cord = np.array(x_cord), np.array(y_cord), np.array(z_cord)
            x_cord = x_cord.reshape((len(lat), len(lon)))
            y_cord = y_cord.reshape((len(lat), len(lon)))
            z_cord = z_cord.reshape((len(lat), len(lon)))

            dim_and_coords = dict(dims=('latitude', 'longitude'),
                                  coords={"latitude": lat, 'longitude': lon})
            input_dset['x_cord'] = xr.DataArray(data=x_cord, **dim_and_coords)
            input_dset['y_cord'] = xr.DataArray(data=y_cord, **dim_and_coords)
            input_dset['z_cord'] = xr.DataArray(data=z_cord, **dim_and_coords)
            if use_cache_for_coords:
                CACHE['x_cord'] = input_dset['x_cord']
                CACHE['y_cord'] = input_dset['y_cord']
                CACHE['z_cord'] = input_dset['z_cord']

    # Flatten spatial dimensions:
    input_dset = input_dset.stack(columns=['latitude', 'longitude'])
    for k, output_dset in output_dsets.items():
        output_dsets[k] = output_dset.stack(columns=['latitude', 'longitude'])

    if add_pressure:
        # pressure is of shape #levels x (lat x lon), shtj too, and pressg of shape (lat x lon)
        input_dset['layer_pressure'] = input_dset['shj'] * input_dset['pressg']
        input_dset['level_pressure'] = input_dset['shtj'] * input_dset['pressg']
    if add_layer_thickness:
        input_dset['layer_thickness'] = input_dset['dshj'] * input_dset['pressg']
    if add_temp_diff:
        input_dset['temp_diff'] = compute_temperature_diff(input_dset['tfrow'])
    if add_absolute_level_height:
        input_dset['height'] = compute_absolute_level_height(input_dset['dz'])

    if verbose:
        print(input_dset)
        print('---' * 25)
        print(output_dsets)
        print('*+*' * 40, '\n')
    return input_dset, output_dsets


def get_ML_dataset(input_files, split: str = "", **kwargs) -> (xr.DataArray, Dict[str, xr.DataArray]):
    print(f"{split}: there are {len(input_files)} files to be loaded")
    X = None
    out_dsets = dict()
    for input_filepath in input_files:
        X_temp, Y_temp_dict = get_single_dataset(input_filepath, **kwargs)
        X = X_temp if X is None else xr.concat([X, X_temp], dim='columns')
        for k, out_dset in Y_temp_dict.items():
            if k not in out_dsets.keys():
                out_dsets[k] = out_dset
            else:
                out_dsets[k] = xr.concat([out_dsets[k], out_dset], dim='columns')

    return X, out_dsets


def create_ML_dataset_and_h5(
        x: Tuple[int, Tuple[int, List[str]]],
):
    save_dir = '/miniscratch/salva.ruhling-cachay/ECC_data/snapshots/1979-2014/hdf5'
    i, (year, fnames) = x
    X_year, Y_dict_year = get_ML_dataset(
        input_files=fnames,
        split=str(year),
        add_pressure=True,
        add_temp_diff=True,
        add_layer_thickness=True,
        add_absolute_level_height=True,
        add_coords=True,
        use_cache_for_coords=True
    )
    h5_dset = ClimART_GeneralHdF5_Writer(
        input_data=X_year,
        output_data=Y_dict_year,
        save_name=str(year),
        save_dir=save_dir
    )


def create_yearly_h5_files(
        raw_data_dir: str,
        val_year_start: int = 2005,
        test_year_start: int = 2007,
        test_pinatubo: bool = True,
        train_files_per_year: Union[str, int] = 'all',
        val_files_per_year: Union[str, int] = 'all',
        test_files_per_year: Union[str, int] = 15,
        which_years: Union[str, List[int]] = 'all',
        multiprocessing_cpus: Optional[int] = 0
):
    """ If h5_dir is None, a directory will be automatically created."""
    all_input_files = glob.glob(raw_data_dir + f'/{INPUT_RAW_SUBDIR}/**/*.nc', recursive=True)
    rng = np.random.default_rng(seed=DATA_CREATION_SEED)

    year_to_all_fnames = get_year_to_canam_files_dict(all_input_files)
    year_to_fnames = dict()
    for year, fnames in year_to_all_fnames.items():
        if isinstance(which_years, list) and year not in which_years:
            log.info(f"Skipping year {year} as requested by `which_years` arg.")
            continue
        elif test_pinatubo and 1991 <= year <= 1993:
            to_add = fnames
        elif 1979 <= year < val_year_start:
            to_add = fnames if train_files_per_year in ['all', -1] else rng.choice(fnames, size=train_files_per_year)
        elif val_year_start <= year < test_year_start:
            to_add = fnames if val_files_per_year in ['all', -1] else rng.choice(fnames, size=val_files_per_year)
        elif test_year_start <= year < 2015:
            to_add = fnames if test_files_per_year in ['all', -1] else rng.choice(fnames, size=test_files_per_year)
        else:
            to_add = fnames
            log.info(f'Using all snapshots for year {year} for h5 creation.')
            # raise ValueError()
        year_to_fnames[year] = to_add

    del year_to_all_fnames

    iterable_years = enumerate(year_to_fnames.items())
    if multiprocessing_cpus <= 0:
        log.info(f"Using a single GPU/CPU")
        for x in iterable_years:
            create_ML_dataset_and_h5(x)
    else:
        log.info(f"Using multiprocessing on {multiprocessing_cpus} CPUs")
        pool = Pool(multiprocessing_cpus)
        total = len(year_to_fnames.keys())
        for _ in tqdm(pool.imap_unordered(
                create_ML_dataset_and_h5,
                iterable_years), total=total
        ):
            pass
