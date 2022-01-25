import json
import logging
import os
import numpy as np
from typing import Optional, Dict

import xarray

GLOBALS = "globals"
LAYERS = "layers"
LEVELS = "levels"
INPUTS = 'inputs'
PRISTINE = 'pristine'
CLEAR_SKY = 'clear_sky'
OUTPUTS_PRISTINE = f"outputs_{PRISTINE}"
OUTPUTS_CLEARSKY = f"outputs_{CLEAR_SKY}"
SHORTWAVE = 'shortwave'
LONGWAVE = 'longwave'
HEATING_RATES = 'heating_rates'
FLUXES = 'fluxes'
TOA_FLUXES = 'toa_fluxes'
SURFACE_FLUXES = 'surface_fluxes'

INPUT_TYPES = [GLOBALS, LEVELS, LAYERS]
DATA_TYPES = [GLOBALS, LAYERS, LEVELS, OUTPUTS_PRISTINE, OUTPUTS_CLEARSKY]
EXP_TYPES = [PRISTINE, CLEAR_SKY]

SPATIAL_DATA_TYPES = [LAYERS, LEVELS, OUTPUTS_PRISTINE, OUTPUTS_CLEARSKY]
DATA_TYPE_DIMS = {GLOBALS: 82, LEVELS: 4, LAYERS: 45, OUTPUTS_CLEARSKY: 1, OUTPUTS_PRISTINE: 1}

TRAIN_YEARS = list(range(1979, 1991)) + list(range(1994, 2005))
VAL_YEARS = [2005, 2006]
TEST_YEARS = list(range(2007, 2015))
OOD_PRESENT_YEARS = [1991]
OOD_FUTURE_YEARS = [2097, 2098, 2099]
OOD_HISTORIC_YEARS = [1850, 1851, 1852]
ALL_YEARS = TRAIN_YEARS + VAL_YEARS + TEST_YEARS + OOD_PRESENT_YEARS + OOD_FUTURE_YEARS + OOD_HISTORIC_YEARS

DATA_DIR = "ClimART_DATA/"
INPUT_RAW_SUBDIR = "input_raw"

DATA_CREATION_SEED = 7


def get_data_subdirs(data_dir: str) -> Dict[str, str]:
    d = {INPUTS: os.path.join(data_dir, INPUTS),
         OUTPUTS_PRISTINE: os.path.join(data_dir, OUTPUTS_PRISTINE),
         OUTPUTS_CLEARSKY: os.path.join(data_dir, OUTPUTS_CLEARSKY)
         }
    d[PRISTINE] = d[OUTPUTS_PRISTINE]
    d[CLEAR_SKY] = d[OUTPUTS_CLEARSKY]
    return d


def get_metadata(data_dir: str = None):
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, 'META_INFO.json')

    if not os.path.isfile(path):
        err_msg = f' Not able to recover meta information from {path}'
        raise ValueError(err_msg)
    with open(path, 'r') as fp:
        meta_info = json.load(fp)
    return meta_info


def get_statistics(data_dir: str = None):
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, 'statistics.npz')
    if not os.path.isfile(path):
        err_msg = f' Not able to recover statistics file from {path}'
        raise ValueError(err_msg)
    statistics = np.load(path)
    return statistics


def get_coordinates(data_dir: str = None):
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, 'areacella_fx_CanESM5.nc')
    if not os.path.isfile(path):
        err_msg = f' Not able to recover coordinates/latitudes/longitudes from {path}'
        raise ValueError(err_msg)
    return xarray.open_dataset(path)


def get_data_dims(exp_type: str) -> (Dict[str, int], Dict[str, int]):
    spatial_dim = {GLOBALS: 0, LEVELS: 50, LAYERS: 49}
    in_dim = {GLOBALS: 82, LEVELS: 4, LAYERS: 14 if exp_type.lower() == PRISTINE else 45}
    out_dim = 100
    return {'spatial_dim': spatial_dim, 'input_dim': in_dim, 'output_dim': out_dim}


def get_flux_mean():
    import numpy
    return numpy.array([
        296.68795572, 295.59927749, 295.1482046, 294.64596736, 294.11837758,
        293.58230163, 293.04465391, 292.50202651, 291.93796111, 291.40537197,
        290.73892157, 290.04539078, 289.39613274, 288.70448136, 288.01931166,
        287.30607635, 286.62727984, 285.93099547, 285.23425452, 284.56900363,
        283.87613833, 283.12472721, 282.30417958, 281.35523027, 280.2217935,
        278.83756356, 277.10443911, 274.98218953, 272.46624459, 269.56133595,
        266.19161612, 262.41764801, 258.2832474, 254.0212223, 250.09734285,
        246.61271747, 243.56633538, 240.91876951, 238.62519398, 236.65407597,
        234.96650554, 233.53089125, 232.3124631, 231.20520435, 230.09577872,
        228.99557943, 227.9148505, 226.85982104, 226.01209087, 225.32944844,

        59.0339687, 59.04093876, 59.03547336, 59.02984351, 59.0244958,
        59.02004596, 59.01671905, 59.0139414, 59.01162252, 59.00892333,
        59.00462615, 58.99430663, 58.97929168, 58.94997202, 58.90542686,
        58.83393261, 58.73876127, 58.60566996, 58.42574877, 58.2087681,
        57.94035763, 57.6132036, 57.23794982, 56.81170941, 56.32479326,
        55.77942549, 55.16748318, 54.49360675, 53.77000922, 52.99446721,
        52.16208255, 51.29397739, 50.39621889, 49.53103161, 48.77409974,
        48.12633895, 47.58119014, 47.12617189, 46.74971375, 46.44106945,
        46.19017345, 45.98759973, 45.82464207, 45.68331371, 45.54797404,
        45.41889672, 45.29627039, 45.18034025, 45.09040981, 45.01957376
    ], dtype=numpy.float64)
