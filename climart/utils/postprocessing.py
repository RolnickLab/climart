from typing import List

import wandb
import os

from functools import partial
import torch
import xarray as xr
import numpy as np
from climart.models.interface import get_model, is_gnn, is_graph_net, get_input_transform
from climart.models.column_handler import ColumnPreprocesser
from climart.data_wrangling.constants import LEVELS, LAYERS, GLOBALS, PRISTINE, get_data_dims, get_metadata, \
    get_coordinates
from climart.data_wrangling.h5_dataset import ClimART_HdF5_Dataset
from climart.utils.utils import year_string_to_list


def get_lat_lon(data_dir: str = None):
    coords_data = get_coordinates(data_dir)
    lat = list(coords_data.get_index('lat'))
    lon = list(coords_data.get_index('lon'))

    latitude = []
    longitude = []
    for i in lat:
        for j in lon:
            latitude.append(i)
            longitude.append(j)
    lat_var = np.array(latitude)
    lon_var = np.array(longitude)
    return {'latitude': lat, 'longitude': lon, 'latitude_flattened': lat_var, 'longitude_flattened': lon_var}


def get_preds_and_pressure(ckpt_path: str,
                           year: str,
                           device='cuda',
                           batch_size: int = 512,
                           model_dir: str = None,
                           load_h5_into_mem: bool = True
                           ):
    if len(ckpt_path.split('/')) == 1:
        model_dir = model_dir or 'out/'
        ckpt_path = f"{model_dir}/{ckpt_path}.pkl"
    if not os.path.isfile(ckpt_path):
        raise ValueError(f"No checkpoint was found at {ckpt_path}")

    model_ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    params = model_ckpt['hyper_params']
    net_params = model_ckpt['model_params']

    if is_gnn(params['model']) or is_graph_net(params['model']):
        spatial_dim, in_dim = get_data_dims(params['exp_type'])
        cp = ColumnPreprocesser(
            n_layers=spatial_dim[LAYERS], input_dims=in_dim, **params['preprocessing_dict']
        )
        input_transform = cp.get_preprocesser
    else:
        cp = None
        input_transform = partial(get_input_transform, model_class=get_model(params['model'], only_class=True))

    dataset_kwargs = dict(
        exp_type=params['exp_type'],
        target_type=params['target_type'],
        target_variable=params['target_variable'],
        input_transform=input_transform,
        input_normalization=params['in_normalize'],
        spatial_normalization_in=params['spatial_normalization_in'],
        log_scaling=params['log_scaling'],
        load_h5_into_mem=load_h5_into_mem
    )
    dset = ClimART_HdF5_Dataset(years=year_string_to_list(str(year)), name='Eval', output_normalization=None,
                           **dataset_kwargs)
    dloader = torch.utils.data.DataLoader(dset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4)
    output_postprocesser = dset.output_variable_splitter

    d = dset.h5_dsets[0].get_raw_input_data()
    lvl_pressure = d[LEVELS][..., 2]
    lay_pressure = d[LAYERS][..., 2]
    cszrow = d[GLOBALS][..., 0]

    trainer_kwargs = dict(
        model_name=params['model'], model_params=net_params,
        device=params['device'], seed=params['seed'],
        model_dir=params['model_dir'],
        output_postprocesser=output_postprocesser,
    )
    if cp is not None:
        trainer_kwargs['column_preprocesser'] = cp

    trainer = get_trainer(**trainer_kwargs)
    trainer.reload_model(model_state_dict=model_ckpt['model'])
    preds, Y, _ = trainer.evaluate(dloader, verbose=True)

    dset.close()
    return {'preds': preds, 'targets': Y, 'pressure': lvl_pressure, 'layer_pressure': lay_pressure, 'cszrow': cszrow}


# %%

def save_preds_to_netcdf(preds, targets,
                         post_fix: str = '',
                         save_path=None, exp_type='pristine', data_dir: str = None,
                         model=None,
                         **kwargs):
    lat_lon = get_lat_lon(data_dir)
    lat, lon = lat_lon['latitude'], lat_lon['longitude']
    spatial_dim, _ = get_data_dims(exp_type)
    n_levels = spatial_dim[LEVELS]
    n_layers = spatial_dim[LAYERS]
    shape = ['snapshot', 'latitude', 'longitude', 'level']
    shape_lay = ['snapshot', 'latitude', 'longitude', 'layer']
    shape_glob = ['snapshot', 'latitude', 'longitude']

    meta_info = get_metadata(data_dir)

    data_vars = dict()
    for k, v in preds.items():
        data_vars[f"{k}_preds"] = (shape, v.reshape((-1, len(lat), len(lon), n_levels)))
    for k, v in targets.items():
        data_vars[f"{k}_targets"] = (shape, v.reshape((-1, len(lat), len(lon), n_levels)))

    data_vars["pressure"] = (shape, kwargs['pressure'].reshape((-1, len(lat), len(lon), n_levels)))
    data_vars["layer_pressure"] = (shape_lay, kwargs['layer_pressure'].reshape((-1, len(lat), len(lon), n_layers)))
    data_vars["cszrow"] = (shape_glob, kwargs['cszrow'].reshape((-1, len(lat), len(lon))))

    xr_dset = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            longitude=lon,
            latitude=lat,
            level=list(range(n_levels))[::-1],
            layer=list(range(n_layers))[::-1],
        ),
        attrs=dict(description="ML emulated RT outputs."),
    )
    if save_path is not None:
        if not save_path.endswith('.nc'):
            save_path += '.nc'
        save_path.replace('.nc', post_fix + '.nc')

    elif model is not None:
        save_path = f'~/RT-DL/example_{exp_type}_preds_{model}_{post_fix}.nc'
    else:
        print("Not saving to NC!")
        return xr_dset
    if not os.path.isfile(save_path):
        xr_dset.to_netcdf(save_path)
        print('saved to\n', save_path)
    return xr_dset


def restore_run(run_id,
                run_path: str,
                api=None,
                ):
    if api is None:
        api = wandb.Api()
    run_path = f"{run_path}/{run_id}"
    run = api.run(run_path)
    return run


def restore_ckpt_from_wandb_run(run, entity: str = 'ecc-mila7', run_path=None, load: bool = False, **kwargs):
    run_id = run.id
    ckpt = [f for f in run.files() if f"{run_id}.pkl" in str(f)]
    ckpt = str(ckpt[0].name)
    ckpt = wandb.restore(ckpt, run_path=f"{entity}/ClimART/{run_id}")
    ckpt_fname = ckpt.name
    if load:
        return torch.load(ckpt_fname, **kwargs)
    return ckpt_fname


def restore_ckpt_from_wandb(run_id,
                            run_path: str,
                            api=None,
                            load: bool = False,
                            **kwargs):
    run = restore_run(run_id, run_path, api)
    return restore_ckpt_from_wandb(run, load=load, **kwargs)


def restore_and_save_preds_to_netcdf(run_id, run_path: str, years: List[int], device='cuda'):
    if not isinstance(years, list):
        years = [years]
    ckpt_file = restore_ckpt_from_wandb(run_id, run_path)
    exp = 'clear_sky' if 'CS' in ckpt_file else 'pristine'
    for year in years:
        save_path = ckpt_file.replace('.pkl', f'_{year}.nc')
        if os.path.isfile(save_path):
            print('Skipping since already exists.')
            continue
        p_gn = get_preds_and_pressure(ckpt_file, year=str(year), device=device)
        save_preds_to_netcdf(**p_gn, save_path=save_path, post_fix=str(year), exp_type=exp)
