"""
Author: Salva Rühling Cachay
"""

import matplotlib
import numpy as np
import wandb
from climart.utils.plotting import level_errors, height_errors, profile_errors


def log_epoch_vals(writer, loss, epoch, val_stats=None, test_stats=None):
    if writer is None:
        return
    writer.add_scalar('train/_loss', loss, epoch)
    # writer.add_scalar('train/_mae', stats['mae'], epoch)
    if val_stats is not None:
        writer.add_scalar('val/_mae', val_stats['mae'], epoch)
        writer.add_scalar('val/_rmse', val_stats['rmse'], epoch)
        writer.add_scalar('val/_corrcoef', val_stats['corrcoef'], epoch)
    writer.add_scalar('test/_mae', test_stats['mae'], epoch)
    writer.add_scalar('test/_rmse', test_stats['rmse'], epoch)
    writer.add_scalar('test/_corrcoef', test_stats['corrcoef'], epoch)


def set_if_exists(dictio_from, dictio_to, key, prefix: str = ""):
    if key in dictio_from:
        dictio_to[f'{prefix}_{key}'.lstrip('_')] = dictio_from[key]


def update_tqdm(tq, logging_dict: dict, **kwargs):
    tq_view_dict = dict(
        train_loss=logging_dict['Train/loss'],
        time=logging_dict['time/train']
    )
    set_if_exists(logging_dict, tq_view_dict, 'Val/MAE')
    set_if_exists(logging_dict, tq_view_dict, 'Test/MAE')
    set_if_exists(logging_dict, tq_view_dict, 'Test/MBE')

    tq.set_postfix(**tq_view_dict, **kwargs)


def dataset_split_wandb_dict(
        data_split: str, statistics: dict, stats_to_save=None, exclude=None, prefix=""
) -> dict:
    if stats_to_save is None or (isinstance(stats_to_save, str) and stats_to_save.lower() == 'default'):
        SAVED_STATS = ['mae', 'mbe', 'rmse']
    elif isinstance(stats_to_save, list):
        SAVED_STATS = stats_to_save
    elif isinstance(stats_to_save, str) and stats_to_save.lower() == 'all':
        SAVED_STATS = statistics.keys()
    else:
        raise ValueError()
    prefix = prefix if isinstance(prefix, str) else ""

    dset_split_dict = dict()
    delim = '_' if '/' in prefix or '/' in data_split else '/'
    for metric in SAVED_STATS:
        if exclude is not None and exclude.lower() in metric.split(' '):
            continue
        dset_split_dict[f"{prefix}{data_split}{delim}{metric.upper()}"] = statistics[metric]
    return dset_split_dict


def height_error_plots_wandb_dict(Ytrue, preds, data_split: str, height_ticks=None, ylabel='Height'):
    data_split = data_split.capitalize()
    kwargs = {'height_ticks': height_ticks, 'ylabel': ylabel, 'show': False, 'fill_between': False}
    if isinstance(Ytrue, np.ndarray):
        figMAE, figMBE = height_errors(Ytrue, preds, xlabel='', **kwargs)
        return {f"{data_split} MBE": figMBE, f"{data_split} MAE": figMAE}
    else:
        figure_dict = {}
        for var_name in Ytrue.keys():
            figMAE, figMBE = height_errors(Ytrue[var_name], preds[var_name], xlabel=var_name, **kwargs)
            figure_dict[f"{data_split} {var_name} MBE"] = wandb.Image(figMBE)
            figure_dict[f"{data_split} {var_name} MAE"] = wandb.Image(figMAE)
            matplotlib.pyplot.close('all')

        return figure_dict


def toa_level_errors_wandb_dict(Ytrue, preds, data_split: str, epoch):
    data_split = data_split.capitalize()

    figure_dict = {}
    for var_name in Ytrue.keys():
        fig = level_errors(Ytrue[var_name], preds[var_name], epoch)
        figure_dict[f"{data_split} {var_name} level-wise error"] = wandb.Image(fig)

        # fig.savefig(f'/Users/Venky/Documents/rad_tran/Radiative_transfer_dl/{var_name}.png')
        matplotlib.pyplot.close('all')

    return figure_dict


def toa_profile_plots_wandb_dict(Ytrue, preds, data_split: str, plot_type='scatter', title=""):
    data_split = data_split.capitalize()

    figure_dict = {}
    kwargs = {'plot_profiles': 200, 'error_type': 'mean', 'plot_type': plot_type, 'set_seed': plot_type == 'scatter'}
    for var_name in Ytrue.keys():
        fig = profile_errors(Ytrue[var_name], preds[var_name], var_name=var_name, title=title, **kwargs)
        figure_dict[f"{data_split} {var_name} mean  profile error"] = wandb.Image(fig)

        # fig.savefig(f'/Users/Venky/Documents/rad_tran/Radiative_transfer_dl/{var_name}.png')
        matplotlib.pyplot.close('all')

    return figure_dict


def prediction_magnitudes_wandb_dict(preds, data_split: str, prefix: str = "", custom_hist=False,
                                     surface=False, toa=False):
    data_split = data_split.capitalize()

    figure_dict = {}
    if custom_hist:
        for var_name in preds.keys():
            data = preds[var_name]
            n_levels = data.shape[1]
            column_mean = np.mean(data, axis=1, keepdims=True)
            data = np.concatenate((column_mean, data), axis=1)
            preds_table = wandb.Table(data=data, columns=['Mean', 'TOA'] +
                                                         [f'Level {n_levels - i}' for i in range(1, n_levels - 1)] +
                                                         ["Surface"])

            fig_name1 = f"{prefix}{data_split} {var_name} preds"
            fig_name2 = f"{prefix}{data_split} {var_name} surface preds"
            fig_name3 = f"{prefix}{data_split} {var_name} TOA preds"
            figure_dict[fig_name1] = wandb.plot.histogram(preds_table, 'Mean', title=fig_name1)
            if surface:
                figure_dict[fig_name2] = wandb.plot.histogram(preds_table, 'Surface', title=fig_name2)
            if toa:
                figure_dict[fig_name3] = wandb.plot.histogram(preds_table, 'TOA', title=fig_name3)
    else:
        for var_name in preds.keys():
            figure_dict[f"{prefix}{data_split} {var_name} preds"] = wandb.Histogram(np.mean(preds[var_name], axis=1))
            if surface:
                figure_dict[f"{prefix}{data_split} {var_name} surface preds"] = wandb.Histogram(preds[var_name][:, -1])
            if toa:
                figure_dict[f"{prefix}{data_split} {var_name} TOA preds"] = wandb.Histogram(preds[var_name][:, 0])

        # fig.savefig(f'/Users/Venky/Documents/rad_tran/Radiative_transfer_dl/{var_name}.png')
    return figure_dict


def log_true_magnitudes(Y_true, wandb_project, data_split='Test'):
    run = wandb.init(
        project=wandb_project,
        settings=wandb.Settings(start_method='fork'),
        entity="ecc-mila7",
        name='True test',
    )
    logging_dict = {
        **prediction_magnitudes_wandb_dict(Y_true, data_split=data_split, prefix='Epoch 10 ', custom_hist=True,
                                           toa=False, surface=False),
        **prediction_magnitudes_wandb_dict(Y_true, data_split=data_split, prefix='Epoch 50 ', custom_hist=True,
                                           toa=False, surface=False),
        **prediction_magnitudes_wandb_dict(Y_true, data_split=data_split, prefix='Epoch 100 ', custom_hist=True,
                                           toa=False, surface=False),

        **prediction_magnitudes_wandb_dict(Y_true, data_split=data_split, prefix='Final ', custom_hist=True, toa=True,
                                           surface=True)
    }
    wandb.log(logging_dict)
    run.finish()
