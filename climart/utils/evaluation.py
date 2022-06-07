from typing import Dict

import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error
from climart.data_loading.constants import get_statistics
from climart.utils import utils

log = utils.get_logger(__name__)


def evaluate_preds(Ytrue: np.ndarray, preds: np.ndarray):
    MSE = np.mean((preds - Ytrue) ** 2)  # mean_squared_error(preds, Ytrue)
    RMSE = np.sqrt(MSE)
    # MAE = mean_absolute_error(preds, Ytrue)
    MBE = np.mean(preds - Ytrue)
    stats = {'mbe': MBE,
             # 'mse': MSE,
             # 'mae': MAE,
             "rmse": RMSE}

    return stats


def evaluate_per_target_variable(Ytrue: dict,
                                 preds: dict,
                                 data_split: str = None) -> Dict[str, float]:
    stats = dict()
    if not isinstance(Ytrue, dict):
        log.warning(f" Expected a dictionary var_name->Tensor/nd_array, but got {type(Ytrue)} for Ytrue!")
        return stats

    for var_name in Ytrue.keys():
        # var_name stands for 'rsuc', 'hrsc', etc., i.e. shortwave upwelling flux, shortwave heating rate, etc.
        var_stats = evaluate_preds(Ytrue[var_name], preds[var_name])
        # pre-append the variable's name to its specific performance on the returned metrics dict
        for metric_name, metric_stat in var_stats.items():
            stats[f"{data_split}/{var_name}/{metric_name}"] = metric_stat

        num_height_levels = Ytrue[var_name].shape[1]
        for lvl in range(0, num_height_levels):
            stats_lvl = evaluate_preds(Ytrue[var_name][:, lvl], preds[var_name][:, lvl])
            for metric_name, metric_stat in stats_lvl.items():
                stats[f"levelwise/{data_split}/{var_name}_level{lvl}/{metric_name}"] = metric_stat

            if lvl == num_height_levels - 1:
                for metric_name, metric_stat in stats_lvl.items():
                    stats[f"{data_split}/{var_name}_surface/{metric_name}"] = metric_stat
            if lvl == 0:
                for metric_name, metric_stat in stats_lvl.items():
                    stats[f"{data_split}/{var_name}_toa/{metric_name}"] = metric_stat

    return stats
