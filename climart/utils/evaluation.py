import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from climart.utils import utils

log = utils.get_logger(__name__)


def evaluate_preds(Ytrue: np.ndarray, preds: np.ndarray, model_name="", verbose=False):
    # corrcoef = np.corrcoef(Ytrue, preds)[0, 1]
    MSE = mean_squared_error(preds, Ytrue)
    RMSE = np.sqrt(MSE)
    MAE = mean_absolute_error(preds, Ytrue)
    MBE = np.mean(preds - Ytrue)
    # r, p = pearsonr(Ytrue, preds)   # same as using np.corrcoef(y, yhat)[0, 1]
    stats = {'mbe': MBE,
             #       'mse': MSE,
             'mae': MAE,
             "rmse": RMSE}  # , "Pearson_r": r, "Pearson_p": p}
    if verbose:
        print(model_name, 'RMSE, MAE, MBE: {:.3f}, {:.3f}, {:.3f}'.format(RMSE, MAE, MBE))
        # print(model_name, 'RMSE, MSE, MAE: {:.3f} {:.3f}, {:.3f}, Corrcoef: {:.3f}'.format(RMSE, MSE, MAE, corrcoef))

    return stats


def evaluate_preds_per_var(Ytrue: dict, preds: dict, model_name="", verbose=False):
    stats = dict()
    if not isinstance(Ytrue, dict):
        log.warning(f" Expected a dictionary var_name->Tensor/nd_array, but got {type(Ytrue)} for Ytrue!")
        return stats

    metrics = ['mbe', 'mae', 'rmse']
    for var_name in Ytrue.keys():
        prefix = f"{model_name}_{var_name.upper()} "
        var_stats = evaluate_preds(Ytrue[var_name], preds[var_name], model_name=prefix + ':',
                                   verbose=verbose)
        # pre-append the variable's name to its speficic performance on the returned metrics dict
        for mn, metric_stat in var_stats.items():
            stats[f"{var_name}_{mn}"] = metric_stat


        for lvl in range(0, 50):
            stats_lvl = evaluate_preds(Ytrue[var_name][:, lvl], preds[var_name][:, lvl], model_name=prefix + f'_{lvl}level:',
                                       verbose=verbose)
            # pre-append the variable's name to its speficic performance on the returned metrics dict
            stats_lvl.pop('mae')
            for mn, metric_stat in stats_lvl.items():
                stats[f"{var_name}_level{lvl}_{mn}"] = metric_stat

            if lvl == 49:
                for mn, metric_stat in stats_lvl.items():
                    stats[f"{var_name}_SURFACE_{mn}"] = metric_stat
            if lvl == 0:
                for mn, metric_stat in stats_lvl.items():
                    stats[f"{var_name}_TOA_{mn}"] = metric_stat

#    keys = Ytrue.keys()
#    if 'rsuc' in keys and 'rsdc' in keys:
#        for mn in metrics:
#            stats[f"SW_flux_{mn}"] = (stats[f"rsuc_{mn}"] + stats[f"rsdc_{mn}"]) / 2
#            stats[f"SW_TOA_{mn}"] = (stats[f"rsuc_TOA_{mn}"] + stats[f"rsdc_TOA_{mn}"]) / 2
#            stats[f"SW_SURFACE_{mn}"] = (stats[f"rsuc_SURFACE_{mn}"] + stats[f"rsdc_SURFACE_{mn}"]) / 2
#            for lvl in range(0, 50):
#                stats[f"SW_level{lvl}_{mn}"] = (stats[f"rsuc_level{lvl}_{mn}"] + stats[f"rsdc_level{lvl}_{mn}"]) / 2
#
#    if 'rluc' in keys and 'rldc' in keys:
#        for mn in metrics:
#            stats[f"LW_flux_{mn}"] = (stats[f"rluc_{mn}"] + stats[f"rldc_{mn}"]) / 2
#            stats[f"LW_TOA_{mn}"] = (stats[f"rluc_TOA_{mn}"] + stats[f"rldc_TOA_{mn}"]) / 2
#            stats[f"LW_SURFACE_{mn}"] = (stats[f"rluc_SURFACE_{mn}"] + stats[f"rldc_SURFACE_{mn}"]) / 2

    return stats
