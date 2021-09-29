import os
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt

from analysis.clean_naming import get_model_name
from analysis.wandb_api import get_runs_df, has_tags, topk_run_of_each_model_type, hasnt_tags, has_hyperparam_values, \
    larger_than, df_larger_than
from climart.data_wrangling.constants import TEST_YEARS
from climart.utils.plotting import set_labels_and_ticks, RollingCmaps, RollingLineFormats

topk = 3
save_dir = "./"
exp_type = "pristine"
ADD_OOD_STAT = True
var = "SW"
prefix = 'SURFACE'
# prefix = "TOA"
prefix = "FLUX"
metric = 'RMSE'

TAGS = ["mlp", "gcn", "gn", "cnn", "lgcn"]

cmaps = RollingCmaps(TAGS, max_key_occurence=topk)
markers = RollingLineFormats(TAGS, linewidth=3 if metric == 'MBE' else 4)

if ADD_OOD_STAT:
    xaxis_labels = ['1991 (OOD)'] + TEST_YEARS
    xaxis_main = list(range(1, len(xaxis_labels)))
else:
    xaxis_labels = TEST_YEARS
    xaxis_main = list(range(len(xaxis_labels)))
xaxis_full = list(range(len(xaxis_labels)))

# cols_to_avg = [f'Final/Test_{metric}'] + [f"Final_yearly/Test_{year}_{metric}" for year in TEST_YEARS]
cols_to_avg = [f'Final/Test_{var}_{prefix}_{metric}'] + [f"Final_yearly/Test_{year}_{var}_{prefix}_{metric}" for year in TEST_YEARS]
cols_to_avg_cnn = [f"Final/Test_{var}_{prefix}_{metric}"]
cols_to_avg_cnn += [f"Final_YEAR/Test_{year}_{var}_{prefix}_{metric}" for year in TEST_YEARS]

if ADD_OOD_STAT:
    cols_to_avg += [f"Final/Test_OOD_{var}_{prefix}_{metric}"]
    cols_to_avg_cnn += [f"Final/Test_1991_{var}_{prefix}_{metric}"]

run_post_filters = [df_larger_than(epoch=90)]
# run_pre_filters = [has_tags(tags=[f'{tag}-base', 'random-train-batches'])]
run_pre_filters = [has_tags(tags=['random-train-batches'])]
runs_df = get_runs_df(
    exp_type=exp_type,
    run_pre_filters=run_pre_filters,
    run_post_filters=run_post_filters
)
runs_df = runs_df[runs_df.exp_type == exp_type]

fig, ax = plt.subplots(1)
for tag in TAGS:
    tagg = f'{tag}-base' if tag != 'lgcn' else 'LGCNB'
    model_stats = runs_df[runs_df['tags'].str.replace('lgcn-base', 'LGCNB').str.contains(tagg)]
    assert len(model_stats) <= 3, f"{tag}, {model_stats[['id', 'tags']]}"
    n_seeds = model_stats.seed.nunique()

    mdl_name = get_model_name(model_stats.iloc[0]['model'])
    stats, stat_stds = [], []
    if tag == 'lgcn':
        mdl_name = 'L-GCN'

    if mdl_name == 'CNN':
        avg_values = model_stats[cols_to_avg_cnn].mean()
        stds = model_stats[cols_to_avg_cnn].std()
        if ADD_OOD_STAT:
            ood_stat = avg_values[f"Final/Test_1991_{var}_{prefix}_{metric}"]
            ood_std = stds[f"Final/Test_1991_{var}_{prefix}_{metric}"]
        stats += [avg_values[f"Final_YEAR/Test_{year}_{var}_{prefix}_{metric}"] for year in TEST_YEARS]
        stat_stds += [stds[f"Final_YEAR/Test_{year}_{var}_{prefix}_{metric}"] for year in TEST_YEARS]

    else:
        avg_values = model_stats[cols_to_avg].mean()
        stds = model_stats[cols_to_avg].std()
        if ADD_OOD_STAT:
            ood_stat = avg_values[f"Final/Test_OOD_{var}_{prefix}_{metric}"]
            ood_std = stds[f"Final/Test_OOD_{var}_{prefix}_{metric}"]
        stats += [avg_values[f"Final_yearly/Test_{year}_{var}_{prefix}_{metric}"] for year in TEST_YEARS]
        stat_stds += [stds[f"Final_yearly/Test_{year}_{var}_{prefix}_{metric}"] for year in TEST_YEARS]

    avg = avg_values[f'Final/Test_{var}_{prefix}_{metric}']
    label = f"{mdl_name} (#seed={n_seeds})"
    label = f"{mdl_name}"
    yearly_std = np.array(stat_stds)
    yearly_stats = np.array(stats)
    # ax.plot(xaxis, yearly_stats, label=label, c=cmaps[model], marker='x')
    line_format, kwargs = markers[tag]
    alpha = 0.05 if metric == "MBE" else 0.1
    if ADD_OOD_STAT:
        ax.errorbar(xaxis_full[:2], [ood_stat, yearly_stats[0]], yerr=ood_std, fmt='--', **kwargs)
        da1, da2 = ood_stat - ood_std, ood_stat + ood_std
        db1, db2 = yearly_stats[0] - yearly_std[0], yearly_stats[0] + yearly_std[0]
        ax.fill_between(xaxis_full[:2], [da1, db1], [da2, db2], color=kwargs['c'], alpha=alpha)

    ax.errorbar(xaxis_main, yearly_stats, yerr=yearly_std, fmt=line_format, label=label, **kwargs)
    ax.fill_between(xaxis_main, yearly_stats - yearly_std, yearly_stats + yearly_std, color=kwargs['c'], alpha=alpha)

if metric == "MBE":
    ax.plot(xaxis_full, np.zeros(len(xaxis_full)), 'r:')

save_to = f"model_{var}_{prefix}_{metric}_per_year.png"
vname = "Upwelling flux" if var == "RSUC" else "Downwelling flux " if var == "RSDC" else "Avg. "
if prefix == 'FLUX':
    ylabel = f'{vname}{metric}'
elif prefix == 'TOA':
    ylabel = f'TOA {vname}{metric}'
elif prefix == 'SURFACE':
    ylabel = f'Surface {vname}{metric}'
else:
    raise ValueError()

set_labels_and_ticks(
    ax,
    xlabel='Test year', ylabel=ylabel,
    xticks=xaxis_full, xtick_labels=xaxis_labels,
    xlabel_fontsize=14, ylabel_fontsize=18,
    xticks_fontsize=18, yticks_fontsize=15,
    full_screen=True,
    show=True, legend=True, legend_loc=2 if metric == "MBE" else 'best',
    grid=True, save_to=os.path.join(save_dir, save_to)
)
