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

plt.rcParams['figure.figsize'] = [12, 6]  # general matplotlib parameters
plt.rcParams['figure.dpi'] = 100

topk = 3
save_dir = "./"
exp_type = "pristine"
ADD_OOD_STAT = False
prefix = 'SURFACE'
prefix = "FLUX"
metric = 'MBE'

TAGS = ["mlp", "gcn", "gn", "cnn", "lgcn"]

cmaps = RollingCmaps(TAGS, max_key_occurence=topk)
markers = RollingLineFormats(TAGS)

xaxis_labels = ["2007-2014"]
xaxis_labels += ['1991 (OOD)'] + ['1850-52', '2097-99'] if ADD_OOD_STAT else ['1850-52', '2097-99']
xaxis = list(range(len(xaxis_labels)))

cols_to_avg = [f"Final/Test_SW_{prefix}_{metric}", f'Final/Future_SW_{prefix}_{metric}', f'Final/Historic_SW_{prefix}_{metric}']
cols_to_avg_cnn = cols_to_avg.copy()
if ADD_OOD_STAT:
    cols_to_avg += [f"Final/Test_OOD_{metric}"]
    cols_to_avg_cnn += [f"Final/Test_1991_SW_{prefix}_{metric}"]

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
    tagg = f'{tag}-base'
    model_stats = runs_df[runs_df['tags'].str.contains(tagg)]
    n_seeds = model_stats.seed.nunique()

    mdl_name = get_model_name(model_stats.iloc[0]['model'])
    if tag == 'lgcn':
        mdl_name = 'L-GCN'
    if mdl_name == 'CNN':
        avg_values = model_stats[cols_to_avg_cnn].mean()
        stds = model_stats[cols_to_avg_cnn].std()
        stats = [avg_values[f'Final/Test_SW_{prefix}_{metric}']]
        stat_stds = [stds[f"Final/Test_SW_{prefix}_{metric}"]]
        if ADD_OOD_STAT:
            stats += [avg_values[f"Final/Test_1991_SW_{prefix}_{metric}"]]
            stat_stds += [stds[f"Final/Test_1991_SW_{prefix}_{metric}"]]
    else:
        avg_values = model_stats[cols_to_avg].mean()
        stds = model_stats[cols_to_avg].std()
        stats = [avg_values[f"Final/Test_SW_{prefix}_{metric}"]]
        stat_stds = [stds[f"Final/Test_SW_{prefix}_{metric}"]]
        if ADD_OOD_STAT:
            stats += [avg_values[f"Final/Test_OOD_{metric}"]]
            stat_stds += [stds[f"Final/Test_OOD_{metric}"]]

    stats += [avg_values[f"Final/Historic_SW_{prefix}_{metric}"]]
    stat_stds += [stds[f"Final/Historic_SW_{prefix}_{metric}"]]
    stats += [avg_values[f"Final/Future_SW_{prefix}_{metric}"]]
    stat_stds += [stds[f"Final/Future_SW_{prefix}_{metric}"]]

    label = f"{mdl_name}"
    yearly_std = np.array(stat_stds)
    yearly_stats = np.array(stats)
    # ax.plot(xaxis, yearly_stats, label=label, c=cmaps[model], marker='x')
    line_format, kwargs = markers[tag]
    ax.errorbar(xaxis, yearly_stats, yerr=yearly_std, fmt=line_format, label=label, **kwargs)
    alpha = 0.1 # if metric == "MBE" else 0.1
    if mdl_name != 'GCN':
        ax.fill_between(xaxis, yearly_stats - yearly_std, yearly_stats + yearly_std, color=kwargs['c'], alpha=alpha)

if metric == "MBE":
    ax.plot(xaxis, np.zeros(len(xaxis)), 'r--')

save_to = f"model_{prefix}_{metric}_OOD.png"
vname = "Avg. "
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
    xticks=xaxis, xtick_labels=xaxis_labels,
    xlabel_fontsize=14, ylabel_fontsize=18,
    xticks_fontsize=18, yticks_fontsize=15,
    full_screen=False,
    show=True, legend=True, legend_loc=2 if metric == "MBE" else 'best', legend_prop=20,
    grid=True, save_to=os.path.join(save_dir, save_to)
)
