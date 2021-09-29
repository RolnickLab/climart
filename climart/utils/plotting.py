import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import einops
import xarray as xr


def set_labels_and_ticks(ax,
                         xlabel: str = "", ylabel: str = "",
                         xlabel_fontsize: int = 10, ylabel_fontsize: int = 14,
                         xlim=None, ylim=None,
                         xticks=None, yticks=None,
                         xticks_fontsize: int = None, yticks_fontsize: int = None,
                         xtick_labels=None, ytick_labels=None,
                         show: bool = True,
                         grid: bool = True,
                         legend: bool = True, legend_loc='best', legend_prop=10,
                         full_screen: bool = False,
                         save_to: str = None
                         ):
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if xticks:
        ax.set_xticks(xticks)
    if xtick_labels:
        ax.set_xticklabels(xtick_labels)
    if xticks_fontsize:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(xticks_fontsize)
        # tick.label.set_rotation('vertical')

    if yticks:
        ax.set_yticks(yticks)
    if ytick_labels:
        ax.set_yticklabels(ytick_labels)
    if yticks_fontsize:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(yticks_fontsize)
    if grid:
        ax.grid()
    if legend:
        ax.legend(loc=legend_loc, prop={'size': legend_prop}) #if full_screen else ax.legend(loc=legend_loc)

    if save_to is not None:
        if full_screen:
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()
        plt.savefig(save_to, bbox_inches='tight')
        if full_screen:
            mng.full_screen_toggle()

    if show:
        plt.show()


class RollingCmaps:
    def __init__(self,
                 unique_keys: list,
                 pos_cmaps: list = None,
                 max_key_occurence: int = 5):
        if pos_cmaps is None:
            pos_cmaps = ['Greens', 'Oranges', 'Blues', 'Greys', 'Purples']
        pos_cmaps = [plt.get_cmap(cmap) for cmap in pos_cmaps]
        self.cmaps = {key: pos_cmaps[i] for i, key in enumerate(unique_keys)}
        self.pos_per_cmap = {key: 0.75 for key in unique_keys}  # lower makes lines too white
        self.max_key_occurence = max_key_occurence

    def __getitem__(self, key):
        color = self.cmaps[key](self.pos_per_cmap[key] / self.max_key_occurence)  # [self.pos_per_cmap[key]]
        self.pos_per_cmap[key] += 1
        return color


class RollingLineFormats:
    def __init__(self,
                 unique_keys: list,
                 pos_markers: list = None,
                 cmap = None,
                 linewidth: int = 4
                 ):
        print(unique_keys)
        if pos_markers is None:
            pos_markers = ['-', '--', ':', '-', '-.']
        if cmap is None:
            cmap = plt.get_cmap('viridis')
        cs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#d62728', '#bcbd22', '#17becf']
        self.pos_markers = pos_markers
        # self.cmaps = {key: cmap(i/len(unique_keys)) for i, key in enumerate(unique_keys)}
        self.cmaps = {key: cs[i] for i, key in enumerate(unique_keys)}
        self.pos_per_key = {key: 0 for key in unique_keys}  # lower makes lines too white
        self.lws = {key: linewidth for key in unique_keys}

    def __getitem__(self, key):
        cur_i = self.pos_per_key[key]
        lw = self.lws[key]
        line_format = self.pos_markers[cur_i]  # [self.pos_per_cmap[key]]
        self.pos_per_key[key] += 1
        self.lws[key] = max(1, lw - 1)
        return line_format, dict(c=self.cmaps[key], linewidth=lw)


def plot_groups(xaxis_key, metric='Test/MAE', ax=None, show: bool = True, **kwargs):
    if not ax:
        fig, ax = plt.subplots()  # 1

    for key, group in kwargs.items():
        group.plot(xaxis_key, metric, yerr='std', label=key, ax=ax)

    set_labels_and_ticks(
        ax, xlabel='Used training points', ylabel=metric, show=show
    )


def height_errors(Ytrue: np.ndarray, preds: np.ndarray, height_ticks=None,
                  xlabel='', ylabel='height', fill_between=True, show=True):
    """
    Plot MAE and MBE as a function of the height/pressure
    :param Ytrue:
    :param preds:
    :param height_ticks: must have same shape as Ytrue.shape[1]
    :param show:
    :return:
    """
    n_samples, n_levels = Ytrue.shape
    diff = Ytrue - preds
    abs_diff = np.abs(diff)
    levelwise_MBE = np.mean(diff, axis=0)
    levelwise_MAE = np.mean(abs_diff, axis=0)

    levelwise_MBE_std = np.std(diff, axis=0)
    levelwise_MAE_std = np.std(abs_diff, axis=0)

    # Plotting
    plotting_kwargs = {'yticks': height_ticks, 'ylabel': ylabel, 'show': show, "fill_between": fill_between}
    yaxis = np.arange(n_levels)
    figMBE = height_plot(yaxis, levelwise_MBE, levelwise_MBE_std, xlabel=xlabel + ' MBE', **plotting_kwargs)
    figMAE = height_plot(yaxis, levelwise_MAE, levelwise_MAE_std, xlabel=xlabel + ' MAE', **plotting_kwargs)

    if show:
        plt.show()
    return figMAE, figMBE


def height_plot(yaxis, line, std, yticks=None, ylabel=None, xlabel=None, show=False, fill_between=True):
    fig, ax = plt.subplots(1)
    if "mbe" in xlabel.lower():
        # to better see the bias
        ax.plot(np.zeros(yaxis.shape), yaxis, '--', color='grey')

    p = ax.plot(line, yaxis, '-', linewidth=3)
    if fill_between:
        ax.fill_betweenx(yaxis, line - std, line + std, alpha=0.2)
    else:
        ax.plot(line - std, yaxis, '--', color=p[0].get_color(), linewidth=1.5)
        ax.plot(line + std, yaxis, '--', color=p[0].get_color(), linewidth=1.5)

    if yticks is not None:
        ax.set_yticks(yaxis)  # yaxis
        ax.yaxis.set_ticklabels(yticks)  # change the ticks' names to yticks

    if xlabel is not None:
        ax.set_xlabel(xlabel.strip())

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if 'mae' in xlabel.lower() or 'rmse' in xlabel.lower():
        ax.set_xlim([0, ax.get_xlim()[1]])

    if show:
        plt.show()
    return fig


def level_errors(Y_true, Y_preds, epoch):
    errors = np.mean((Y_true - Y_preds), axis=0)
    colours = ['red' if x < 0 else 'green' for x in errors]
    index = np.arange(0, len(colours), 1)

    # Draw plot
    lev_fig = plt.figure(figsize=(14, 14), dpi=80)
    plt.hlines(y=index, xmin=0, xmax=errors)
    for x, y, tex in zip(errors, index, errors):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 10})

    # Styling    
    plt.yticks(index, ['Level: ' + str(z) for z in index], fontsize=12)
    plt.title(f'Average Level-wise error for epoch: {epoch}', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(-5, 5)

    return lev_fig


def profile_errors(Y_true, Y_preds, plot_profiles=200, var_name=None,
                   error_type='mean', plot_type='scatter', set_seed=False, title=""):
    coords_data = xr.open_dataset(
        '/miniscratch/venkatesh.ramesh/ECC_data/snapshots/coords_data/areacella_fx_CanESM5_amip_r1i1p1f1_gn.nc'
    )
    lat = list(coords_data.get_index('lat'))
    lon = list(coords_data.get_index('lon'))

    total_profiles, n_levels = Y_true.shape

    if set_seed:  # To get the same profiles everytime
        np.random.seed(7)

    errors = np.abs(Y_true - Y_preds)
    # print(errors.shape, Y_true.shape, total_profiles / 8192)

    if plot_type.lower() == 'scatter':
        latitude = []
        longitude = []

        for i in lat:
            for j in lon:
                latitude.append(i)
                longitude.append(j)

        lat_var = np.array(latitude)
        lon_var = np.array(longitude)

        n_times = int(total_profiles / 8192)
        indices = np.arange(0, total_profiles)
        indices_train = np.random.choice(total_profiles, total_profiles - plot_profiles, replace=False)
        indices_rest = np.setxor1d(indices_train, indices, assume_unique=True)

        lat_var = np.mean(np.vstack([np.expand_dims(lat_var, 1)] * n_times), axis=1)
        lon_var = np.mean(np.vstack([np.expand_dims(lon_var, 1)] * n_times), axis=1)
        lon_plot = lon_var[indices_rest]
        lat_plot = lat_var[indices_rest]
        errors_lev = errors[indices_rest]
        errors_lev = einops.rearrange(np.array(errors_lev), 'p l -> l p')  # p = profile dim
        print(errors.shape, Y_true.shape)  # (81920, 50) (81920, 50)
    else:
        errors_lev = errors.reshape(n_levels, 8192, -1)  # level x spatial_dim x snapshot_dim
        errors_lev = np.mean(errors_lev, axis=2)  # mean over all snapshots
        errors_lev = errors_lev.reshape((n_levels, len(lat), len(lon)))  # reshape back to spatial grid
        lon_plot, lat_plot = np.meshgrid(lon, lat)

    if error_type.lower() == 'toa':
        err = errors_lev[0]
    elif error_type.lower() == 'surface':
        err = errors_lev[-1]
    elif error_type.lower() == 'mean':
        err = np.mean(errors_lev, axis=0)

    pp = profile_plot(lon_plot, lat_plot, err, var_name, plot_type=plot_type, title=title)

    return pp


def profile_plot(lon_plot, lat_plot, errors, var_name=None, plot_type='scatter', dpi=70, title=""):
    """

    :param lon_plot:
    :param lat_plot:
    :param errors: Note that if plot_type is 2D, i.e is in ['heatmap', 'contour'], it has to have shape lat x lon
    :param var_name:
    :param plot_type:
    :param dpi:
    :return:
    """
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['figure.dpi'] = dpi
    plot_type = plot_type.lower()

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.add_feature(cfeature.COASTLINE)
    ax.gridlines()
    # ax.stock_img()
    ax.set_global()

    # jet = plt.cm.get_cmap('RdBu_r')
    jet = plt.cm.get_cmap('plasma')

    # nightshade
    # current_time = datetime.now() Can work well if we can get the UTC time for the data
    # ax.add_feature(Nightshade(current_time, alpha=0.3))

    # Circle params
    fs_text = 10
    padd = -0.18
    stroffset = -0.2
    circlesize = 100
    lw = 2.2

    if plot_type == 'contour':
        sc = ax.contourf(lon_plot, lat_plot, errors, transform=ccrs.PlateCarree(), alpha=0.85, cmap="Reds",
                         levels=100)
    elif plot_type == 'heatmap':
        sc = ax.pcolormesh(lon_plot, lat_plot, errors, cmap="Reds", transform=ccrs.PlateCarree())
    else:
        sc = ax.scatter(x=lon_plot, y=lat_plot, s=circlesize,
                        c=errors, norm=TwoSlopeNorm(5, vmin=0, vmax=10),
                        alpha=0.8, cmap=jet, linewidths=lw,
                        transform=ccrs.PlateCarree())

    ax.set_title(f'{title}{plot_type.upper()}-{var_name.upper()} error', fontsize=fs_text)
    # Colour Bar
    cbar = plt.colorbar(sc, ax=ax, aspect=30, pad=0.01, shrink=0.4, orientation='vertical')
    cbar.ax.set_ylabel('W m$^{-2}$', rotation=270, labelpad=10)
    return fig


def prediction_hist(preds: dict, TOA=False, surface=False,
                    title="", show=True, figsize=(16, 12), axes=None,
                    label="", **kwargs):
    n_vars = len(preds.keys())
    n_cols = 3 if TOA and surface else 2 if (TOA or surface) else 1

    surface_ax = 1
    TOA_ax = 2 if surface else 1

    if axes is None:
        fig, axs = plt.subplots(n_vars, n_cols, figsize=figsize)
        fig.suptitle("Prediction magnitudes" if title == "" else title)
        axs[0, 0].set_title('Mean')

        if surface:
            axs[0, surface_ax].set_title('Surface')
        if TOA:
            axs[0, TOA_ax].set_title('TOA')
    else:
        axs = axes

    def set_bar_colors(patches, upto=5):
        return
        jet = plt.get_cmap('jet', len(patches))
        for i in range(len(patches)):
            if i > upto:
                return
            patches[i].set_facecolor(jet(i * 10))

    for (var_name, var_preds), ax_row in zip(preds.items(), axs):
        # n_samples, n_levels = var_preds.shape
        N, bins, patches = ax_row[0].hist(np.mean(var_preds, axis=1), label=label, **kwargs)
        set_bar_colors(patches)
        ax_row[0].set_ylabel(f"{var_name.upper()}", fontsize=20)
        if surface:
            N, bins, patches = ax_row[surface_ax].hist(var_preds[:, -1], label=label, **kwargs)
            set_bar_colors(patches)
        if TOA:
            N, bins, patches = ax_row[TOA_ax].hist(var_preds[:, 0], label=label, **kwargs)
            set_bar_colors(patches)

    axs[0, 0].legend()
    if show:
        plt.show()

    return axs


def prediction_bars(preds: dict, bins, TOA=False, surface=False,
                    title="", show=True, figsize=(16, 12), axes=None,
                    label="", **kwargs):
    n_vars = len(preds.keys())
    n_cols = 3 if TOA and surface else 2 if (TOA or surface) else 1

    surface_ax = 1
    TOA_ax = 2 if surface else 1

    if axes is None:
        fig, axs = plt.subplots(n_vars, n_cols, figsize=figsize)
        fig.suptitle("Prediction magnitudes" if title == "" else title)
        axs[0, 0].set_title('Mean')

        if surface:
            axs[0, surface_ax].set_title('Surface')
        if TOA:
            axs[0, TOA_ax].set_title('TOA')
    else:
        axs = axes

    for i, ((var_name, var_preds), ax_row) in enumerate(zip(preds.items(), axs)):

        if False:  # i == 1:
            kwargs['tick_label'] = ['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)]

        hist, bin_edges = np.histogram(np.mean(var_preds, axis=1), bins)
        ax_row[0].bar(range(len(hist)), hist, width=1, align='center', label=label, **kwargs)
        ax_row[0].set_ylabel(f"{var_name.upper()}", fontsize=20)
        if surface:
            hist, bin_edges = np.histogram(var_preds[:, -1], bins)
            ax_row[surface_ax].bar(range(len(hist)), hist, width=1, align='center', label=label, **kwargs)
        if TOA:
            hist, bin_edges = np.histogram(var_preds[:, 0], bins)
            ax_row[TOA_ax].bar(range(len(hist)), hist, width=1, align='center', label=label, **kwargs)

    axs[0, 0].legend()
    if show:
        plt.show()

    return axs
