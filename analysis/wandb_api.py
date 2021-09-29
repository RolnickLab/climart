from typing import Union, Callable, List

import wandb
import pandas as pd

DF_MAPPING = Callable[[pd.DataFrame], pd.DataFrame]

exp_to_wandb_project = {
    'pristine': "ClimART",
    'clear_sky': "ClimART"
}


# Pre-filters
def has_finished(run) -> bool:
    return run.state == "finished"


def has_final_metric(run) -> bool:
    return 'Final/Test_MAE' in run.summary.keys()


def has_max_metric_value(metric: str = 'Final/Test_MAE', max_metric_value: float = 1.0) -> Callable:
    return lambda run: run.summary[metric] <= max_metric_value


def has_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag in run.tags for tag in tags])


def hasnt_tags(tags: Union[str, List[str]]) -> Callable:
    if isinstance(tags, str):
        tags = [tags]
    return lambda run: all([tag not in run.tags for tag in tags])


def has_hyperparam_values(**kwargs) -> Callable:
    return lambda run: all(hasattr(run.config, hyperparam) and value == run.config[hyperparam]
                           for hyperparam, value in kwargs.items())


def larger_than(**kwargs) -> Callable:
    return lambda run: all(hasattr(run.config, hyperparam) and value > run.config[hyperparam]
                           for hyperparam, value in kwargs.items())


def lower_than(**kwargs) -> Callable:
    return lambda run: all(hasattr(run.config, hyperparam) and value < run.config[hyperparam]
                           for hyperparam, value in kwargs.items())


def df_larger_than(**kwargs) -> DF_MAPPING:
    def f(df) -> pd.DataFrame:
        for k, v in kwargs.items():
            df = df.loc[getattr(df, k) > v]
        return df

    return f


def df_lower_than(**kwargs) -> DF_MAPPING:
    def f(df) -> pd.DataFrame:
        for k, v in kwargs.items():
            df = df.loc[getattr(df, k) < v]
        return df

    return f


def is_model_type(model: str) -> Callable:
    return lambda run: model.lower() in run.config['model'].lower()


str_to_run_pre_filter = {
    'has_finished': has_finished,
    'has_final_metric': has_final_metric
}


# Post-filters
def topk_runs(k: int = 5,
              metric: str = 'Final/Test_MAE',
              lower_is_better: bool = True) -> DF_MAPPING:
    if lower_is_better:
        return lambda df: df.nsmallest(k, metric)
    else:
        return lambda df: df.nlargest(k, metric)


def topk_run_of_each_model_type(k: int = 1,
                                metric: str = 'Final/Test_MAE',
                                lower_is_better: bool = True) -> DF_MAPPING:
    topk_filter = topk_runs(k, metric, lower_is_better)

    def topk_runs_per_model(df: pd.DataFrame) -> pd.DataFrame:
        models = df.model.unique()
        dfs = []
        for model in models:
            dfs += [topk_filter(df[df.model == model])]
        return pd.concat(dfs)

    return topk_runs_per_model


def flatten_column_dicts(df: pd.DataFrame) -> pd.DataFrame:
    types = df.dtypes
    df = pd.concat([df.drop(['preprocessing_dict'], axis=1), df['preprocessing_dict'].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['spatial_dim'], axis=1), df['spatial_dim'].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['input_dim'], axis=1), df['input_dim'].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['target_variable'], axis=1), df['target_variable'].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['target_type'], axis=1), df['target_type'].apply(pd.Series)], axis=1)
    df = pd.concat([df.drop(['hidden_dims'], axis=1), df['hidden_dims'].apply(tuple)], axis=1)
    if 'channels_list' in df.columns:
        df = df.drop('channels_list', axis=1)
        # df = pd.concat([df.drop(['channels_list'], axis=1), df['channels_list'].apply(tuple)], axis=1)

    # df['channels_list'] = df['channels_list'].apply(frozenset)
    for col, dtype in types.items():
        if dtype == dict and dtype != object:
            df = pd.concat([df.drop([col], axis=1), df[col].apply(pd.Series)], axis=1)

    return df


def non_unique_cols_dropper(df: pd.DataFrame) -> pd.DataFrame:
    nunique = df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df = df.drop(cols_to_drop, axis=1)
    return df


def groupby(df: pd.DataFrame, group_by='seed', metric='Test/MAE'):
    grouped_df = df.groupby(group_by)
    stats = grouped_df[[metric, 'name']].mean()
    stats['std'] = grouped_df[[metric, 'name']].std()
    return stats


str_to_run_post_filter = {
    **{
        f"top{k}": topk_runs(k=k)
        for k in range(1, 21)
    },
    'best_per_model': topk_run_of_each_model_type(k=1),
    **{
        f'top{k}_per_model': topk_run_of_each_model_type(k=k)
        for k in range(1, 6)
    },
    'unique_columns': non_unique_cols_dropper,
    'flatten_dicts': flatten_column_dicts
}


def get_runs_df(
        get_metrics: bool = True,
        run_pre_filters: Union[str, List[Union[Callable, str]]] = 'has_finished',
        run_post_filters: Union[str, List[Union[DF_MAPPING, str]]] = None,
        exp_type: str = 'pristine', verbose: bool = False
) -> pd.DataFrame:
    if run_pre_filters is None:
        run_pre_filters = []
    elif not isinstance(run_pre_filters, list):
        run_pre_filters: List[Union[Callable, str]] = [run_pre_filters]
    run_pre_filters = [(f if callable(f) else str_to_run_pre_filter[f.lower()]) for f in run_pre_filters]
    if run_post_filters is None:
        run_post_filters = []
    elif not isinstance(run_post_filters, list):
        run_post_filters: List[Union[Callable, str]] = [run_post_filters]
    run_post_filters = [(f if callable(f) else str_to_run_post_filter[f.lower()]) for f in run_post_filters]

    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(f"ecc-mila7/{exp_to_wandb_project[exp_type]}")
    summary_list = []
    config_list = []
    group_list = []
    name_list = []
    tag_list = []
    id_list = []
    for run in runs:
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files
        if 'model' not in run.config.keys():
            if verbose:
                print(f"Run {run.config['wandb_name'] if 'wandb_name' in run.config else run} filtered out, I.")
            continue

        def filter_out():
            for filtr in run_pre_filters:
                if not filtr(run):
                    if verbose:
                        print(f"Run {run.config['wandb_name']} filtered out, by {filtr.__qualname__}.")
                    return False
            return True

        b = filter_out()
        if not b:
            continue

        id_list.append(str(run.id))
        tag_list.append(str(run.tags))
        if get_metrics:
            summary_list.append(run.summary._json_dict)
        # run.config is the input metrics.
        config_list.append(run.config)

        # run.name is the name of the run.
        name_list.append(run.name)
        group_list.append(run.group)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({'name': name_list, 'id': id_list, 'tags': tag_list})
    group_df = pd.DataFrame({'group': group_list})
    all_df = pd.concat([name_df, config_df, summary_df, group_df], axis=1)

    cols = [c for c in all_df.columns if not c.startswith('gradients/') and c != 'graph_0']
    all_df = all_df[cols]
    if all_df.empty:
        raise ValueError('Empty DF!')
    for post_filter in run_post_filters:
        all_df = post_filter(all_df)
    return all_df
