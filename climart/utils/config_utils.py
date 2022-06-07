import logging
import os
import time
import warnings
from typing import Union, Sequence, List

import omegaconf
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict

from climart.utils.naming import get_group_name, get_detailed_name
from climart.utils.utils import no_op, get_logger


def print_config(
        config,
        fields: Union[str, Sequence[str]] = (
                "datamodule",
                "model",
                "trainer",
                # "callbacks",
                # "logger",
                "seed",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Args:
        config (ConfigDict): Configuration
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    import importlib
    if not importlib.util.find_spec("rich") or not importlib.util.find_spec("omegaconf"):
        # no pretty printing
        return
    import rich.syntax
    import rich.tree

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)
    if isinstance(fields, str):
        if fields.lower() == 'all':
            fields = config.keys()
        else:
            fields = [fields]

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Modifies DictConfig in place.
    """
    log = get_logger()

    # Create working dir if it does not exist yet
    if config.get('work_dir'):
        os.makedirs(name=config.get("work_dir"), exist_ok=True)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    USE_WANDB = "logger" in config.keys() and config.logger.get("wandb")
    if USE_WANDB:
        if not config.logger.wandb.get('id'):  # no wandb id has been assigned yet
            wandb_id = wandb.util.generate_id()
            config.logger.wandb.id = wandb_id
        else:
            log.info(f" This experiment config already has a wandb run ID = {config.logger.wandb.id}")
        if not config.logger.wandb.get('group'):  # no wandb group has been assigned yet
            group_name = get_group_name(config)
            config.logger.wandb.group = group_name if len(group_name) < 128 else group_name[:128]
        if not config.logger.wandb.get('name'):  # no wandb name has been assigned yet
            config.logger.wandb.name = get_detailed_name(config) + '_' + time.strftime(
                '%Hh%Mm_on_%b_%d') + '_' + config.logger.wandb.id

    check_config_values(config)
    if USE_WANDB:
        wandb_kwargs = {
            k: config.logger.wandb.get(k) for k in ['id', 'project', 'entity', 'name', 'group',
                                                    'tags', 'notes', 'reinit', 'mode', 'resume']
        }
        wandb_kwargs['dir'] = config.logger.wandb.get('save_dir')
        wandb.init(**wandb_kwargs)
        log.info(f"Wandb kwargs: {wandb_kwargs}")
        save_hydra_config_to_wandb(config)


def check_config_values(config: DictConfig):
    exp_type = config.datamodule.exp_type.lower()
    config.datamodule.exp_type = exp_type
    if exp_type not in ["clear_sky", "pristine"]:
        raise ValueError(f"Arg `exp_type` should be one of clear_sky or pristine, but got {exp_type}")

    if "net_normalization" in config.model.keys():
        if config.model.net_normalization is None:
            config.model.net_normalization = "none"
        config.model.net_normalization = config.model.net_normalization.lower()

    if config.logger.get("wandb"):
        if 'callbacks' in config and config.callbacks.get('model_checkpoint'):
            id_mdl = config.logger.wandb.get('id')
            d = config.callbacks.model_checkpoint.dirpath
            if id_mdl is not None:
                with open_dict(config):
                    new_dir = os.path.join(d, id_mdl)
                    config.callbacks.model_checkpoint.dirpath = new_dir
                    os.makedirs(new_dir, exist_ok=True)
                    logging.info(f" Model checkpoints will be saved in: {new_dir}")
    else:
        config.save_config_to_wandb = False


def get_all_instantiable_hydra_modules(config, module_name: str):
    from hydra.utils import instantiate as hydra_instantiate
    modules = []
    if module_name in config:
        for _, module_config in config[module_name].items():
            if module_config is not None and "_target_" in module_config:
                modules.append(hydra_instantiate(module_config))
    return modules


def log_hyperparameters(
        config,
        model: pl.LightningModule,
        data_module: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.
    Credits go to: https://github.com/ashleve/lightning-hydra-template

    Additionally saves:
        - number of {total, trainable, non-trainable} model parameters
    """

    def copy_and_ignore_keys(dictionary, *keys_to_ignore):
        new_dict = dict()
        for k in dictionary.keys():
            if k not in keys_to_ignore:
                new_dict[k] = dictionary[k]
        return new_dict

    params = dict()
    if 'seed' in config:
        params['seed'] = config['seed']
    if 'model' in config:
        params['model'] = config['model']

    # Remove redundant keys or those that are not important to know after training -- feel free to edit this!
    params["datamodule"] = copy_and_ignore_keys(config["datamodule"], 'pin_memory', 'num_workers')
    params['model'] = copy_and_ignore_keys(config['model'], 'optimizer', 'scheduler')
    params['normalizer'] = config['normalizer']
    params["trainer"] = copy_and_ignore_keys(config["trainer"])
    # encoder, optims, and scheduler as separate top-level key
    params['optim'] = config['model']['optimizer']
    params['scheduler'] = config['model']['scheduler'] if 'scheduler' in config['model'] else None

    if "callbacks" in config:
        if 'model_checkpoint' in config['callbacks']:
            params["model_checkpoint"] = copy_and_ignore_keys(
                config["callbacks"]['model_checkpoint'], 'save_top_k'
            )

    # save number of model parameters
    params["model/params_total"] = sum(p.numel() for p in model.parameters())
    params["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    params["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    params['dirs/work_dir'] = config.get('work_dir')
    params['dirs/ckpt_dir'] = config.get('ckpt_dir')
    params['dirs/wandb_save_dir'] = config.logger.wandb.save_dir if (
            config.get('logger') and config.logger.get('wandb')) else None

    # send hparams to all loggers
    trainer.logger.log_hyperparams(params)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = no_op


def save_hydra_config_to_wandb(config: DictConfig):
    if config.get('save_config_to_wandb'):
        logging.info(f"Hydra config will be saved to WandB as hydra_config.yaml and in wandb run_dir: {wandb.run.dir}")
        # files in wandb.run.dir folder get directly uploaded to wandb
        with open(os.path.join(wandb.run.dir, "hydra_config.yaml"), "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)
        wandb.save(os.path.join(wandb.run.dir, "hydra_config.yaml"))
    else:
        logging.info(f"Hydra config will NOT be saved to WandB.")


def get_config_from_hydra_compose_overrides(overrides: list) -> DictConfig:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    overrides = list(set(overrides))
    if '-m' in overrides:
        overrides.remove('-m')  # if multiruns flags are mistakenly in overrides
    hydra.initialize(config_path="../../configs")
    try:
        config = hydra.compose(config_name="main_config", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()   # always clean up global hydra
    return config


def get_model_from_hydra_compose_overrides(overrides: list):
    from climart.interface import get_model
    cfg = get_config_from_hydra_compose_overrides(overrides)
    return get_model(cfg)
