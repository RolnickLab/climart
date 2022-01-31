import os
import wandb
from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

import climart.utils.utils as utils
from climart.data_transform.normalization import Normalizer
from climart.data_transform.transforms import AbstractTransform
from climart.datamodules.pl_climart_datamodule import ClimartDataModule
from climart.models.base_model import BaseModel


def run_model(config: DictConfig):
    seed_everything(config.seed, workers=True)
    log = utils.get_logger(__name__)
    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config, fields='all')

    # First we instantiate our normalization preprocesser, then our model
    normalizer: Normalizer = hydra_instantiate(config.normalizer, datamodule_config=config.datamodule,
                                               _recursive_=False)
    if config.model.get("input_transform"):
        input_transform: AbstractTransform = hydra_instantiate(config.model.input_transform)
    else:
        input_transform = None
    data_module: ClimartDataModule = hydra_instantiate(config.datamodule, input_transform=input_transform,
                                                       normalizer=normalizer)
    emulator_model: BaseModel = hydra_instantiate(
        config.model, _recursive_=False,
        datamodule_config=config.datamodule,
        output_postprocesser=normalizer.output_variable_splitter,
        output_normalizer=normalizer.output_normalizer
    )

    # Then, with all the convenience and ease of PyTorch Lightning,
    # we can train our model on the DataModule from above (checkpointing the best model w.r.t. a small validation set),
    # and passing any callbacks you fancy to the Trainer.
    # Init Lightning callbacks and loggers
    callbacks = utils.get_all_instantiable_hydra_modules(config, 'callbacks')
    loggers = utils.get_all_instantiable_hydra_modules(config, 'logger')
    # wandb_logger = [l for l in loggers if isinstance(l, WandbLogger)][0]

    # Init Lightning trainer
    trainer: pl.Trainer = hydra_instantiate(
        config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"  # , deterministic=True
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters to the PyTorch Lightning loggers.")
    utils.log_hyperparameters(config=config, model=emulator_model, data_module=data_module, trainer=trainer,
                              callbacks=callbacks)

    trainer.fit(model=emulator_model, datamodule=data_module)

    # Testing:
    trainer.test(datamodule=data_module, ckpt_path='best')

    if config.get('save_model_to_wandb'):
        path = trainer.checkpoint_callback.best_model_path
        log.info(f"Best checkpoint path will be saved to wandb from path: {path}")
        wandb.log({'best_model_filepath': path})
        wandb.save(path)

    if config.get('save_config_to_wandb'):
        log.info("Hydra config will be saved to WandB as hydra_config.yaml")
        # files in wandb.run.dir folder get directly uploaded to wandb
        with open(os.path.join(wandb.run.dir, "hydra_config.yaml"), "w") as fp:
            OmegaConf.save(config, f=fp.name, resolve=True)

    wandb.finish()

    final_model = emulator_model.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        datamodule_config=config.datamodule,
        output_postprocesser=normalizer.output_variable_splitter,
        output_normalizer=normalizer.output_normalizer
    )

    return final_model
