import logging
import time
from typing import Optional, List, Any, Dict

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn
from timm.optim import create_optimizer_v2
from pytorch_lightning import LightningModule

from climart.data_loading.constants import TEST_YEARS, get_data_dims, LAYERS, LEVELS
from climart.data_transform.transforms import AbstractTransform
from climart.utils.callbacks import PredictionPostProcessCallback
from climart.utils.evaluation import evaluate_per_target_variable
from climart.utils.optimization import get_loss, get_trainable_params
from climart.data_transform.normalization import NormalizationMethod
from climart.utils.utils import get_logger, fluxes_to_heating_rates, to_DictConfig


class BaseModel(LightningModule):
    """
    This is a template class, that should be inherited by any neural net emulator model.
    Methods that need to be implemented by your concrete NN model (just as if you would define a torch.nn.Module):
        - __init__(.)
        - forward(.)

    The other methods may be overridden as needed.
    It is recommended to define the attribute
        - self.example_input_array = torch.randn(<YourModelInputShape>)  # batch dimension can be anything, e.g. 7

    ------------
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,
                 datamodule_config: DictConfig = None,
                 optim: Optional[DictConfig] = None,
                 scheduler: Optional[DictConfig] = None,
                 monitor: Optional[str] = None,
                 mode: str = "min",
                 loss_function: str = "mean_squared_error",
                 downwelling_loss_contribution: float = 0.5,
                 upwelling_loss_contribution: float = 0.5,
                 heating_rate_loss_contribution: float = 0.0,
                 input_transform: Optional[AbstractTransform] = None,
                 output_normalizer: Optional[Dict[str, NormalizationMethod]] = None,
                 out_layer_bias_init: Optional[np.ndarray] = None,
                 name: str = "",
                 verbose: bool = True,
                 *args, **kwargs
                 ):
        super().__init__()
        self.log_text = get_logger(name=self.__class__.__name__ if name == '' else name)
        self.name = name
        self.verbose = verbose
        if not self.verbose:
            self.log_text.setLevel(logging.WARN)
        if input_transform is None or isinstance(input_transform, AbstractTransform):
            self.input_transform = input_transform
        else:
            self.input_transform = hydra.utils.instantiate(input_transform)
        if datamodule_config is not None:
            input_output_dimensions = get_data_dims(exp_type=datamodule_config.get("exp_type"))
            self.raw_input_dim = input_output_dimensions['input_dim']
            self.raw_output_dim = input_output_dimensions['output_dim']
            self.raw_spatial_dim = input_output_dimensions['spatial_dim']
            self.num_layers = self.raw_spatial_dim[LAYERS]
            self.num_levels = self.raw_spatial_dim[LEVELS]

        self.output_normalizer = output_normalizer.copy() if output_normalizer is not None else None
        if out_layer_bias_init is not None:
            self.out_layer_bias_init = out_layer_bias_init if torch.is_tensor(
                out_layer_bias_init) else torch.from_numpy(out_layer_bias_init)
        else:
            self.out_layer_bias_init = None

        if self.output_normalizer is None:
            self.log_text.info(' No output normalization for outputs is used.')
        else:
            normalizer_name = list(self.output_normalizer.values())[0].__class__.__name__
            self.log_text.info(f' Using output normalization "{normalizer_name}" for prediction.')
            for v in self.output_normalizer.values():
                v.change_input_type(torch.Tensor)

        # loss function
        self.criterion = get_loss(loss_function)
        if datamodule_config is not None:
            # shortwave or longwave ?
            self.target_type = "l" if datamodule_config.get('target_type') == "longwave" else "s"
            self._downwelling_flux_id = f"r{self.target_type}dc"
            self._upwelling_flux_id = f"r{self.target_type}uc"
            self._heating_rate_id = f"hr{self.target_type}c"
            self._out_var_ids = [self._downwelling_flux_id, self._upwelling_flux_id, self._heating_rate_id]
            flux_var_ids = [self._downwelling_flux_id, self._upwelling_flux_id]
            self.output_postprocesser = PredictionPostProcessCallback(variables=flux_var_ids, sizes=self.num_levels)

        self.upwelling_loss_contribution = upwelling_loss_contribution
        self.downwelling_loss_contribution = downwelling_loss_contribution
        self.heating_rate_loss_contribution = heating_rate_loss_contribution
        if heating_rate_loss_contribution > 0 and self.output_normalizer is not None:
            raise ValueError(" Computing errors on normalized heating rates is not supported yet.")

        self._start_validation_epoch_time = self._start_test_epoch_time = self._start_epoch_time = None
        # Metrics
        # self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        # self.test_metrics_rmse = nn.ModuleList([torchmetrics.MeanSquaredError(squared=False) for _ in range(12)])
        # self.test_metrics_mbe = nn.ModuleList([torchmetrics.MeanSquaredError(squared=False) for _ in range(12)])

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, X):
        """
        Downstream model forward pass, input X will be the (batched) output from self.input_transform
        """
        raise NotImplementedError('Base model is an abstract class!')

    def raw_predict(self, X) -> Dict[str, Tensor]:
        X_feats = X['data']
        if isinstance(self.input_transform, nn.Module):
            X_feats = self.input_transform(X['data'])
        Y = self(X_feats)  # Y might be in normalized scale or not
        flux_profile_pred = self.output_postprocesser.split_vector_by_variable(Y)
        return flux_profile_pred

    def predict(self, X, *args, **kwargs) -> Dict[str, Tensor]:
        Y_normed = self.raw_predict(X)
        flux_prediction = self.model_output_to_fluxes(Y_normed, *args, **kwargs)
        heating_rate_profile_pred = fluxes_to_heating_rates(
            downwelling_flux=flux_prediction[self._downwelling_flux_id],
            upwelling_flux=flux_prediction[self._upwelling_flux_id],
            pressure=X['level_pressure_profile'])
        return {**flux_prediction, self._heating_rate_id: heating_rate_profile_pred}

    def model_output_to_fluxes(self, flux_profile_pred, *args, **kwargs) -> Dict[str, Tensor]:
        # downwelling_profile = flux_profile_pred[self._downwelling_flux_id]
        # upwelling_profile = flux_profile_pred[self._upwelling_flux_id]
        for flux_type, flux_pred in flux_profile_pred.items():
            if self.output_normalizer is not None:
                flux_pred = self.output_normalizer[flux_type].inverse_normalize(flux_pred)
            flux_pred = self._predict_flux(flux_pred, *args, **kwargs)
            flux_profile_pred[flux_type] = flux_pred

        return flux_profile_pred
        # return dict(downwelling_flux=downwelling_profile, upwelling_flux=upwelling_profile)

    def _predict_flux(self, Y, *args, **kwargs):
        return F.relu(Y)

    # --------------------- training with PyTorch Lightning
    def on_train_start(self) -> None:
        self.log('Parameter count', self.n_params)

    def on_train_epoch_start(self) -> None:
        self._start_epoch_time = time.time()

    def training_step(self, batch: Any, batch_idx: int):
        X, Y = batch

        if self.output_normalizer is None:
            # Directly predict full/raw/non-normalized outputs
            preds = self.predict(X)
        else:
            # Predict normalized outputs
            preds = self.raw_predict(X)

        up = preds[self._upwelling_flux_id]
        up_true = Y[self._upwelling_flux_id]
        down = preds[self._downwelling_flux_id]
        down_true = Y[self._downwelling_flux_id]
        if self.output_normalizer is not None:
            # normalize targets
            up_true = self.output_normalizer[self._upwelling_flux_id].normalize(up_true)
            down_true = self.output_normalizer[self._downwelling_flux_id].normalize(down_true)

        train_log = dict()
        l1 = l2 = l3 = 0
        if self.upwelling_loss_contribution > 0:
            l1 = self.upwelling_loss_contribution * self.criterion(up, up_true)
            train_log["train/loss_upwelling"] = l1
        if self.downwelling_loss_contribution > 0:
            l2 = self.downwelling_loss_contribution * self.criterion(down, down_true)
            train_log["train/loss_downwelling"] = l2
        if self.output_normalizer is None and self.heating_rate_loss_contribution > 0:
            hr = preds[self._heating_rate_id]
            l3 = self.heating_rate_loss_contribution * self.criterion(hr, Y[self._heating_rate_id])
            train_log["train/loss_heating_rate"] = l3
        loss = l1 + l2 + l3

        n_zero_gradients = sum(
            [int(torch.count_nonzero(p.grad == 0))
             for p in self.parameters() if p.grad is not None
             ]) / self.n_params

        self.log_dict({**train_log, "train/loss": loss, "n_zero_gradients": n_zero_gradients})
        return {
            "loss": loss,
            'n_zero_gradients': n_zero_gradients,
            "targets": Y,
            "preds": preds,
        }

    def training_epoch_end(self, outputs: List[Any]):
        train_time = time.time() - self._start_epoch_time
        self.log_dict({'epoch': self.current_epoch, "time/train": train_time})

    # --------------------- evaluation with PyTorch Lightning
    def _evaluation_step(self, batch: Any, batch_idx: int):
        X, Y = batch
        preds = self.predict(X)
        return {'targets': Y, 'preds': preds}

    def _evaluation_get_preds(self, outputs: List[Any]) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        Y = {
            out_var: torch.cat([batch['targets'][out_var] for batch in outputs], dim=0).cpu().numpy()
            for out_var in self._out_var_ids
        }
        preds = {
            out_var: torch.cat([batch['preds'][out_var] for batch in outputs], dim=0).detach().cpu().numpy()
            for out_var in self._out_var_ids
        }
        return Y, preds

    def on_validation_epoch_start(self) -> None:
        self._start_validation_epoch_time = time.time()

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: List[Any]) -> dict:
        val_time = time.time() - self._start_validation_epoch_time
        self.log("time/validation", val_time)
        validation_outputs = outputs
        Y_val, validation_preds = self._evaluation_get_preds(validation_outputs)
        val_stats = evaluate_per_target_variable(Y_val, validation_preds, data_split='val')
        target_val_metric = val_stats.pop(self.hparams.monitor)
        self.log_dict({**val_stats, 'epoch': self.current_epoch}, prog_bar=False)
        # Show the main validation metric on the progress bar:
        self.log(self.hparams.monitor, target_val_metric, prog_bar=True)
        return val_stats

    def on_test_epoch_start(self) -> None:
        self._start_test_epoch_time = time.time()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]) -> dict:
        test_time = time.time() - self._start_test_epoch_time
        self.log("time/test", test_time)
        main_test_stats = dict()
        for i, test_subset_outputs in enumerate(outputs):
            split_name = self.trainer.datamodule.test_set_names[i]
            Y, preds = self._evaluation_get_preds(test_subset_outputs)
            if i < len(TEST_YEARS):
                split_name = f"test_year/{split_name}"
            else:
                split_name = f"test_{split_name}"
            test_stats = evaluate_per_target_variable(Y, preds, data_split=split_name)
            self.log_dict({**test_stats, 'epoch': self.current_epoch}, prog_bar=False)

            if i < len(TEST_YEARS):
                for v in [self._heating_rate_id, self._upwelling_flux_id, self._downwelling_flux_id]:
                    if i == 0:
                        main_test_stats[f'test/{v}/rmse'] = main_test_stats[f'test/{v}/mbe'] = 0.0
                    main_test_stats[f'test/{v}/rmse'] += test_stats[f"{split_name}/{v}/rmse"]
                    main_test_stats[f'test/{v}/mbe'] += test_stats[f"{split_name}/{v}/mbe"]

        for v in [self._heating_rate_id, self._upwelling_flux_id, self._downwelling_flux_id]:
            main_test_stats[f'test/{v}/rmse'] /= len(TEST_YEARS)
            main_test_stats[f'test/{v}/mbe'] /= len(TEST_YEARS)

        self.log_dict({**main_test_stats, 'epoch': self.current_epoch}, prog_bar=False)
        return main_test_stats

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = None):
        return self._evaluation_step(batch, batch_idx)

    def on_predict_epoch_end(self, results: List[Any]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        return self.aggregate_predictions(results)

    def aggregate_predictions(self, results: List[Any]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        Args:
         results: The list that pl.Trainer() returns when predicting, i.e.
                        results = trainer.predict(model, datamodule)
        Returns:
            A dict mapping prediction_set_name_i -> {'targets': t_i, 'preds': p_i}
                for each prediction subset i (for each prediction year).
                E.g.: To access the shortwave heating rate predictions for year 2012:
                    model, datamodule, trainer = ...
                    datamodule.predict_years = "2012"
                    results = trainer.predict(model, datamodule)
                    results = model.aggregate_predictions(results)
                    sw_hr_preds_2012 = results[2012]['preds']['hrsc']
        """
        if not isinstance(results[0], list):
            results = [results]  # when only a single predict dataloader is passed
        per_subset_outputs = dict()
        for pred_set_name, predict_subset_outputs in zip(self.trainer.datamodule.predict_years, results):
            Y, preds = self._evaluation_get_preds(predict_subset_outputs)
            per_subset_outputs[pred_set_name]['preds'] = preds
            per_subset_outputs[pred_set_name]['targets'] = Y
        return per_subset_outputs

    # ---------------------------------------------------------------------- Optimizers and scheduler(s)
    def configure_optimizers(self):
        if '_target_' in to_DictConfig(self.hparams.optimizer).keys():
            self.hparams.optimizer.name = str(self.hparams.optimizer._target_.split('.')[-1]).lower()
        if 'name' not in to_DictConfig(self.hparams.optimizer).keys():
            self.log_text.info(" No optimizer was specified, defaulting to AdamW with 1e-4 lr.")
            self.hparams.optimizer.name = 'adamw'

        if hasattr(self, 'no_weight_decay'):
            self.log_text.info(f"Model has method no_weight_decay, which will be used.")
        optim_kwargs = {k: v for k, v in self.hparams.optimizer.items() if k not in ['name', '_target_']}
        optimizer = create_optimizer_v2(model_or_params=self, opt=self.hparams.optimizer.name, **optim_kwargs)
        self._init_lr = optimizer.param_groups[0]['lr']

        if self.hparams.scheduler is None:
            return optimizer  # no scheduler
        else:
            if '_target_' not in to_DictConfig(self.hparams.scheduler).keys():
                raise ValueError("Please provide a _target_ class for model.scheduler arg!")
            scheduler_params = to_DictConfig(self.hparams.scheduler)
            scheduler = hydra.utils.instantiate(scheduler_params, optimizer=optimizer)

        if not hasattr(self.hparams, 'monitor') or self.hparams.monitor is None:
            self.hparams.monitor = f'val/{self._heating_rate_id}/rmse'
        if not hasattr(self.hparams, 'mode') or self.hparams.mode is None:
            self.hparams.mode = 'min'

        lr_dict = {'scheduler': scheduler, 'monitor': self.hparams.monitor, 'mode': self.hparams.mode}
        return {'optimizer': optimizer, 'lr_scheduler': lr_dict}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
