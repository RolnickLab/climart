import logging
import os
import time
from datetime import datetime
from typing import Optional, List, Any, Dict

import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from climart.utils.callbacks import TestingScheduleCallback
from climart.utils.evaluation import evaluate_preds, evaluate_preds_per_var
from climart.utils.model_logging import update_tqdm, dataset_split_wandb_dict, toa_profile_plots_wandb_dict, toa_level_errors_wandb_dict
from climart.utils.optimization import get_loss, get_optimizer, get_scheduler, ReduceLROnPlateau
from climart.utils.preprocessing import Normalizer
from climart.utils.utils import set_seed, get_logger


class BaseModel(nn.Module):
    """
    This is a template class, that should be inherited by all downstream models.
    It defines all necessary operations, although __init__ and forward will definitely need to be implemented
    by the concrete downstream model.
    """

    def __init__(self,
                 normalizer: Optional[Normalizer] = None,
                 out_layer_bias_init: Optional[np.ndarray] = None,
                 name: str = "",
                 verbose: bool = True,
                 *args, **kwargs):
        super().__init__()
        self.log = get_logger(name=self.__class__.__name__ if name == '' else name)
        if not verbose:
            self.log.setLevel(logging.WARN)
        self.out_size = -1
        self.out_normalizer = normalizer.copy() if isinstance(normalizer, Normalizer) else None
        if out_layer_bias_init is not None:
            self.out_layer_bias_init = out_layer_bias_init if torch.is_tensor(
                out_layer_bias_init) else torch.from_numpy(out_layer_bias_init)
        else:
            self.out_layer_bias_init = None

        if self.out_normalizer is not None:
            normalizer_name = self.out_normalizer.__class__.__name__
            self.log.info(f' Using an output inverse normalizer {normalizer_name} for prediction.')
            self.out_normalizer.change_input_type(torch.Tensor)
        else:
            self.log.info(' No inverse normalization for outputs is used.')

    @staticmethod
    def _input_transform(X: Dict[str, np.ndarray]) -> Any:
        """
        How to transform dict
            X = {
                'layer': layer_array,   # shape (#layers, #layer-features)
                'levels': levels_array, # shape (#levels, #level-features)
                'globals': globals_array (#global-features,)
                }
        to the form the model will use/receive it in forward.
        Implementation will be applied (with multi-processing) in the _get_item(.) method of the dataset
            --> IMPORTANT: the arrays in X will *not* have the batch dimension!
        """
        return X

    @staticmethod
    def _batched_input_transform(X: Dict[str, np.ndarray]) -> Any:
        """
        How to transform dict
            X = {
                'layer': layer_array,   # shape (batch-size, #layers, #layer-features)
                'levels': levels_array, # shape (batch-size, #levels, #level-features)
                'globals': globals_array (batch-size, #global-features,)
                }
        to the form the model will use/receive it in forward.
        """
        return X

    def forward(self, X):
        """
        Downstream model forward pass, input X will be the (batched) output from self.input_transform
        """
        raise NotImplementedError('Base model is an abstract class!')

    def predict(self, X, *args, **kwargs):
        Y_normed = self(X)
        Y_raw = self.Y_to_raw(Y_normed, *args, **kwargs)
        return Y_raw

    def Y_to_raw(self, Y, *args, **kwargs):
        if self.out_normalizer is not None:
            Y = self.out_normalizer.inverse_normalize(Y)
        return self._predict_flux(Y, *args, **kwargs)

    def _predict_flux(self, Y, *args, **kwargs):
        return F.relu(Y)

    def _apply(self, fn):
        super(BaseModel, self)._apply(fn)
        if self.out_normalizer is not None:
            self.out_normalizer.apply_torch_func(fn)
        return self


def concat_dicts(*args):
    x = args[0]
    for y in args[1:]:
        x = {**x, **y}
    return x


class BaseTrainer:
    """
    A template class to be wrapped around for training & evaluating a
     specific downstream model that inherits from DownstreamBaseModel.

    To wrap around it you just need to inherit from it and overwrite as follows (and replace xyz with an appropriate name):
    >>
        class YourModelTrainer(DownstreamBaseTrainer):
            def __init__(self, model_params, name='WXYZ', model_dir="out/WXYZ"):
                super().__init__(downstream_params, name=name, model_dir=model_dir)
                self.model_class = YourModelClass
    where
         YourModelClass is a pointer to your model class, e.g. ResNet, MLPNet,...
         name  is a name ID for your model, e.g. 'ResNet', 'MLP',...
    """

    def __init__(self, model_params, name: str = None, seed=None, verbose=False, model_dir=None, device='cuda',
                 notebook_mode=False, model=None,
                 output_normalizer: Normalizer = None,
                 output_postprocesser=None,
                 out_layer_bias: Optional[np.ndarray] = None
                 ):
        self.log = get_logger(self.__class__.__name__)
        if seed is not None:
            set_seed(seed)
        self.model_class: BaseModel = BaseModel
        self.model_params = model_params
        self.model = model
        self._device = device
        self.verbose = verbose
        self.model_dir = model_dir
        self.name = name or self.__class__.__name__
        self.save_model_filepath = None
        self.current_epoch = 1
        self._gradient_steps = self._example_count = 0
        self.best_valid_val = None
        self.val_metric = 'mae'
        self.out_layer_bias = out_layer_bias


        if output_normalizer is not None and not isinstance(output_normalizer, Normalizer):
            self.log.warning(
                f" Out normalizer passed to trainer is not an instance of 'Normalizer', but {type(output_normalizer)}")
        self.output_normalizer = output_normalizer if isinstance(output_normalizer, Normalizer) else None
        if self.output_normalizer is not None and hasattr(self.output_normalizer, 'var_splitter'):
            self.output_postprocesser = self.output_normalizer.var_splitter
        else:
            self.output_postprocesser = output_postprocesser
        if self.output_postprocesser is not None:
            self.log.info(' Using an output channel-by-variable splitter.')
        if notebook_mode:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_model_dir(self):
        return f"out/" if self.model_dir is None else self.model_dir

    '''
    def _get_best_model_path(self):
        if self.save_model_filepath is None:
            return f'{self._get_model_dir()}best_{self.name}_model.pkl'
        return self._get_model_dir() + self.save_model_filepath
    '''

    def _model_init_kwargs(self):
        return {**self.model_params,
                'normalizer': self.output_normalizer,
                'out_layer_bias_init': self.out_layer_bias
                }

    def _get_model(self, get_new=False, **kwargs):
        """
        Args:
         get_new:
         kwargs: Any keyword args to give the model class init method.
        """
        if self.model is None or get_new:
            model_kwargs = self._model_init_kwargs()
            for k, i in kwargs.items():
                model_kwargs[k] = i
            model = self.model_class(**model_kwargs)
            return model.to(self._device).float()
        return self.model

    def reload_model(self, model_state_dict, **kwargs):
        self.model = self._get_model(get_new=True, **kwargs)
        self.model.load_state_dict(model_state_dict)

    def _compile(self, params: dict):
        self.batch_size = params["batch_size"]
        self.criterion = get_loss(params['loss'])  # loss function
        self._train_on_raw_targets = params['train_on_raw_targets']
        if self.train_on_raw_targets:
            self.log.info(" Train will directly call model.predict() for loss computation.")
        self.gradient_clipping_func = None
        if params['gradient_clipping'] in ['l2', 'norm']:
            self.gradient_clipping_func = nn.utils.clip_grad_norm_
        elif params['gradient_clipping'] == 'value':
            self.gradient_clipping_func = nn.utils.clip_grad_value_

        self.add_noise_to_input = 'add_noise' in params and params['add_noise']
        if self.add_noise_to_input:
            print('Gaussian noise will be added to the inputs during training!')
        self.val_metric = params['val_metric'].lower()
        self.best_valid_val = 1e5  # if smaller_is_better(self.val_metric) else -1

        # Initialize Optimizer and Scheduler
        self._example_count = self._gradient_steps = 0
        optim_name = params['optim']
        scheduler_name = params['scheduler']
        lr, wd = params["lr"], params['weight_decay']
        self.optimizer = get_optimizer(optim_name, self.model, lr=lr, weight_decay=wd)
        self.scheduler = get_scheduler(scheduler_name, self.model, self.optimizer, total_epochs=params['epochs'])

    def reload_from_checkpoint(self, checkpoint_path: str, **kwargs):
        saved_model = torch.load(checkpoint_path)

        self.current_epoch = saved_model['epoch'] + 1
        self._gradient_steps = saved_model['step']
        self._example_count = saved_model['example_count']
        self.best_valid_val = saved_model['validation_stat']

        self.reload_model(saved_model['model'], **kwargs)
        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(saved_model['optimizer'])
        if 'scheduler' in saved_model.keys() and hasattr(self, 'scheduler'):
            self.scheduler.scheduler.load_state_dict(saved_model['scheduler'])

        self.log.info(f" Reloading checkpoint at epoch {self.current_epoch} and best validation {self.val_metric}={self.best_valid_val}")

    def change_device(self, device):
        self._device = device
        if self.model is not None:
            self.model.to(self._device)

    @property
    def train_on_raw_targets(self):
        return self._train_on_raw_targets or self.output_normalizer is None

    def batch_to_device(self, batch):
        batch_features, batch_y = batch
        batch_features = self.data_to_device(batch_features)
        batch_y = self.data_to_device(batch_y)
        return batch_features, batch_y

    def data_to_device(self, data):
        if torch.is_tensor(data):
            data = data.to(self._device)
        elif isinstance(data, dict):
            data = {k: x.to(self._device) for k, x in data.items()}
        else:
            data = [x.to(self._device) for x in data]
        return data

    def train_epoch(self, trainloader, hyper_params, epoch):
        start_t = time.time()
        self.model.train()
        epoch_loss = epoch_zero_gradients = 0.0
        # Cycle through batches
        for i, batch in enumerate(trainloader, 1):
            x, y = self.batch_to_device(batch)
            self._example_count += y.shape[0]
            if self.add_noise_to_input:
                x += torch.normal(mean=0, std=hyper_params['scale'], size=x.size())
            self.optimizer.zero_grad()

            # Forward pass through
            if self.train_on_raw_targets:
                # Directly predict full/raw/non-normalized outputs
                yhat = self.model.predict(x)
            else:
                # Compute loss on normalized outputs and predictions
                yhat = self.model(x)

            if not torch.is_tensor(y):
                y = y.values() if isinstance(y, dict) else y
                y = torch.cat(y, dim=1)

            loss = self.criterion(yhat, y)
            loss.backward()  # Compute the gradients

            if self.gradient_clipping_func is not None:
                self.gradient_clipping_func(self.model.parameters(), hyper_params['clip'])

            self.optimizer.step()  # Step the optimizer to update the model weights
            self._gradient_steps += 1
            n_zero_gradients = sum(
                [int(torch.count_nonzero(p.grad == 0))
                 for p in self.model.parameters() if p.grad is not None]
            ) / self.n_params

            epoch_zero_gradients += n_zero_gradients
            epoch_loss += loss.detach().item()
            # wandb.log({"Train loss": loss, 'Zero gradients':  n_zero_gradients}, step=self._example_count)

        epoch_loss, epoch_zero_gradients = epoch_loss / i, epoch_zero_gradients / i
        train_t = time.time() - start_t
        logging_dict = {'epoch': epoch, 'Train/loss': epoch_loss, 'time/train': train_t,
                        'Gradient steps': self._gradient_steps, 'Zero gradients': epoch_zero_gradients,
                        'Lr': self.scheduler.get_last_lr()}
        return logging_dict

    def on_train_end(self, logging_dict: dict) -> dict:
        return logging_dict

    def fit(self,
            trainloader: DataLoader,
            valloader: DataLoader,
            hyper_params: dict,
            testloader: List[DataLoader] = None,
            testloader_names: List[str] = None,
            ood_testloader: List[DataLoader] = None,
            ood_testloader_names: List[str] = None,
            expID='',
            checkpoint: Optional[str] = None
            ):
        self.model = self._get_model(get_new=False)

        self._compile(hyper_params)
        if isinstance(checkpoint, str):
            self.reload_from_checkpoint(checkpoint)

        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #wandb.watch(self.model)
        wandb.log({'Parameter count': self.n_params})

        start_epoch = self.current_epoch
        when_to_test = TestingScheduleCallback(start_epoch=start_epoch)
        # Cycle through epochs
        with self.tqdm(range(start_epoch, hyper_params["epochs"] + 1)) as t:
            t.set_description(f'{self.name}')
            for epoch in t:
                self.current_epoch = epoch
                start_t = time.time()
                logging_dict = self.train_epoch(trainloader, hyper_params, epoch)
                logging_dict = self.on_train_end(logging_dict)

                _, _, val_stats = self.evaluate(valloader)
                valid_metric_val = val_stats[self.val_metric]
                logging_dict = concat_dicts(
                    logging_dict,
                    dataset_split_wandb_dict(data_split='Val', statistics=val_stats, stats_to_save='default')
                )
                if when_to_test(is_new_best_val_model=self._is_new_best_validation_stat(val_stats)):
                    if testloader is not None:
                        logging_dict = self.test(
                            testloader, testloader_names, use_best_model=False,
                            parent_dict=logging_dict, aggregated_test_name="Test"
                        )
                    if ood_testloader is not None:
                        logging_dict = self.test(
                            ood_testloader, ood_testloader_names, use_best_model=False,
                            parent_dict=logging_dict, aggregated_test_name="Test_OOD"
                        )

                self.scheduler.step(self.model, epoch, valid_metric_val)

                update_tqdm(t, logging_dict=logging_dict)
                logging_dict['time/eval'] = time.time() - start_t - logging_dict['time/train']
                wandb.log(logging_dict, step=self._gradient_steps)
                self._save_best_model(val_stats, logging_dict, hyper_params, remove_previous=True)

        self.model = self.scheduler.end(trainloader)
        return self.best_valid_val

    def evaluate(self,
                 dataloader,
                 model_name=None,
                 verbose=False, use_best_model=False, **kwargs):
        """
        This function evaluates the downstream model based on gold labels.
        :param dataloader: A torch DataLoader instance, that returns batches (X, Y), where
                            Y (n,) are true labels corresponding to the features X (n, d)
        :param model_name: A prefix to be printed with the metrics, if verbose is True
        :param use_best_model: If false, the model at the state the function is called is used for prediction
                               if true, the saved model is reloaded and used for prediction (only possible if models
                                  are being saved, i.e. model_dir is not None).
        :return: A tuple (Y_endmodel, Y_true, stats_dict), where
                Y_endmodel are the final predictions of the downstream model
                Y_true are the corresponding gold labels provided by the dataloader, and
                stats_dict is a dictionary of metric_name --> metric_value pairs, including
                'mae', 'mse', 'rmse',
        """
        verbose = verbose or model_name is not None
        model_name = model_name or self.name

        if use_best_model:
            if self.model_dir is not None:
                try:
                    saved_model = torch.load(self.save_model_filepath)
                    model = self._get_model(get_new=True)
                    model.load_state_dict(state_dict=saved_model['model'])
                except AttributeError:
                    print('No model was saved, since no validation set was given, using the last one.')
                    model = self.model
            else:
                print('No models were saved, using the current one.')
                model = self.model
        else:
            model = self.model

        preds, Y = self.get_preds(dataloader, labels_too=True, model=model, split_preds_by_var=False)

        try:
            stats = evaluate_preds(Y, preds, verbose=verbose, model_name=model_name)
            if self.output_postprocesser is not None:
                # Split output variables back into separate vectors/predictions
                splitted_preds = self.output_postprocesser.split_vector_by_variable(preds)
                splitted_Y = self.output_postprocesser.split_vector_by_variable(Y)
                stats_per_var = evaluate_preds_per_var(splitted_Y, splitted_preds, verbose=verbose,
                                                       model_name=model_name)
                stats = {**stats, **stats_per_var}
                return splitted_preds, splitted_Y, stats
        except ValueError as e:
            print(e)
            stats = {'mae': 1e5, 'mbe': 1e5, 'rmse': 1e5}

        return preds, Y, stats

    def test(self,
             testloaders: List[DataLoader],
             testloader_names: List[str] = None,
             aggregated_test_name: str = "Test",
             use_best_model: bool = True,
             checkpoint: str = None,
             parent_dict: Optional[dict] = None,
             logging_key_prefix: Optional[str] = None,
             verbose: bool = False,
             model_verbose: bool = False,
             **kwargs
             ) -> dict:
        logging_dict = dict() if parent_dict is None else parent_dict
        if checkpoint is not None:
            self.reload_from_checkpoint(checkpoint, verbose=model_verbose)
            if use_best_model:
                self.log.info(" Reloading model from checkpoint, instead of the one saved as best!")
            use_best_model = False
        if isinstance(testloaders, DataLoader):
            testloaders = [testloaders]
            testloader_names = [aggregated_test_name] if testloader_names is None else [testloader_names]
        elif testloader_names is None:
            testloader_names = [f"Test_{i}" for i in range(1, len(testloaders) + 1)]
        else:
            assert len(testloader_names) == len(testloaders)

        num_testloaders = len(testloaders)
        aggregate_test_metrics = None
        kwargs = dict(
            stats_to_save='all',
            exclude='mse',
            prefix=logging_key_prefix
        )
        for single_test_loader, test_name in zip(testloaders, testloader_names):
            test_preds, test_Y, test_stats = self.evaluate(
                single_test_loader, use_best_model=use_best_model, model_name=f"{self.name}_{test_name}",
                verbose=verbose
            )
            if num_testloaders <= 1:
                pass
            elif aggregate_test_metrics is None:
                aggregate_test_metrics = test_stats.copy()
            else:
                for k, v in test_stats.items():
                    aggregate_test_metrics[k] += v

            test_stats = dataset_split_wandb_dict(data_split=test_name, statistics=test_stats, **kwargs)
            print(test_name, test_stats.keys())
            logging_dict = {**logging_dict, **test_stats}
            if False: #self.current_epoch % 20 == 0 or use_best_model:
                ep = self.current_epoch
                logging_dict = concat_dicts(
                    logging_dict,
                    # height_error_plots_wandb_dict(test_Y, test_preds, data_split=test_name).
                    toa_profile_plots_wandb_dict(
                        test_Y, test_preds, plot_type='scatter', data_split=test_name, title=f'Epoch {ep} '
                    ),
                    toa_level_errors_wandb_dict(test_Y, test_preds, data_split=test_name, epoch=ep)
                )
        if aggregate_test_metrics is not None and num_testloaders > 1:
            for k in aggregate_test_metrics:
                aggregate_test_metrics[k] /= num_testloaders
            aggregate_test_metrics = dataset_split_wandb_dict(data_split=aggregated_test_name,
                                                              statistics=aggregate_test_metrics, **kwargs)
            logging_dict = {**logging_dict, **aggregate_test_metrics}
        return logging_dict

    def get_preds(self, dataset, labels_too=False, split_preds_by_var=False, model=None):
        if model is None:
            model = self.model
        if not isinstance(dataset, torch.utils.data.DataLoader):
            dataloader = DataLoader(dataset, batch_size=self.batch_size, pin_memory=True, shuffle=False)
        else:
            dataloader = dataset
        model.eval()
        with torch.no_grad():
            for i, (batch_features, batch_y) in enumerate(dataloader):
                batch_features = self.data_to_device(batch_features)
                batch_y = batch_y.numpy()

                yhat = model.predict(batch_features)
                yhat = yhat.detach().cpu().numpy()
                if i == 0:
                    Y, preds = batch_y, yhat
                else:
                    Y, preds = np.concatenate((Y, batch_y), axis=0), np.concatenate((preds, yhat), axis=0)

        if self.output_postprocesser is not None:
            # Split concatenated output into the multiple variables, e.g. down- and upwelling flux
            if split_preds_by_var:
                preds = self.output_postprocesser.split_vector_by_variable(preds)
                Y = self.output_postprocesser.split_vector_by_variable(Y)

        if labels_too:
            return preds, Y
        return preds

    def _is_new_best_validation_stat(self, new_stats: dict):
        return new_stats[self.val_metric] < self.best_valid_val

    def _save_best_model(self, new_model_stats: dict, model_stats: dict, hyperparams, remove_previous=True):
        r"""
        :param new_model_stats: a metric dictionary containing a key name 'metric_name'
        :param old_best: Old best metric_name value
        :param metric_name: A metric, e.g. RMSE  (note that the code only supports metrics, where lower is better)
        :return: The best new metric, i.e. old_best if it is better than the newer model's performance and vice versa.
        """
        if new_model_stats is None:
            return self.best_valid_val
        if self._is_new_best_validation_stat(new_model_stats):  # save best model (wrt validation data)
            self.best_valid_val = new_model_stats[self.val_metric]
            print(f"Best model so far with validation {self.val_metric} =", '{:.3f}'.format(self.best_valid_val))
            self._save_model(hyperparams, epoch=self.current_epoch,
                             validation_stat=self.best_valid_val, remove_previous=remove_previous)
            for k, val in model_stats.items():
                if isinstance(val, float) or isinstance(val, np.float32):
                    wandb.run.summary[k] = val
            # Save the model in the exchangeable ONNX format
            # torch.onnx.export(self.model, x, f"{self.name}_model.onnx")
            # wandb.save(f"{self.name}_model.onnx")
            # torch.save(self.model.state_dict(), self._get_best_model_path())
            # wandb.save(self._get_best_model_path())
        return self.best_valid_val

    def _save_model(self, hyper_params: dict, epoch, validation_stat=1e5, filepath: str = None,
                    remove_previous=True):
        checkpoint_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'epoch': epoch,
            'example_count': self._example_count,
            'step': self._gradient_steps,
            'validation_stat': validation_stat,
            'hyper_params': hyper_params,
            'model_params': self.model_params
        }
        if not isinstance(self.scheduler, ReduceLROnPlateau):
            checkpoint_dict['scheduler'] = self.scheduler.scheduler.state_dict()
        # In case a model dir was given --> save best model (wrt validation data)
        if filepath is None:
            filepath = self._get_model_dir()
            if validation_stat < 1e5:
                filepath += f"{validation_stat:.4f}val{self.val_metric.upper()}_"
            if epoch is not None:
                filepath += f'{epoch}ep_'
            if 'wandb_name' in hyper_params:
                filepath += hyper_params['wandb_name'] + '.pkl'
            else:
                filepath += f'{self.name}_model.pkl'

        torch.save(checkpoint_dict, filepath)
        if remove_previous and self.save_model_filepath is not None:
            os.remove(self.save_model_filepath)
        self.save_model_filepath = filepath
