import logging
from typing import Optional, List, Callable

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader

from climart.data_loading.constants import TRAIN_YEARS, TEST_YEARS, OOD_PRESENT_YEARS, OOD_HISTORIC_YEARS, \
    OOD_FUTURE_YEARS, VAL_YEARS
from climart.data_loading.h5_dataset import ClimART_HdF5_Dataset
from climart.data_transform.normalization import Normalizer
from climart.data_transform.transforms import AbstractTransform
from climart.utils.utils import year_string_to_list

log = logging.getLogger(__name__)


class ClimartDataModule(LightningDataModule):
    """
    ----------------------------------------------------------------------------------------------------------
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            exp_type: str,
            target_type: str,
            target_variable: str,
            data_dir: Optional[str] = None,
            train_years: str = "1999-2000",
            validation_years: str = "2005",
            predict_years: str = "2014",
            input_transform: Optional[AbstractTransform] = None,
            normalizer: Optional[Normalizer] = None,
            batch_size: int = 64,
            eval_batch_size: int = 512,
            num_workers: int = 0,
            pin_memory: bool = True,
            load_train_into_mem: bool = False,
            load_test_into_mem: bool = False,
            load_valid_into_mem: bool = True,
            test_main_dataset: bool = True,
            test_ood_1991: bool = True,
            test_ood_historic: bool = True,
            test_ood_future: bool = True,
            verbose: bool = True
    ):
        """
        Args:
            exp_type (str): 'pristine' or 'clear-sky'
            target_type (str): 'longwave" or 'shortwave'
            target_variable (str): 'fluxes' or 'heating-rate'
            data_dir (str or None): If str: A path to the data folder, if None: constants.DATA_DIR will be used.
            batch_size (int): Batch size for the training dataloader
            eval_batch_size (int): Batch size for the test and validation dataloader's
            num_workers (int): Dataloader arg for higher efficiency
            pin_memory (bool): Dataloader arg for higher efficiency
            test_main_dataset (bool): Whether to test and compute metrics on main test dataset (2007-14). Default: True
            test_ood_1991 (bool): Whether to test and compute metrics on OOD/anomaly test year 1991. Default: True
            test_ood_historic (bool): Whether to test and compute metrics on historic test years 1850-52. Default: True
            test_ood_future (bool): Whether to test and compute metrics on future test years 2097-99. Default: True
            seed (int): Used to seed the validation-test set split, such that the split will always be the same.
        """
        super().__init__()
        # The following makes all args available as, e.g., self.hparams.batch_size
        self.save_hyperparameters(ignore=["input_transform", "normalizer"])
        self.input_transform = input_transform  # self.hparams.input_transform
        self.normalizer = normalizer

        self._data_train: Optional[ClimART_HdF5_Dataset] = None
        self._data_val: Optional[ClimART_HdF5_Dataset] = None
        self._data_test: Optional[List[ClimART_HdF5_Dataset]] = None
        self._data_predict: Optional[List[ClimART_HdF5_Dataset]] = None
        self._test_set_names: Optional[List[str]] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        dataset_kwargs = dict(
            data_dir=self.hparams.data_dir,
            exp_type=self.hparams.exp_type,
            target_type=self.hparams.target_type,
            target_variable=self.hparams.target_variable,
            verbose=self.hparams.verbose,
            input_transform=self.input_transform,
            normalizer=self.normalizer,
        )

        # Training set:
        if stage == "fit" or stage is None:
            # Get & check list of training/validation years
            train_years = year_string_to_list(self.hparams.train_years)
            assert all([y in TRAIN_YEARS for y in train_years]), f"All years in --train_years must be in {TRAIN_YEARS}!"

            self._data_train = ClimART_HdF5_Dataset(years=train_years, name='Train',
                                                    load_h5_into_mem=self.hparams.load_train_into_mem,
                                                    **dataset_kwargs)
        # Validation set
        if stage in ['fit', 'validate', None] and self.hparams.validation_years is not None:
            val_yrs = year_string_to_list(self.hparams.validation_years)
            assert all([y in VAL_YEARS for y in val_yrs]), f'All years in --validation_years must be in {VAL_YEARS}!'
            self._data_val = ClimART_HdF5_Dataset(years=val_yrs, name='Val',
                                                  load_h5_into_mem=self.hparams.load_valid_into_mem,
                                                  **dataset_kwargs)
        # Test sets:
        #       - Main Present-day Test Set(s):
        #       To compute metrics for each test year -> use a separate dataloader for each of the test years (2007-14).
        if stage == "test" or stage is None:
            dataset_kwargs["load_h5_into_mem"] = self.hparams.load_test_into_mem
            if self.hparams.test_main_dataset:
                test_sets = [
                    ClimART_HdF5_Dataset(years=[test_year], name=f'Test_{test_year}', **dataset_kwargs)
                    for test_year in TEST_YEARS
                ]
            else:
                test_sets = []
                log.info(" Main test dataset (2007-14) will not be tested on in this run.")
            #   - OOD Test Sets:
            ood_test_sets = []
            if self.hparams.test_ood_1991:
                # 1991 OOD test year accounts for Mt. Pinatubo eruption: especially challenging for clear-sky conditions
                ood_test_sets += [ClimART_HdF5_Dataset(years=OOD_PRESENT_YEARS, name='OOD Test', **dataset_kwargs)]
            if self.hparams.test_ood_historic:
                ood_test_sets += [
                    ClimART_HdF5_Dataset(years=OOD_HISTORIC_YEARS, name='Historic Test', **dataset_kwargs)]
            if self.hparams.test_ood_future:
                ood_test_sets += [ClimART_HdF5_Dataset(years=OOD_FUTURE_YEARS, name='Future Test', **dataset_kwargs)]

            self._data_test = test_sets + ood_test_sets

        # Prediction set:
        if stage == "predict" and self.hparams.predict_years is not None:
            dataset_kwargs["load_h5_into_mem"] = self.hparams.load_test_into_mem
            predict_years = year_string_to_list(self.hparams.predict_years)
            self._data_predict = [
                ClimART_HdF5_Dataset(years=[pred_year], name=f'Predict_{pred_year}', **dataset_kwargs)
                for pred_year in predict_years
            ]

    @property
    def test_set_names(self) -> List[str]:
        if self._test_set_names is None:
            test_names = []
            if self.hparams.test_main_dataset:
                test_names += [f'{test_year}' for test_year in TEST_YEARS]
            if self.hparams.test_ood_1991:
                test_names += ['OOD']
            if self.hparams.test_ood_historic:
                test_names += ['historic']
            if self.hparams.test_ood_future:
                test_names += ['future']
            self._test_set_names = test_names
        return self._test_set_names

    @property
    def predict_years(self) -> List[int]:
        return year_string_to_list(self.hparams.predict_years)

    @predict_years.setter
    def predict_years(self, predict_years: str):
        self.hparams.predict_years = predict_years

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch

    def _shared_dataloader_kwargs(self) -> dict:
        shared_kwargs = dict(num_workers=int(self.hparams.num_workers), pin_memory=self.hparams.pin_memory)
        return shared_kwargs

    def _shared_eval_dataloader_kwargs(self) -> dict:
        return dict(**self._shared_dataloader_kwargs(), batch_size=self.hparams.eval_batch_size, shuffle=False)

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            **self._shared_dataloader_kwargs(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val,
            **self._shared_eval_dataloader_kwargs()
        ) if self._data_val is not None else None

    def test_dataloader(self) -> List[DataLoader]:
        return [DataLoader(
            dataset=data_test_subset,
            **self._shared_eval_dataloader_kwargs()
        ) for data_test_subset in self._data_test]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=data_test_subset,
            **self._shared_eval_dataloader_kwargs()
        ) for data_test_subset in self._data_predict]