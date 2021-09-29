from typing import Optional, Dict, List, Union
import numpy as np
import torch


class TestingScheduleCallback:
    def __init__(self,
                 start_epoch: int = 1,
                 test_at_most_every_n_epochs: int = 10,
                 test_at_least_every_n_epochs: int = 20,
                 test_on_new_best_validation: bool = True,
                 ignore_first_n_epochs: int = 5,
                 ):
        self.test_at_most_every_n_epochs = test_at_most_every_n_epochs
        self.test_at_least_every_n_epochs = test_at_least_every_n_epochs
        self.test_on_new_best_validation = test_on_new_best_validation
        self.untested_epochs = 1
        self.cur_epoch = start_epoch
        self.ignore_first_n_epochs = ignore_first_n_epochs

    def __call__(self, is_new_best_val_model: bool = False) -> bool:
        do_test = False
        if self.cur_epoch <= self.ignore_first_n_epochs:
            do_test = False
        elif self.untested_epochs >= self.test_at_least_every_n_epochs:
            do_test = True
        elif self.test_on_new_best_validation and is_new_best_val_model:
            if self.untested_epochs < self.test_at_most_every_n_epochs:
                do_test = False
            else:
                do_test = True

        self.cur_epoch += 1
        if do_test:
            self.untested_epochs = 1
        else:
            self.untested_epochs += 1
        return do_test


class PredictionPostProcessCallback:
    def __init__(self,
                 variable_to_channel: Optional[Dict[str, Dict[str, int]]],
                 variables: List[str]
                 ):
        self.variable_to_channel = dict()
        cur = 0
        for var in variables:
            self.variable_to_channel[var] = {'start': cur, 'end': cur + variable_to_channel[var]['end']}
            cur += variable_to_channel[var]['end']

    def split_vector_by_variable(self,
                                 vector: Union[np.ndarray, torch.Tensor]
                                 ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        if isinstance(vector, dict):
            return vector
        splitted_vector = dict()
        for var_name, var_dict in self.variable_to_channel.items():
            splitted_vector[var_name] = vector[..., var_dict['start']:var_dict['end']]
        return splitted_vector

    def __call__(self, vector, *args, **kwargs):
        return self.split_vector_by_variable(vector)


