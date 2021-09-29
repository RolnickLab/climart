import torch

from climart.data_wrangling.constants import get_data_dims, LAYERS
from climart.models.column_handler import ColumnPreprocesser
from climart.models.interface import get_model
from climart.utils.postprocessing import restore_ckpt_from_wandb

run_id = "3n1if6lg"
saved_model = restore_ckpt_from_wandb(run_id, run_path="ecc-mila7/RT+ML_Dataset_paper", load=True)

params = saved_model['hyper_params']
net_params = saved_model['model_params']

spatial_dim, in_dim = get_data_dims(params['exp_type'])

cp = ColumnPreprocesser(
    n_layers=spatial_dim[LAYERS], input_dims=in_dim, **params['preprocessing_dict']
)
input_transform = cp.get_preprocesser

model = get_model(params['model'], column_preprocesser=cp, **net_params)
model.load_state_dict(saved_model['model'])
learned_adj = model.learned_adj





