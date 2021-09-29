import os

import torch
from main import main
from climart.utils.hyperparams_and_args import get_argparser

ckpt_dir = "/home/mila/s/salva.ruhling-cachay/RT-DL/scripts/out"
checkpoints = [
 #   '0.6099valMAE_145ep_GN+READOUT_1990+1999+2003train_2005val_Z_7seed___23h26m_on_Aug_11',
 #   '0.8350valMAE_148ep_GCN+READOUT_1990+1999+2003train_2005val_Z_7seed___01h10m_on_Aug_12',
 #   '0.7648valMAE_144ep_GCN+READOUT_1990+1999+2003train_2005val_Z_7seed___00h28m_on_Aug_12',
 #   '1.2602valMAE_150ep_GCN+READOUT_1990+1999+2003train_2005val_Z_7seed___20h21m_on_Aug_11',
 #   '0.9853valMAE_149ep_GN+READOUT_1990+1999+2003train_2005val_Z_7seed___21h05m_on_Aug_10',
 #   '0.8255valMAE_114ep_GN+READOUT_1990+1999+2003train_2005val_Z_7seed___21h07m_on_Aug_10',
 #   '1.1793valMAE_138ep_GN+READOUT_1979+1989-90+1999+2003train_2005val_Z_7seed___12h30m_on_Aug_09',
    '1.6397valMAE_133ep_MLP_1990+1999+2003train_2005val_Z_7seed___23h03m_on_Aug_08',
]

ckpt_dir2 = "/home/mila/s/salva.ruhling-cachay/RT-DL/out"
checkpoints2 = [
#    '1.2432valMAE_120ep_GN+READOUT_1990+1999+2003train_2005val_Z_7seed___12h00m_on_Aug_08',
    '1.4350valMAE_148ep_MLP_1990+1999+2003train_2005val_Z_7seed___03h43m_on_Aug_08'
]

for ckpt_file in checkpoints2:
    ckpt_file = os.path.join(ckpt_dir2, ckpt_file + '.pkl')
    saved_model = torch.load(ckpt_file)
    if saved_model['validation_stat'] > 1.8:
        print('Skipping saved model due to high validation stat=', saved_model['validation_stat'])
        continue

    params = saved_model['hyper_params']
    net_params = saved_model['model_params']
    ps, _, other_args = get_argparser()
    for p in ps.keys():
        if p not in params.keys():
            params[p] = ps[p]
    try:
        main(params, net_params, other_args,            model_state_dict=saved_model['model'], only_final_eval=True)
    except RuntimeError as e:
        print(e)
        pass
