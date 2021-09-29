

python ../main.py \
  --model "GCN+Readout" \
  --target_type "shortwave" \
  --workers 8 \
  --expID "" \
  --hidden_dims 128 128 128 \
  --scheduler "expdecay" \
  --lr 2e-4 \
  --weight_decay 1e-6 \
  --batch_size 128 \
  --act "GELU" \
  --net_normalization "none" \
  --gradient_clipping "Norm" \
  --clip 1.0 \
  --epochs 100 \
  --residual \
  --improved_self_loops \
  --preprocessing "mlp_projection" \
  --projector_net_normalization "layer_norm" \
  --graph_pooling "mean" \
  --drop_last_level \
  --in_normalize Z \
  --train_years "1999" \
  --validation_years "2005" \
  --load_val_into_mem \
  --load_train_into_mem \
  --wandb_mode disabled \
 # --learn_edge_structure \
