# MLP

python ../main.py \
  --expID "" \
  --model "MLP" \
  --scheduler "expdecay" \
  --exp_type "pristine" \
  --target_type "shortwave" \
  --lr 2e-4 \
  --weight_decay 1e-6 \
  --batch_size 128 \
  --net_normalization "layer_norm" \
  --dropout 0.0 \
  --act "GELU" \
  --optim Adam \
  --gradient_clipping "Norm" \
  --clip 1 \
  --epochs 100 \
  --workers 6 \
  --hidden_dims 512 256 256 \
  --in_normalize Z \
  --train_years "1999" \
  --validation_years "2005" \
  --seed 7 \
  --load_val_into_mem \
  --load_train_into_mem \
  --wandb_mode disabled \
