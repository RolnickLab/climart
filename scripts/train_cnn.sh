

python ../main.py --expID "" --model CNN --scheduler "expdecay" --lr 2e-4 --weight_decay 1e-6 --batch_size 128 \
  --dropout 0.0 --act GELU --optim Adam --loss mse --epochs 100 --workers 8 --in_normalize Z \
  --train_years "1999" --validation_years "2005" --net_norm none \
  --gap --gradient_clipping norm \
  --seed 7 \
  --load_val_into_mem \
  --load_train_into_mem \
  --wandb_mode disabled \


