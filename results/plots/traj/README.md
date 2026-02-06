Sample script to plot predicted trajectories after training synthetic experiments using Pyomo models.

```bash
python -m src.synthetic_data.plot_training_fit \
  --model_type pyomo \
  --data_type ho \ 
  --layer_width "[2,96,2]" \
  --penalty_lambda_reg 0.1 \
  --tol 1e-8 \
  --seed 42 --noise 0.2\
  --outdir results/plots/traj --x_ticks_endpoints
```