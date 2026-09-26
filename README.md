# Robust Gradient Estimator Experiments

Research code for comparing geometric-median (GeoMed) and radius-based robust
gradient aggregation under dataset-level corruption. The experiments use
paired random seeds: within each condition, both aggregators receive the same
corrupted dataset, initialization, and sampled observations.

The repository contains code only. Generated results, temporary files, MNIST
data, and manuscript sources are intentionally excluded.

## Setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The MNIST scripts expect the standard Keras `mnist.npz` archive, containing
`x_train`, `y_train`, `x_test`, and `y_test`. By default they read
`~/.keras/datasets/mnist.npz`; use `--mnist-path` to select another location.

## Main workshop experiments

The ten-class trajectory experiment with 10% fixed reverse-image corruption,
minibatch size 8, learning rate 0.01, and five paired seeds can be run with:

```bash
python experiments/compare_multiclass_trajectories.py \
  --epsilon 0.1 \
  --batch-size 8 \
  --seeds 0 1 2 3 4 \
  --steps 10000 \
  --learning-rate 0.01 \
  --budget 640 \
  --checkpoints 0 25 50 100 200 300 500 750 1000 1500 2000 3000 4000 5000 6000 7500 10000 \
  --output-dir output/experiments/multiclass_eps0p1_lr0p01_zero_init
```

The matched minibatch-size sweep at iteration 5,000 can be run with:

```bash
python experiments/compare_multiclass_batch_sizes.py \
  --epsilon 0.1 \
  --batch-sizes 8 16 32 64 80 128 \
  --seeds 0 1 2 3 4 \
  --steps 5000 \
  --summary-iteration 5000 \
  --learning-rate 0.01 \
  --budget 640 \
  --output-dir output/experiments/multiclass_batch_sweep_eps0p1_lr0p01
```

After both runs finish, generate the paper figure with:

```bash
python experiments/make_workshop_three_panel_figure.py
```

All confidence intervals displayed by these scripts are two-sided 95%
Student-*t* intervals across paired seeds.

## Other experiments

- `reproduce_small_b_accuracy.py`: binary digit-1-versus-rest endpoint study.
- `compare_accuracy_trajectories.py`: binary MNIST training trajectories.
- `compare_static_corruptions.py`: ten-class MNIST under several fixed attacks.
- `compare_gaussian_static_corruptions.py`: balanced synthetic Gaussian study.
- `plot_cross_epsilon_trajectories.py`: cross-corruption trajectory plots.

Run any experiment with `--help` to see its configurable parameters. Output
directories are resumable: completed method/seed conditions are read from the
existing raw CSV and skipped.
