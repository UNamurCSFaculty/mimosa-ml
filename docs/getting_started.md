# Getting started

A complete fit-and-predict pipeline, from a synthetic dataset to posterior predictions.

```python
import jax
jax.config.update("jax_enable_x64", True)  # before any other jax/kernax/mimosa import
import jax.random as jr

from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel
from mimosa import (
	Dimensions, ModelConfig, Parameters, BasicModel, UnionGrid, build_parameters, generate_data,
)

# 20 tasks, 2 clusters, 1D inputs/outputs, 25 points per task, 50 grid points
dims = Dimensions(T=20, K=2, I=1, C=1, O=1, N=25, G=50)
config = ModelConfig(isotopic_tasks=False)

# The "base" mean/kernels: one set of hyperparameters, before any per-task/cluster batching.
base_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=1.0),
	task_kernel=VarianceKernel(0.1) * SEKernel(length_scale=2.0),
	noise_kernel=WhiteNoiseKernel(noise=0.01),
)

dataset, *_ = generate_data(jr.PRNGKey(0), dims, base_params, config, input_range=[(-2.5, 2.5)])

# `fit` expects parameters batched to the shapes `dims`/`config` imply, not the base ones.
init_params = build_parameters(base_params, dims, config)

# We define the grid of points we will train and predict on, and our model
grid = UnionGrid(dataset.inputs)
model = BasicModel(jr.PRNGKey(1), n_clusters=dims.K)

# --- Fitting --- #
hyperposterior, mixture, fitted_params = model.fit(dataset, grid, init_params, n_iter=50)

# --- Predicting --- #
predictions = model.predict(dataset, grid, mixture, fitted_params)
```

## The pieces

`Dimensions`
: How big everything is — `T` tasks, `K` clusters, `I` input dimensions, `C` channels, `O`
  correlated outputs, `N` points observed per task, `G` points in the grid.

`ModelConfig`
: Which hyperparameters are shared across tasks, clusters and channels, and whether tasks share
  input locations. See {doc}`examples/level1/configurations`.

`Parameters`
: The four model components: `cluster_mean` and `cluster_kernel` describe the mean-processes each
  cluster is centred on, `task_kernel` how tasks vary around their mean-process, and `noise_kernel`
  the observation noise.

`Grid`
: The input locations the mean-processes live on. `UnionGrid(dataset.inputs)` reuses every
  observed location; `RegularGrid` and `KMeansGrid` build smaller ones.

`BasicModel`
: `fit` returns the fitted hyperposterior, the cluster mixture and the optimised parameters;
  `predict` turns those into a posterior distribution per task, cluster and channel.

## Using your own data

Swap `generate_data` for `load_csv`, and set `Dimensions` to match what you loaded:

```python
from mimosa import load_csv

dataset = load_csv("my_data.csv")
```

For information on the csv format to follow, look at {doc}`examples/level1/run_your_data`.

The full walkthrough — missing points, clustering, plotting and sampling from a prediction — is in
{doc}`examples/level1/basic_example`.

```{tip}
Read {doc}`sharp_bits` before your first real fit. Most early surprises with Mimosa are either
numerical precision or JAX semantics.
```
