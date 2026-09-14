# %% tags=["remove-cell"]
import importlib.util, subprocess, sys
from pathlib import Path
if importlib.util.find_spec("mimosa") is None:
	# When running in Colab, you can select a GPU for execution and un-comment the next line
	# subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[cuda]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mimosa-ml"], check=True)
# On Colab the notebook runs from /content, where the example's data folder does not exist.
# The docs build runs each notebook from its own level folder, while the data folder is shared at
# docs/examples/data. On Colab the notebook runs from /content, where neither exists.
import os
if Path("../data").is_dir():
    os.chdir("..")
Path("data").mkdir(exist_ok=True)

# %% [markdown]
"""
# Basic usage of Mimosa

This example walks through the full pipeline on a synthetic dataset: configure the dimensions,
generate data and remove some points at random, fit a `BasicModel`, then predict a task and sample
from that prediction.

Use the "launch" button to run it interactively in Colab or clone the repository and 
run the `examples/level1/basic_example.py` script!
"""

# %% [markdown]
"""
## Getting started

First, the usual imports and configs:
"""

# %%
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dimensions, ModelConfig, DataRemovalConfig, Dataset, GPDataset, Parameters, GPParameters,
	BasicModel, GPModel, UnionGrid, RegularGrid, MergedGrid,
	generate_data, RandomDataRemover, save_csv, load_csv, build_parameters, build_gp_parameters, sample_gp,
	plot_dataset, plot_clusters, plot_single_task_prediction,
)

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi']=300
jax.devices()

# %% [markdown]
"""
Throughout the framework, Mimosa keeps track of the dimensions of the datasets and the shape of the
model parameters via two objects: `Dimensions` and `ModelConfig`. Here we also generate the data
ourselves, so these two objects describe both the data we create and the model we will fit on it.
"""

# %% 1. Configuration
# Dimensions: T tasks, K clusters, I input dims, C channels, O correlated outputs, N points
# observed per task, G points in the full grid.
dims = Dimensions(T=32, K=2, I=1, C=2, O=1, N=50, G=150)

# ModelConfig controls which hyperparameters are shared (across tasks/clusters/channels/outputs) and
# whether tasks/outputs share input locations.
model_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=True,
)

# How many points to remove at random per task, to simulate missing data.
removal_config = DataRemovalConfig(max_missing=5, random_missing_count=True, same_missing_across_channels=False)

# %% [markdown]
"""
A model is described by 4 parameters:
* `cluster_mean`: the mean function of the mean-processes, i.e. the long-term trend clusters are
  centered on
* `cluster_kernel`: the kernel of the mean-processes, i.e. their "shape" (smooth, wiggly, periodic...)
* `task_kernel`: the kernel of the tasks, i.e. how they vary around their mean-process
* `noise_kernel`: the noise on the observed points

Below, they play the role of the *true* parameters used to synthesise the dataset.
"""

# %% 2. Generative parameters
# These are the "true" parameters used to synthesise the toy dataset below. Swap any kernel/mean for
# another kernax one (e.g. MaternKernel, PeriodicKernel, LinearMean, ...) to change the shape of the
# data generated.
true_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=.5),
	task_kernel=VarianceKernel(1.0) * SEKernel(length_scale=.4),
	noise_kernel=WhiteNoiseKernel(noise=.05),
)

# %% 3. Generate synthetic data, then remove points at random
key, gen_key, removal_key = jr.split(key, 3)

dataset, grid, hyperprior, true_mixture, true_params, cluster_means, tasks = generate_data(
	gen_key, dims, true_params, model_config, input_range=[(-2.5, 2.5)]
)
dataset = RandomDataRemover()(removal_key, dataset, removal_config)

# %% 3bis. Alternatively, you can load a dataset from a local file through load_csv.
# Check load_csv's doc or open the csv file to see the expected file format.
save_csv("data/dummy.csv", dataset)
dataset = load_csv("data/dummy.csv")

# %% 4. Plot the raw dataset (coloured by each task's true cluster)
fig, ax = plot_dataset(dataset, dims, mixture=true_mixture, figsize=(8 * dims.C, 6))
fig.suptitle("Synthetic dataset (colored by true cluster)")
plt.show()

# %% [markdown]
"""
## Training the model

The model knows nothing about the parameters above: it starts from its own guess and optimises it.
The initial values do not matter too much, but their *structure* does, which is what
`build_parameters` takes care of.
"""

# %% 5. Instantiate the model
# n_clusters can differ from the true K above (the model doesn't know it).
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=dims.K)

# Starting guess for the parameters to fit. In practice these would be a rough, uninformed guess
# rather than the true generative ones — feel free to try different starting kernels/values here.
init_params = Parameters(
		cluster_mean=ZeroMean(),
		cluster_kernel=VarianceKernel(2.0) * SEKernel(length_scale=1.),
		task_kernel=VarianceKernel(.5) * SEKernel(length_scale=.5),
		noise_kernel=WhiteNoiseKernel(noise=0.5))

# build_parameters batches the base kernels/mean below to match model_config's sharing structure
# so their shapes line up with what model.fit expects.
init_params = build_parameters(init_params, dims, model_config)

# %% 6. Fit
# Grid construction (union of every task's input points) isn't jit-compatible, so it's built once
# outside of fit/predict. Swap UnionGrid for another GridBuilder to change how the grid is built.
fitted_grid = UnionGrid()(dataset.inputs)

hyperposterior, fitted_mixture, fitted_params = model.fit(dataset, fitted_grid, init_params, n_iter=50)

# %% 7. Plot the fitted clusters (mean-processes)
fig, ax = plot_dataset(dataset, dims, mixture=fitted_mixture, figsize=(8 * dims.C, 6), alpha=.1)
fig, ax = plot_clusters(fitted_grid, dims, hyperposterior=hyperposterior, figsize=(8 * dims.C, 6), fig=fig, ax=ax)
fig.suptitle("Fitted clusters (mean-processes) on the dataset")
plt.show()

# %% [markdown]
"""
Note: you might have noticed that the dataset "swapped colors" compared to the previous plot. This
is because the model doesn't know anything about the true mixture at the start, so it can swap indexes
of each cluster. In other terms, even when training goes perfectly, `fitted_mixture` can be a
"swapped" version of the `true_mixture`.
"""

# %% [markdown]
"""
## Predicting  

Predictions are multimodal: the model returns one Gaussian process per cluster for each task. Here we
simply keep the one of the task's **most probable cluster**.

Try changing `t_id` to see another task, and `k_id` to see what the prediction would look like if the
task belonged to that cluster instead!

To see what a prediction is actually worth, we hide part of one task and compare the prediction to
the points we hid. For simplicity, we predict on the fitting data, but in prectice you can have two
different sets of tasks.
"""

# %% 7bis. Hide part of the task we predict
# First 2/3 of the domain: a few points dropped at random, the usual "missing data" case. Last 1/3:
# every point dropped, so that stretch is a genuine forecast. Only this task is thinned.
t_id, c_id = 0, 0
key, hide_key = jr.split(key)

x_task = dataset.inputs[t_id, :, 0]
forecast_from = x_task.min() + 2 / 3 * (x_task.max() - x_task.min())
hidden = jnp.where(x_task < forecast_from, jr.uniform(hide_key, x_task.shape) < .2, True)
observed = Dataset(dataset.inputs, dataset.outputs.at[t_id, hidden].set(jnp.nan))


def show_held_out(ax):
	"""Overlay the hidden points (red crosses) and mark where the forecast starts."""
	ax[0, 0].scatter(x_task[hidden], dataset.outputs[t_id, hidden, c_id],
					 color="tab:red", marker="x", s=25, zorder=3, label="held out")
	ax[0, 0].axvline(forecast_from, color="gray", linestyle=":")


# %% 8. Predict, conditioning on the thinned task only
predictions = model.predict(observed, fitted_grid, fitted_mixture, fitted_params)  # MultivariateNormal, batched (T, K, C, O*G)

k_id = int(fitted_mixture.assignments[t_id])  # task's dominant cluster
prediction = predictions[t_id, k_id, c_id]

# %% 9. Plot the prediction: observed points, cluster means, and predictive mean + confidence interval
fig, ax = plot_single_task_prediction(
	observed, fitted_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, prediction=prediction, figsize=(8 * dims.C, 6)
)
show_held_out(ax)
fig.suptitle(f"Prediction — task {t_id}, channel {c_id}")
plt.show()

# %% [markdown]
"""
It's often better to look at **samples** of the predictive distribution to get a true
feeling about the actual shape of the predicted function.
"""

# %% 10. Draw samples from the prediction and plot them alongside it
key, sample_key = jr.split(key)
n_samples = 32
sample_keys = jr.split(sample_key, n_samples)
samples = vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(sample_keys)  # (S, O*G)

fig, ax = plot_single_task_prediction(
	observed, fitted_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, samples=samples, figsize=(8 * dims.C, 6)
)
show_held_out(ax)
fig.suptitle(f"Prediction samples — task {t_id}, channel {c_id}")
plt.show()

# %% [markdown]
"""
## Predicting on another grid

You might have noticed that our predictions look a bit...spiky. This is because we compute them on the 
UnionGrid of the inputs in the dataset. However, the underlying object is an *infinite function distribution*,
we can sample it at any input location!

`fitted_grid` is the union of the observed locations, which is what fitting needs. To predict 
somewhere else, merge that grid with the one you actually want, and run the prediction once on the merged pool.

Using the `.marginal()` methods of the MultiVariateNormal providing the points from
the second grid (aka `merged_grid.sources[1]`) allows you te retrieve the portion of
the prediction related to this exact grid.
"""

# %% 11. Merge an evenly-spaced grid into the fitted one
# `bounds` is one (min, max) per input dimension -- here the range the data was generated over.
regular_grid = RegularGrid(bounds=((-2.5, 2.5),), n_points=300)(dataset.inputs)
merged_grid = MergedGrid(fitted_grid, regular_grid)

# %% 12. Predict on the merged grid, then marginalise onto the regular one
# `model.predict` would do both steps at once, but the hyperposterior is wanted for the plot below.
merged_hyperposterior = model.hyperpost(observed, merged_grid, fitted_mixture, fitted_params)
merged_predictions = model.predictor(observed, merged_grid, merged_hyperposterior, fitted_params)

regular_hyperposterior = merged_hyperposterior.marginal(merged_grid.sources[1])
regular_prediction = merged_predictions.marginal(merged_grid.sources[1])[t_id, k_id, c_id]

# %% 13. Plot the same task, now on the regular grid
fig, ax = plot_single_task_prediction(
	observed, regular_grid, dims, regular_hyperposterior, fitted_mixture, t_id, c_id,
	prediction=regular_prediction, figsize=(8 * dims.C, 6)
)
show_held_out(ax)
fig.suptitle(f"Prediction on a regular grid — task {t_id}, channel {c_id}")
plt.show()

# %% [markdown]
"""
## Comparing against a single-task GP

Everything above leaned on the other 31 tasks. To see what that buys, fit a plain Gaussian process
— `GPModel` — on this task's remaining points *alone*, and predict on the same grid.

It gets the same starting hyperparameters as Mimosa did and is free to optimise them on this task,
so the comparison is about the **information** each model has, not about tuning: the vanilla GP has
no mean-process to fall back on, only its own prior.
"""

# %% 14. Fit a vanilla GP on this task's remaining points only
# `GPDataset` holds one task, so it has no task axis. Rows with no observation left on any channel
# carry no information and are dropped rather than kept as all-NaN.
kept = ~jnp.isnan(observed.outputs[t_id]).all(axis=-1)
gp_dataset = GPDataset(inputs=observed.inputs[t_id][kept], outputs=observed.outputs[t_id][kept])

# Same base mean/kernels as the model's starting guess, batched over channels instead of over
# tasks/clusters/channels -- a single-task GP has only the channel axis.
gp_params = build_gp_parameters(
	GPParameters(
		mean=ZeroMean(),
		kernel=VarianceKernel(.5) * SEKernel(length_scale=.5),
		noise_kernel=WhiteNoiseKernel(noise=0.5),
	),
	n_channels=dims.C,
	shared_channel_hps=model_config.shared_channel_hps,
)

gp_model = GPModel()
gp_params = gp_model.fit(gp_dataset, gp_params)

# %% 15. Predict with the vanilla GP on the same regular grid
gp_prediction = gp_model.predict(gp_dataset, regular_grid, gp_params)[c_id]

# %% 16. Plot both predictions on the same axes
fig, ax = plot_single_task_prediction(
	observed, regular_grid, dims, regular_hyperposterior, fitted_mixture, t_id, c_id,
	prediction=regular_prediction, figsize=(8 * dims.C, 6)
)
show_held_out(ax)

x_grid = regular_grid.points[:, 0]
gp_mean = gp_prediction.mean
gp_std = jnp.sqrt(jnp.diagonal(gp_prediction.covariance))
ax[0, 0].plot(x_grid, gp_mean, color="tab:green", label="vanilla GP")
ax[0, 0].fill_between(x_grid, gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std,
					  color="tab:green", alpha=.15, linewidth=0)
ax[0, 0].legend(loc="upper left", fontsize=8)

fig.suptitle(f"Mimosa (black) vs single-task GP (green) — task {t_id}, channel {c_id}")
plt.show()
