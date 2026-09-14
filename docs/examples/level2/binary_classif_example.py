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
# Binary classification with Laplace matching

Mimosa only knows how to handle *Gaussian* observations. So how do we cluster tasks made of
booleans? By turning them into Gaussian ones first, through **Laplace matching**: observations are
binned along the input axis, each bin's Beta posterior over the success probability is
moment-matched with a Gaussian over its log-odds, and the matched variance is given to the model as
a known, non-trainable noise.

Everything in between is the ordinary pipeline of [the basic example](basic_example.ipynb). Only the
last step differs: predictions live in log-odds space and are mapped back to probabilities.

Written using jupytext's py:percent format. This script can be run cell-by-cell or as a usual Python
script.
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
	Dataset, Dimensions, ModelConfig, Parameters,
	BasicModel, UnionGrid, BinomialLaplaceApproximator,
	generate_data, build_parameters, known_noise_kernel, sample_gp,
	plot_dataset, plot_clusters,
)

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi'] = 300
jax.devices()

# %% 1. Configuration
# The latent (continuous) processes are generated exactly as in basic_example.py; only their sign
# reaches us, through a Bernoulli draw. N is large and the input range narrow on purpose: Laplace
# matching aggregates the booleans into bins of width INTERVAL, and a bin needs enough draws for its
# success rate to mean anything.
dims = Dimensions(T=32, K=2, I=1, C=1, O=1, N=300, G=600)
INPUT_RANGE = (-5., 5.)
INTERVAL = 0.5
PRIOR_COUNT = .1

# When binning, tasks need not share their input locations (isotopic_tasks=False): Laplace matching
# puts them on a lattice shared by every task, so the wrapped dataset comes out isotopic whatever we
# start from. With INTERVAL=0 there is no lattice -- the wrapped points are the observed ones -- so
# the tasks have to share them already, or the union grid would hold T*N distinct points.
gen_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=INTERVAL == 0.,
)

# %% 2. Generative parameters
# The "true" parameters of the latent processes. Their scale sets how extreme the success
# probabilities get: a latent value of +-2 already means a ~88% / ~12% chance of a True.
true_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(2.0) * SEKernel(length_scale=1.5),
	task_kernel=VarianceKernel(0.5) * SEKernel(length_scale=1.0),
	noise_kernel=WhiteNoiseKernel(noise=1e-3),
)

# %% [markdown]
"""
The dataset is synthesised in two steps: first the usual continuous processes, then a coin flip at
each point whose bias is given by the process. Only the flips are observed -- the latent processes
are what the model will have to recover.
"""

# %% 3. Generate the latent processes, then Bernoulli-sample them
key, gen_key, bernoulli_key = jr.split(key, 3)

latent, _, _, true_mixture, true_params, _, _ = generate_data(
	gen_key, dims, true_params, gen_config, input_range=[INPUT_RANGE]
)

true_probs = jax.nn.sigmoid(latent.outputs)  # (T, N, C), the probability each draw came from
booleans = jr.bernoulli(bernoulli_key, true_probs).astype(float)
dataset = Dataset(inputs=latent.inputs, outputs=booleans)

# %% 4. Plot the raw booleans (coloured by each task's true cluster)
fig, ax = plot_dataset(dataset, dims, mixture=true_mixture, figsize=(10, 6), alpha=.05)
ax[0, 0].set_yticks([0, 1], ["False", "True"])
fig.suptitle("Raw boolean observations (colored by true cluster)")
plt.show()

# %% [markdown]
"""
## Laplace matching

Here is the interesting part. A single boolean says almost nothing, but a group of them has a
success rate, and a success rate can be described by a Gaussian over its log-odds. `wrap` does
exactly that, and returns a dataset the model can be fitted on as usual, plus the variance of each
group in `known_output_noise`.

The bin width is the trade-off to watch: wider bins mean more confident Gaussians, but a coarser
view of the input axis.
"""

# %% 5. Laplace matching
# `interval` bins the inputs on a lattice shared by every task (bin edges of width INTERVAL, anchored
# at the global input minimum), so the wrapped inputs collapse to a single (1, B, I) block. Each
# (task, bin) group becomes one Gaussian: its mean lands in `outputs`, its variance in
# `known_output_noise`. A bin holding none of a task's points becomes NaN padding for that task.
approximator = BinomialLaplaceApproximator(interval=INTERVAL, prior_count=PRIOR_COUNT)
wrapped = approximator.wrap(dataset)

n_groups = wrapped.outputs.shape[1]
print(f"{dims.N} points per task -> {n_groups} groups; "
	  f"{float(jnp.mean(jnp.isnan(wrapped.outputs))):.1%} of the groups are empty for their task")

# The grid is built here rather than at fitting time, because the wrapped dataset's dimensions are
# read off it: binning collapses every task onto the lattice, INTERVAL=0 keeps the observed points.
fitted_grid = UnionGrid()(wrapped.inputs)

# The wrapped dataset has its own shape and its own sharing structure -- both taken from the data
# rather than assumed, since they differ between the binned and the unbinned case.
wrapped_dims = Dimensions(T=dims.T, K=dims.K, I=dims.I, C=dims.C, O=dims.O,
						  N=n_groups, G=fitted_grid.points.shape[0])
model_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=wrapped.inputs.shape[0] == 1,
)

fig, ax = plot_dataset(wrapped, wrapped_dims, mixture=true_mixture, figsize=(10, 6), alpha=.3)
ax[0, 0].set_ylabel("log-odds")
fig.suptitle("Laplace-matched dataset (log-odds of each bin's success rate)")
plt.show()

# %% 6. Instantiate the model
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=wrapped_dims.K)

init_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=2.),
	task_kernel=VarianceKernel(1.0) * SEKernel(length_scale=1.),
	noise_kernel=WhiteNoiseKernel(noise=.5))
init_params = build_parameters(init_params, wrapped_dims, model_config)

# The per-bin variances Laplace matching produced enter the model here, as a non-trainable
# white-noise kernel added on top of the (trainable) noise kernel.
init_params = Parameters(
	cluster_mean=init_params.cluster_mean,
	cluster_kernel=init_params.cluster_kernel,
	task_kernel=init_params.task_kernel,
	noise_kernel=init_params.noise_kernel + known_noise_kernel(
		wrapped.known_output_noise, wrapped_dims, model_config))

# %% 7. Fit
hyperposterior, fitted_mixture, fitted_params = model.fit(wrapped, fitted_grid, init_params, n_iter=50)

# Cluster *labels* are arbitrary, so compare the clusterings rather than the labels: how often do
# two tasks end up together in the fit exactly when they were together in the truth?
together_true = true_mixture.assignments[:, None] == true_mixture.assignments[None, :]
together_fit = fitted_mixture.assignments[:, None] == fitted_mixture.assignments[None, :]
print(f"pairs of tasks clustered as in the truth: {float(jnp.mean(together_true == together_fit)):.0%}")

# %% 8. Plot the fitted clusters, in log-odds space
fig, ax = plot_dataset(wrapped, wrapped_dims, mixture=true_mixture, figsize=(10, 6), alpha=.1)
fig, ax = plot_clusters(fitted_grid, wrapped_dims, hyperposterior=hyperposterior, figsize=(10, 6), fig=fig, ax=ax)
ax[0, 0].set_ylabel("log-odds")
fig.suptitle("Fitted clusters (mean-processes), in log-odds space")
plt.show()

# %% 9. Predict, then sample
predictions = model.predict(wrapped, fitted_grid, fitted_mixture, fitted_params)

t_id, c_id = 0, 0
k_id = int(fitted_mixture.assignments[t_id])
prediction = predictions[t_id, k_id, c_id]

key, sample_key = jr.split(key)
n_samples = 256
samples = vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(jr.split(sample_key, n_samples))

# %% [markdown]
"""
## Back to probabilities

The model predicts log-odds, so the last step is to map them back with `unwrap`. Because that
mapping is non-linear, it must be applied to *samples* rather than to the predictive mean.
"""

# %% 10. Unwrap the samples back to probabilities, and compare with the truth
# `unwrap` is the elementwise inverse link (a sigmoid here), so it has to be applied to the *samples*
# rather than to the predictive mean: it does not commute with taking an average.
probability_samples = approximator.unwrap(samples)  # (S, B)
lower, median, upper = jnp.percentile(probability_samples, jnp.array([2.5, 50., 97.5]), axis=0)

order = jnp.argsort(latent.inputs[t_id, :, 0])
true_x, true_p = latent.inputs[t_id, order, 0], true_probs[t_id, order, c_id]
observed_x, observed_y = dataset.inputs[t_id, :, 0], dataset.outputs[t_id, :, c_id]
grid_x = fitted_grid.points[:, 0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(observed_x, observed_y, s=12, alpha=.25, color="black", label="observed booleans")
ax.plot(true_x, true_p, color="C1", lw=2, label="true probability")
ax.plot(grid_x, median, color="C0", lw=2, label="predicted probability (median of samples)")
ax.fill_between(grid_x, lower, upper, color="C0", alpha=.2, label="95% interval")
ax.set_xlabel("input")
ax.set_ylabel("P(True)")
ax.set_yticks([0, .5, 1])
ax.legend(loc="best")
fig.suptitle(f"Predicted vs. true probability — task {t_id}, channel {c_id}")
plt.show()

# %% 11. The same prediction, drawn as bare samples
# Each faint line is one draw from the predictive Gaussian, mapped through the sigmoid. The
# percentile band above summarises exactly these curves; drawn individually they also show how a
# single realisation behaves — in particular that the sigmoid pins samples against 0 and 1 wherever
# the log-odds prediction is confident.
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(grid_x, probability_samples.T, color="C0", lw=.6, alpha=.15)
ax.plot([], [], color="C0", lw=.6, alpha=.6, label=f"{n_samples} samples")  # one proxy for the legend
ax.scatter(observed_x, observed_y, s=12, alpha=.25, color="black", label="observed booleans")
ax.plot(true_x, true_p, color="C1", lw=2, label="true probability")
ax.set_xlabel("input")
ax.set_ylabel("P(True)")
ax.set_yticks([0, .5, 1])
ax.legend(loc="best")
fig.suptitle(f"Prediction samples — task {t_id}, channel {c_id}")
plt.show()
