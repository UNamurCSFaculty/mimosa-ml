# %% tags=["remove-cell"]
import importlib.util, os, subprocess, sys
from pathlib import Path
from urllib.request import urlretrieve

if importlib.util.find_spec("mimosa") is None:
	# When running in Colab, you can select a GPU for execution and un-comment the next line
	# subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[cuda]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mimosa-ml"], check=True)

# The docs build runs each notebook from its own level folder, while the data folder is shared at
# docs/examples/data. On Colab the notebook runs from /content, where neither exists.
if Path("../data").is_dir():
    os.chdir("..")
Path("data").mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/UNamurCSFaculty/mimosa-ml/main/docs/examples/data"
for name in ("car_trajectories_200.csv",):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))

# %% [markdown]
"""
# Linking channels through the mixture: car trajectories in a roundabout

284 real cars drive through the same roundabout. Each one was tracked as an (X, Y) position over
time. Can we exploit the mixture shared across the two channels (X and Y) to cluster and predict
better than two uncorrelated Gaussian processes would?

The trajectories come from [openDD](https://arxiv.org/abs/2007.08463) (Breuer et al., *openDD: A
Large-Scale Roundabout Drone Dataset*, IEEE ITSC 2020): 84,774 drone-tracked vehicle trajectories
across seven real roundabouts in Wolfsburg and Ingolstadt, Germany, over 62+ hours of recordings.
This example uses a subset of one roundabout's trajectories.
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
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from kernax import AffineMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dataset, Dimensions, ModelConfig, Parameters,
	BasicModel, UnionGrid, Mixture,
	load_csv, build_parameters, sample_gp,
	plot_dataset, plot_clusters, plot_single_task_prediction,
)
from mimosa.mixture import MixtureInitialiser, _summary_statistics
from mimosa.kmeans import soft_kmeans

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi'] = 300

K, STIFFNESS, N_RESTARTS_KMEANS = 6, 50.0, 64

# %% [markdown]
"""
## The dataset

200 of the 284 car trajectories through the roundabout (the openDD dataset, cited above), already
aligned onto a common time grid: X and Y position over time, for each car. We'll use "car" and
"task" interchangeably from here on. 
"""

# %%
dataset = load_csv("data/car_trajectories_200.csv")
T, N = dataset.outputs.shape[0], dataset.inputs.shape[1]
x, y = np.asarray(dataset.outputs[..., 0]), np.asarray(dataset.outputs[..., 1])

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x.T, y.T, color="0.4", alpha=0.15, linewidth=0.8)
ax.set_xlabel("X (rescaled)"); ax.set_ylabel("Y (rescaled)"); ax.set_aspect("equal")
ax.set_title(f"{T} real car trajectories through a roundabout (openDD dataset) -- raw, unlabeled.")
plt.show()

# %% [markdown]
"""
## What's a channel?

X and Y above are two **channels**: independent scalar signals measured on every task, each with
its own Gaussian Process, never cross-covaried. An **output**, by contrast, is one of `O`
*correlated* dimensions of a single field, modeled jointly so their covariance captures how they
move together. Channels don't talk to each other through covariance at all -- they only interact
through the mixture: tasks are clustered once, from the sum of every channel's log-likelihood.

This example is only about `C`. For `O`, see [the multi-output example](multi_output_example.ipynb);
for the ordinary fit/predict pipeline this one builds on, see [the basic example](../level1/basic_example.ipynb).

## Initialisation
"""

# %%
class StiffKMeansInitialiser(MixtureInitialiser):
	"""`KMeansMixtureInitialiser`'s own recipe, with `stiffness` exposed instead of fixed at 1.0."""
	prng_key: jax.Array
	n_clusters: int
	stiffness: float
	n_restarts: int

	def __call__(self, dataset):
		features = jnp.nan_to_num(_summary_statistics(dataset.outputs))
		_, resp = soft_kmeans(
			self.prng_key, features, self.n_clusters, stiffness=self.stiffness, n_restarts=self.n_restarts,
		)
		return Mixture(responsibilities=resp)

# %% [markdown]
"""

## Configuring and fitting

The model is configured the same way as any other mimosa model. Channel handling introduces one new
switch, `shared_channel_hps`, controlling whether channels share hyperparameters across the mixture
or stay fully independent.
"""

# %%
dims = Dimensions(T=T, K=K, I=1, C=2, O=1, N=N, G=N)  # C represents the number of channels
config = ModelConfig(
	shared_task_hps=True, shared_cluster_hps=False,
	shared_channel_hps=True, cluster_specific_task_hps=True,
	isotopic_tasks=True,
)
params = build_parameters(Parameters(
	cluster_mean=AffineMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=0.2),
	task_kernel=VarianceKernel(0.01) * SEKernel(length_scale=0.2),
	noise_kernel=WhiteNoiseKernel(noise=0.001),
), dims, config)
grid = UnionGrid(dataset.inputs)

key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=K)
model = eqx.tree_at(
	lambda m: m.mixture_initialiser, model,
	StiffKMeansInitialiser(prng_key=model_key, n_clusters=K, stiffness=STIFFNESS, n_restarts=N_RESTARTS_KMEANS),
)
hyperposterior, fitted_mixture, fitted_params = model.fit(
	dataset, grid, params, n_iter=25, freeze_task_parameters=True,
)

# %% [markdown]
"""
X(t)/Y(t) view, one subplot per channel:
"""

# %%
fig, ax = plot_dataset(dataset, dims, mixture=fitted_mixture, alpha=0.4, figsize=(8 * dims.C, 4))
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior, fig=fig, ax=ax)
fig.suptitle(f"{T} cars, clustered jointly on (X, Y).")
plt.show()

# %% [markdown]
"""
## Pulling channels apart

The same fit, back in (X, Y) space, where the clusters are readable as actual routes through the
roundabout:
"""

# %%
assignments = np.asarray(fitted_mixture.assignments)
cluster_sizes = np.bincount(assignments, minlength=K)
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(7, 7))
for k in range(K):
	members = assignments == k
	ax.plot(x[members].T, y[members].T, color=cmap(k), alpha=0.4, linewidth=0.9)
ax.set_aspect("equal")
ax.set_xlabel("X (rescaled)"); ax.set_ylabel("Y (rescaled)")
ax.set_title(f"{K} routes through the roundabout, found from (X, Y) jointly.")
plt.show()
print("cluster sizes:", cluster_sizes.tolist())




# %% [markdown]
"""


## Predicting

`model.predict` gives an actual posterior distribution for one task. 

First, mask the second half of **both** channels, for a task than, in the second plot, mask only half of the Y axis for the same task. 
"""

# %%
cutoff = 64
task_kernel = fitted_params.task_kernel + fitted_params.noise_kernel

def mask_and_update(mask_channels, task_ids=slice(None)):
	"""One E-step against the already-fitted hyperposterior, masking the second half of
	`mask_channels` for `task_ids` (every task by default)."""
	masked = np.asarray(dataset.outputs).copy()
	masked[task_ids, cutoff:, mask_channels] = np.nan
	dataset_masked = Dataset(inputs=dataset.inputs, outputs=jnp.asarray(masked))
	mixture_masked = model.mixture_updater(
		dataset_masked, grid, task_kernel, hyperposterior, fitted_mixture, jitter=model.jitter,
	)
	return dataset_masked, mixture_masked

dataset_both_masked, mixture_both_masked = mask_and_update([0, 1])
resp_both_all = np.asarray(mixture_both_masked.responsibilities)
t_demo = int(np.argmin(resp_both_all.max(axis=1)))
resp_both = resp_both_all[t_demo]
k_true = int(fitted_mixture.assignments[t_demo])
dataset_y_masked, mixture_y_masked = mask_and_update([1], task_ids=[t_demo])
resp_y = np.asarray(mixture_y_masked.responsibilities[t_demo])

# %% [markdown]
"""
Side by side, the responsibilities over the `K` clusters -- the true cluster (from the unmasked fit)
highlighted:
"""

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, resp, title in [
	(axes[0], resp_both, "X and Y both half-masked"),
	(axes[1], resp_y, "Y only half-masked"),
]:
	colors = [cmap(k_true) if k == k_true else "0.7" for k in range(K)]
	ax.bar(range(K), resp, color=colors)
	ax.set_xlabel("cluster")
	ax.set_title(title)
axes[0].set_ylabel("responsibility")
fig.suptitle(f"Car {t_demo}'s cluster responsibilities (true cluster = {k_true})")
plt.show()



# %% [markdown]
"""
With both channels half gone, responsibility spreads across two clusters and induce a genuine doubt about
which route this car is on. With X still intact, responsibility concentrates back onto the true
cluster: one fully observed channel is enough to anchor the mixture even with the other half gone.
"""

# %%
def plot_masked_prediction(dataset_masked, mixture_masked, c_id, key, title):
	k_id = int(mixture_masked.assignments[t_demo])
	prediction = model.predict(dataset_masked, grid, mixture_masked, fitted_params)[t_demo, k_id, c_id]
	sample_keys = jr.split(key, 20)
	samples = jax.vmap(lambda k: sample_gp(k, prediction))(sample_keys)
	fig, ax = plot_single_task_prediction(
		dataset_masked, grid, dims, hyperposterior, mixture_masked, t_demo, c_id,
		prediction=prediction, samples=samples, ci_alpha=0, figsize=(8, 4),
	)
	fig.suptitle(f"{title} (cluster {k_id})")
	plt.show()
	truth = (x if c_id == 0 else y)[t_demo, cutoff:]
	rmse = float(np.sqrt(np.mean((np.asarray(prediction.mean)[cutoff:] - truth) ** 2)))
	print(f"{title}: RMSE = {rmse:.4f}")
	return np.asarray(samples)

key, sk_x_both, sk_y_both, sk_y_only = jr.split(key, 4)
samples_x_both = plot_masked_prediction(dataset_both_masked, mixture_both_masked, 0, sk_x_both,
                                         f"Car {t_demo}, channel X -- X and Y both masked")
samples_y_both = plot_masked_prediction(dataset_both_masked, mixture_both_masked, 1, sk_y_both,
                                         f"Car {t_demo}, channel Y -- X and Y both masked")

# %% [markdown]
"""
Forecasting capabilities greatly increase with knowledge of one channel.
"""

# %%
samples_y_only = plot_masked_prediction(dataset_y_masked, mixture_y_masked, 1, sk_y_only,
                                         f"Car {t_demo}, channel Y -- Y only masked")

# %% [markdown]
"""

## Back in (X, Y) space

Same story, read off the roundabout itself rather than two separate time series: the observed half of
the route (solid dots), the hidden half actually driven (crosses), and candidate hidden-half routes
built by pairing each channel's samples above point-by-point (not a joint sample -- X and Y are still
two independent GPs -- just a way to draw what each pairing would look like as a path).
"""

# %%
# X is fully observed in this scenario, so pair the known X with each Y sample instead of sampling X too.
samples_x_y_only = np.tile(x[t_demo][None, :], (samples_y_only.shape[0], 1))

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
for ax, samples_x, samples_y, title in [
	(axes[0], samples_x_both, samples_y_both, "X and Y both masked"),
	(axes[1], samples_x_y_only, samples_y_only, "Y only masked"),
]:
	ax.plot(x[t_demo], y[t_demo], color="0.75", linewidth=1, zorder=1, label="full true route")
	for sx, sy in zip(samples_x, samples_y):
		ax.plot(sx[cutoff:], sy[cutoff:], color="tab:blue", alpha=0.25, linewidth=1, zorder=2)
	ax.scatter(x[t_demo, :cutoff], y[t_demo, :cutoff], color="black", s=14, zorder=3, label="observed")
	ax.scatter(x[t_demo, cutoff:], y[t_demo, cutoff:], color="black", marker="x", s=20, zorder=4,
	           label="hidden truth")
	ax.set_aspect("equal")
	ax.set_xlabel("X (rescaled)")
	ax.set_title(title)
	ax.legend(fontsize=8)
axes[0].set_ylabel("Y (rescaled)")
fig.suptitle(f"Car {t_demo}, (X, Y) samples")
plt.show()

# %% [markdown]
r"""
## No dark magic here

Why treat X and Y as channels (`C=2`) at all, instead of correlated outputs (`O=2`, as
[the multi-output example](multi_output_example.ipynb) does)?

There is a cluster-like structure in this roundabout dataset, and clustering is exactly where
channels earn their keep: the mixture is the mechanism that lets X and Y each vote on the same
cluster assignment. An output's covariance shares *values* across dimensions; a channel's mixture
shares *membership* across dimensions instead. (Mimosa also supports combining both, cluster-based
correlated outputs.)

Cost is the second reason: channels are handled with `vmap`, so training scales *linearly* in `C`
(roughly $O(T \cdot C \cdot N^3)$); correlated outputs instead share one
$(O \cdot G) \times (O \cdot G)$ Cholesky factorization, so training scales *cubically*
in `O` (roughly $O(T \cdot O^3 \cdot N^3)$).

Setting cost aside entirely, multi-output is always at least as applicable as multi-channel: X(t)
and Y(t) *do* look like they could plausibly be correlated outputs and not two unrelated signals.
Multi-channel is really reserved for specific problems or computational trade-offs -- a way to
exploit one particular kind of link through the mixture, cheaply, without taking on multi-output's
full complexity.
"""
