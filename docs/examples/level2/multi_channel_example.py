# %% tags=["remove-cell"]
import importlib.util, subprocess, sys
from pathlib import Path
if importlib.util.find_spec("mimosa") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mimosa-ml"], check=True)
# On Colab the notebook runs from /content, so fetch the dataset the example reads.
from urllib.request import urlretrieve
DATA_URL = "https://raw.githubusercontent.com/UNamurCSFaculty/mimosa-ml/main/docs/examples/data"
# The docs build runs each notebook from its own level folder, while the data folder is shared at
# docs/examples/data. On Colab the notebook runs from /content, where neither exists.
import os
if Path("../data").is_dir():
    os.chdir("..")
Path("data").mkdir(exist_ok=True)
if not Path("data", "car_trajectories.csv").exists():
    urlretrieve(f"{DATA_URL}/car_trajectories.csv", Path("data", "car_trajectories.csv"))

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
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt

from kernax import AffineMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dataset, Dimensions, ModelConfig, Parameters,
	BasicModel, UnionGrid, Mixture, FunctionPredictor, ObservationPredictor,
	load_csv, build_parameters, sample_gp,
	plot_dataset, plot_clusters, plot_single_task_prediction,
)
from mimosa.mixture import MixtureInitialiser

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi'] = 300

# %% [markdown]
"""
## The dataset

284 real car trajectories through a roundabout (the openDD dataset, cited above),
already aligned onto a common time grid: X and Y position (metres) over time, for each car. We'll
use "car" and "task" interchangeably from here on. The preprocessing has already been completed
for your convenience. You can load the prepared dataset using the following command.
"""

# %%
full_dataset = load_csv("data/car_trajectories.csv")
T_FULL, N_FULL, _ = full_dataset.outputs.shape

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(np.asarray(full_dataset.outputs[:, :, 0]).T, np.asarray(full_dataset.outputs[:, :, 1]).T,
        color="0.4", alpha=0.15, linewidth=0.8)
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
ax.set_title(f"{T_FULL} real car trajectories through a roundabout (OpenDD dataset) -- raw, unlabeled.")
plt.show()

# %% [markdown]
"""
*(The CSV itself already trades resolution for file size -- 64 points per car, quartered down from
OpenDD's native 256 -- and the fit below only ever sees a further-downsampled slice of 24 cars)*

## What's a channel?

X and Y above are two **channels**: independent scalar signals measured on every task, each with
its own Gaussian Process, never cross-covaried. An **output**, by contrast, is one of `O`
*correlated* dimensions of a single field, modeled jointly so their covariance captures how they
move together. Channels don't talk to each other through covariance at all -- they only interact
through the mixture: tasks are clustered once, from the sum of every channel's log-likelihood.

This example is only about `C`. For `O`, see [the multi-output example](multi_output_example.ipynb);
for the ordinary fit/predict pipeline this one builds on, see [the basic example](basic_example.ipynb).

## Configuring and fitting

This example works on a fixed random subsample: 24 cars, downsampled to 32 points. Stack `X` and `Y` along a
**new last axis** to get `(T, N, C=2)`:
"""

# %%
rng = np.random.RandomState(0)
T_SUB, K_SUB, GRID_STEP = 24, 4, 2

sub_idx = rng.choice(T_FULL, size=T_SUB, replace=False)
x_sub = full_dataset.outputs[sub_idx, ::GRID_STEP, 0] / 100.0
y_sub = full_dataset.outputs[sub_idx, ::GRID_STEP, 1] / 70.0
t_sub = full_dataset.inputs[0, ::GRID_STEP, 0]
N_SUB = t_sub.shape[0]

dataset = Dataset(inputs=t_sub[None, :, None], outputs=jnp.stack([x_sub, y_sub], axis=-1))
dims = Dimensions(T=T_SUB, K=K_SUB, I=1, C=2, O=1, N=N_SUB, G=N_SUB)

# %% [markdown]
"""
`Dimensions` tracks the scale of everything mimosa needs to know about this problem -- the names
are mostly self-explanatory, but `N`/`G` are worth spelling out:

* `T`: number of tasks (24 cars)
* `K`: number of clusters (4, chosen for this example, see "No dark magic here")
* `I`: dimensionality of the input (1 -- time)
* `C`: number of channels (2 -- X and Y)
* `O`: number of correlated outputs (1 -- unused here, see "What's a channel?" above)
* `N`: number of points *observed* per task (32, after downsampling)
* `G`: number of points in the *grid* predictions are made on

`N` and `G` coincide here because every car shares the same downsampled time grid and we predict
nowhere else -- they'd differ if you asked for predictions on a finer grid than what's observed (see
["Predicting on another grid"](basic_example.ipynb) in the basic example).

The model is configured the same way as any other mimosa model. Channel handling introduces one new
switch, `shared_channel_hps`, controlling whether channels share hyperparameters across the mixture
or stay fully independent -- here they don't, since X was rescaled by `/100`, Y by `/70`: same
spatial units, different scale, so letting each channel keep its own length-scale and variance is
the more defensible choice. `shared_cluster_hps=False` and `cluster_specific_task_hps=True` give
every cluster its own kernel hyperparameters instead of pooling them -- in principle: see
"No dark magic here" for why this particular fit doesn't actually exercise that freedom.
"""

# %%
config = ModelConfig(
	shared_task_hps=True, shared_cluster_hps=False,
	shared_channel_hps=False, cluster_specific_task_hps=True,
	isotopic_tasks=True,
)

params = build_parameters(Parameters(
	cluster_mean=AffineMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=0.5),
	task_kernel=VarianceKernel(0.01) * SEKernel(length_scale=0.5),
	noise_kernel=WhiteNoiseKernel(noise=0.01),
), dims, config)
grid = UnionGrid()(dataset.inputs)

# %% [markdown]
"""
One more thing before fitting: `BasicModel`'s default mixture initialiser is k-means over each
task's summary statistics. We are seeding the mixture
from a handful of visibly distinct cars because of the sensitivity of multi-channel clustering to
misattribution. We'll call these four cars *witnesses*
below; in code they're just `seed_ids`. `MixtureInitialiser` is mimosa's own extension point for
this: pick
one witness car per cluster you want to find, and assign every other car to whichever witness it
ends up closest to:
"""

# %%
class WitnessInitialiser(MixtureInitialiser):
	"""Seed the mixture from a handful of witness tasks instead of k-means."""
	seed_ids: jax.Array

	def __call__(self, dataset):
		# nanmean, not mean: a masked (NaN) point must not poison its task's whole feature.
		features = jnp.nanmean(dataset.outputs, axis=1)  # (T, C) -- each task's average position
		centers = features[self.seed_ids]                # (K, C)
		sq_dists = jnp.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
		return Mixture(responsibilities=jax.nn.softmax(-sq_dists, axis=1))

# The 4 witnesses themselves were picked offline (see "No dark magic here"), not learned here.
seed_ids = jnp.array([0, 16, 6, 23])

key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=K_SUB)
model = eqx.tree_at(lambda m: m.mixture_initialiser, model, WitnessInitialiser(seed_ids=seed_ids))
hyperposterior, fitted_mixture, fitted_params = model.fit(
	dataset, grid, params, n_iter=30, freeze_task_parameters=True,
)

# %% [markdown]
"""
X(t)/Y(t) view, one subplot per channel:
"""

# %%
fig, ax = plot_dataset(dataset, dims, mixture=fitted_mixture, alpha=0.4, figsize=(8 * dims.C, 4))
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior, fig=fig, ax=ax)
fig.suptitle(f"{T_SUB} cars, clustered jointly on (X, Y).")
plt.show()

# %% [markdown]
"""
## Pulling channels apart

Same fit, zoomed on its two most populated clusters, back in (X, Y) space:
"""

# %%
assignments = np.asarray(fitted_mixture.assignments)
cluster_sizes = np.bincount(assignments, minlength=K_SUB)
k1, k2 = np.argsort(cluster_sizes)[-2:]
cmap = plt.get_cmap("tab10")

fig, ax = plt.subplots(figsize=(6, 6))
for k in (k1, k2):
	members = assignments == k
	ax.plot(np.asarray(x_sub[members]).T, np.asarray(y_sub[members]).T, color=cmap(k), alpha=0.4, linewidth=1)
	ax.set_aspect("equal")
ax.set_xlabel("X (rescaled)"); ax.set_ylabel("Y (rescaled)")
ax.set_title(f"Clusters {k1} (n={cluster_sizes[k1]}) and {k2} (n={cluster_sizes[k2]}), isolated.")
plt.show()

# %% [markdown]
"""
Same cluster, channels pulled apart -- restricted to cluster `k1`'s cars:
"""

# %%
members_k1 = assignments == k1
dataset_k1 = Dataset(inputs=dataset.inputs, outputs=dataset.outputs[members_k1])
dims_k1 = Dimensions(T=int(members_k1.sum()), K=K_SUB, I=1, C=2, O=1, N=N_SUB, G=N_SUB)

# A one-hot Mixture so plot_dataset colors these cars with cluster k1's own color (tab10 index k1),
# matching the plot above -- there's nothing to cluster here, every car already is cluster k1.
onehot_k1 = jax.nn.one_hot(k1, K_SUB)
mixture_k1 = Mixture(responsibilities=jnp.tile(onehot_k1, (dims_k1.T, 1)))

fig, ax = plot_dataset(dataset_k1, dims_k1, mixture=mixture_k1, legend=False, alpha=0.4, figsize=(8 * 2, 4))
fig, ax = plot_clusters(grid, dims, k_id=int(k1), hyperposterior=hyperposterior, fig=fig, ax=ax, legend=False)
fig.suptitle(f"Cluster {k1}, one channel at a time -- each is its own independent GP, fit and read separately.")
plt.show()

# %% [markdown]
r"""
In cluster `k1`, X and Y positions are not two different *outputs* but two different *signals*
captured from a different point of view by the same instrument.

## Predicting

`model.predict` gives an actual posterior distribution for one task. Predictions come back batched
`(T, K, C, O*G)` -- indexing `[t_id, k_id, c_id]` slices out one task's prediction, under one
cluster, **for one channel**.

One choice worth making explicit: `BasicModel`'s default `predictor` is a `FunctionPredictor`,
which predicts the noise-free latent function $f(\cdot)$. If you'd rather predict an actual
observation, $y(\cdot) = f(\cdot) + \varepsilon$, swap in an `ObservationPredictor` instead.
"""

# %%
PREDICT_OBSERVATIONS = True  # False recovers mimosa's own default: the noise-free f(.)
predictor = ObservationPredictor() if PREDICT_OBSERVATIONS else FunctionPredictor()
model = eqx.tree_at(lambda m: m.predictor, model, predictor)

# %% [markdown]
"""
`Mixture.assignments` gives the hard cluster assignment (the argmax of `responsibilities`) of every
task, so task `t_id`'s own cluster is:
"""

# %%
t_id = 0
predictions = model.predict(dataset, grid, fitted_mixture, fitted_params)
k_id = int(fitted_mixture.assignments[t_id])

# %% [markdown]
"""
First, the predictive mean and 95% confidence interval alone once per channel:
"""

# %%
for c_id in range(dims.C):
	prediction = predictions[t_id, k_id, c_id]
	fig, ax = plot_single_task_prediction(
		dataset, grid, dims, hyperposterior, fitted_mixture, t_id, c_id,
		prediction=prediction, figsize=(8, 4),
	)
	plt.show()

# %% [markdown]
"""
Same distribution, now with 20 samples drawn from it:
"""

# %%
key, sample_key = jr.split(key)
for c_id in range(dims.C):
	prediction = predictions[t_id, k_id, c_id]
	sample_keys = jr.split(sample_key, 20)
	samples = jax.vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(sample_keys)
	fig, ax = plot_single_task_prediction(
		dataset, grid, dims, hyperposterior, fitted_mixture, t_id, c_id,
		prediction=prediction, samples=samples, ci_alpha=0, figsize=(8, 4),
	)
	plt.show()

# %% [markdown]
"""
## Putting the mixture link to the test

We simplify to `K=2` clusters for this test -- with only 4, small
clusters, a wrong assignment doesn't necessarily land on a mean-process that looks any different
from the right one. Two large, well-separated clusters is the case where
a wrong assignment should matter most:
"""

# %%
K_TEST = 2
config_test = ModelConfig(shared_task_hps=True, shared_cluster_hps=False, shared_channel_hps=False,
                           cluster_specific_task_hps=True, isotopic_tasks=True)
params_test = build_parameters(Parameters(
	cluster_mean=AffineMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=0.5),
	task_kernel=VarianceKernel(0.01) * SEKernel(length_scale=0.5),
	noise_kernel=WhiteNoiseKernel(noise=0.01),
), Dimensions(T=T_SUB, K=K_TEST, I=1, C=2, O=1, N=N_SUB, G=N_SUB), config_test)
seed_ids_test = jnp.array([0, 16])  # 2 witnesses, same offline search as before

key, test_key = jr.split(key)
model_test = BasicModel(prng_key=test_key, n_clusters=K_TEST)
model_test = eqx.tree_at(
	lambda m: m.mixture_initialiser, model_test, WitnessInitialiser(seed_ids=seed_ids_test),
)
_, mixture_test, _ = model_test.fit(dataset, grid, params_test, n_iter=30, freeze_task_parameters=True)
assignments_test = np.asarray(mixture_test.assignments)
print("cluster sizes at K=2:", np.bincount(assignments_test, minlength=K_TEST))

# %% [markdown]
"""
Now forecast: mask everything past 3/4 of the way through car 16's own X channel, Y and every
other car untouched. Then fit the ordinary multi-channel model on this masked dataset, and a
**channel-independent** model (X alone, `C=1`, its own mixture, blind to Y entirely) on the exact
same data.
"""

# %%
t_demo, cutoff = 16, 24
outputs_masked = np.asarray(dataset.outputs).copy()
outputs_masked[t_demo, cutoff:, 0] = np.nan
dataset_masked = Dataset(inputs=dataset.inputs, outputs=jnp.asarray(outputs_masked))

key, model_mc_key = jr.split(key)
model_mc = BasicModel(prng_key=model_mc_key, n_clusters=K_TEST)
model_mc = eqx.tree_at(lambda m: m.mixture_initialiser, model_mc, WitnessInitialiser(seed_ids=seed_ids_test))
hyperposterior_mc, mixture_mc, params_mc = model_mc.fit(
	dataset_masked, grid, params_test, n_iter=30, freeze_task_parameters=True,
)

dims_x = Dimensions(T=T_SUB, K=K_TEST, I=1, C=1, O=1, N=N_SUB, G=N_SUB)
config_x = ModelConfig(shared_task_hps=True, shared_cluster_hps=False, shared_channel_hps=True,
                        cluster_specific_task_hps=True, isotopic_tasks=True)
params_x = build_parameters(Parameters(
	cluster_mean=AffineMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=0.5),
	task_kernel=VarianceKernel(0.01) * SEKernel(length_scale=0.5),
	noise_kernel=WhiteNoiseKernel(noise=0.01),
), dims_x, config_x)
dataset_x = Dataset(inputs=dataset.inputs, outputs=jnp.asarray(outputs_masked[:, :, 0:1]))
grid_x = UnionGrid()(dataset_x.inputs)

key, model_x_key = jr.split(key)
model_x = BasicModel(prng_key=model_x_key, n_clusters=K_TEST)
model_x = eqx.tree_at(lambda m: m.mixture_initialiser, model_x, WitnessInitialiser(seed_ids=seed_ids_test))
hyperposterior_x, mixture_x, fitted_params_x = model_x.fit(
	dataset_x, grid_x, params_x, n_iter=30, freeze_task_parameters=True,
)

# %% [markdown]
"""
Did the two agree on `t_demo`'s cluster?
"""

# %%
k_mc = int(mixture_mc.assignments[t_demo])
k_x = int(mixture_x.assignments[t_demo])
print(f"unmasked fit put this car in cluster {int(assignments_test[t_demo])}")
print(f"multi-channel (masked): cluster {k_mc}")
print(f"X alone (masked):       cluster {k_x}")

# %% [markdown]
"""
Forecast the masked tail for both, and overlay the ground truth mimosa never saw:
"""

# %%
pred_mc = model_mc.predict(dataset_masked, grid, mixture_mc, params_mc)[t_demo, k_mc, 0]
pred_x = model_x.predict(dataset_x, grid_x, mixture_x, fitted_params_x)[t_demo, k_x, 0]
std_mc = np.asarray(jnp.sqrt(jnp.diag(pred_mc.covariance)))
std_x = np.asarray(jnp.sqrt(jnp.diag(pred_x.covariance)))

fig, ax = plt.subplots(figsize=(8, 4))
t_plot = np.asarray(t_sub)
ax.axvspan(t_plot[cutoff], t_plot[-1], color="0.9", zorder=0, label="forecast region")
ax.plot(t_plot, np.asarray(pred_mc.mean), color="tab:blue", label="multi-channel")
ax.fill_between(t_plot, np.asarray(pred_mc.mean) - 1.96 * std_mc, np.asarray(pred_mc.mean) + 1.96 * std_mc,
                 color="tab:blue", alpha=0.2)
ax.plot(t_plot, np.asarray(pred_x.mean), color="tab:orange", label="X alone")
ax.fill_between(t_plot, np.asarray(pred_x.mean) - 1.96 * std_x, np.asarray(pred_x.mean) + 1.96 * std_x,
                 color="tab:orange", alpha=0.2)
ax.scatter(t_plot[cutoff:], np.asarray(x_sub[t_demo, cutoff:]), color="black", marker="x", zorder=5,
           label="hidden truth")
ax.set_xlabel("time"); ax.set_ylabel("X (rescaled)"); ax.legend(fontsize=8)
ax.set_title(f"Task {t_demo}, channel X forecast beyond the shaded cutoff")
plt.show()

# %% [markdown]
"""
They don't agree: losing X's own evidence moves the channel-independent fit to the *other* cluster,
while the multi-channel fit stays anchored, by Y, to the same cluster the unmasked fit itself found.
The channel-independent forecast plateaus, tracking the wrong cluster's trend; the multi-channel
forecast keeps climbing with the hidden truth.

The mixture link helps in proportion to how distinct and well-populated the clusters actually are.
It's a real mechanism, not a free lunch that beats channel-independent fitting regardless of how
the data happens to be clustered.
"""

# %% [markdown]
r"""
## No dark magic here

Only 24 of the 284 cars are actually fit, at 32 points each -- refitting the full dataset jointly
is too slow to run on every documentation build. `K_SUB=4` is likewise a number picked for this
example, not a rediscovery of however many traffic patterns the roundabout actually has.

Computational cost can also push you toward channels instead of multi-output when the latter isn't
feasible: multi-channel doesn't claim to capture correlation between outputs, only to route,
through the mixture, information that two entirely separate GPs couldn't share. Multi-channel's
benefit rides entirely on the mixture: if the problem has no real cluster structure, there's
nothing to gain over an ordinary multi-task GP.

`BasicModel`'s default k-means mixture initialiser genuinely fails here, and more restarts of the
same procedure don't help -- multi-channel clustering is only as good as its initialisation, and
k-means starts too close to a bad one too often at this sample size. `WitnessInitialiser`'s witness
cars were themselves chosen by a farthest-point search over each subsample's (X, Y) mean position,
run once, offline, in plain numpy -- a real, reproducible algorithm, not eyeballing.

Lastly: why treat X and Y as channels (`C=2`) at all, instead of correlated outputs (`O=2`, as
[the multi-output example](multi_output_example.ipynb) does)?

There is a cluster-like structure in this roundabout dataset, and clustering is exactly where
channels earn their keep: the mixture is the mechanism that lets X and Y each vote on the same
cluster assignment. An output's covariance shares *values* across dimensions; a channel's mixture
shares *membership* across dimensions instead (Mimosa also supports combining both, cluster-based
correlated outputs, for problems that need both kinds of sharing at once.)

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
