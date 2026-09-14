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
for name in ("turnip_train.csv", "turnip_test.csv"):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))
# %% [md]
"""
# Predicting turnip prices in Animal Crossing

This example explores two facets of Mimosa:
  - training a typical MagmaClust model on a simple dataset
  - performing model selection based on prediction metrics specific to this use case

Written using jupytext's py:percent format. This script can be run cell-by-cell or as a usual Python script.
"""

# %% [md]
"""
## Getting started
"""

# %% [md]
"""
First, the usual imports, configs and helper functions:
"""

# %% jupyter={"source_hidden": true}
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)
import jax.random as jr
import jax.numpy as jnp
import numpy as np
from jax import vmap
import matplotlib.pyplot as plt

from kernax import BatchModule, ConstantMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
    Dataset, Dimensions, Grid, Mixture, ModelConfig, Parameters,
    BasicModel, UnionGrid, load_csv, build_parameters, sample_gp,
    plot_dataset, plot_clusters, plot_single_task_prediction,
)

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi'] = 300

SLOTS = ["buy_price", "mon_am", "mon_pm", "tue_am", "tue_pm", "wed_am", "wed_pm",
         "thu_am", "thu_pm", "fri_am", "fri_pm", "sat_am", "sat_pm"]

# Daisy Mae comes on Sunday morning and mon_am is two half-days later, so the buy price sits at
# input 0 with the week's slots at 2..13. Input 1 (sun_pm) is never observed.
SLOT_INPUTS = np.array([0] + list(range(2, 14)))


def label_slots(fig):
	"""Replace the numeric input ticks with the half-day slot names."""
	for axes in fig.axes:
		axes.set_xticks(SLOT_INPUTS)
		axes.set_xticklabels(SLOTS, rotation=45, ha="right")
		axes.set_xlabel("slot")
		axes.set_ylabel("price (bells)")


def to_wide(dataset):
	"""Dense `(T, 13)` array of prices in slot order, NaN where a week has no observation."""
	inputs = jnp.broadcast_to(dataset.inputs, dataset.outputs.shape[:2] + dataset.inputs.shape[-1:])
	x, y = np.asarray(inputs)[..., 0], np.asarray(dataset.outputs)[..., 0]
	observed = ~np.isnan(x) & ~np.isnan(y)
	wide = np.full(y.shape[:1] + SLOT_INPUTS.shape, np.nan)
	rows = np.broadcast_to(np.arange(len(y))[:, None], x.shape)
	wide[rows[observed], np.searchsorted(SLOT_INPUTS, x[observed])] = y[observed]
	return wide


def draw_weeks(ax, wide, **kwargs):
	"""Join each week's observed prices, so a week reads as a curve rather than a cloud."""
	for prices in wide:
		ax[0, 0].plot(SLOT_INPUTS, prices, **kwargs)

# %% [md]
"""
Then, dataset loading:
"""

# %%
train_data = load_csv("data/turnip_train.csv")
test_data = load_csv("data/turnip_test.csv")

train_wide, test_wide = to_wide(train_data), to_wide(test_data)

# %% [md]
"""
Throughout the framework, Mimosa keeps track of the dimensions of the datasets and the shape of the model parameters
via two objects: `Dimensions` and `ModelConfig`.

We will also set `K`, the number of clusters, to 6 here. We'll see in the second part of this notebook how we can find 
the "best" number of clusters empirically. The model config is also something worth validating empirically, here we 
just set it to a sensible guess.

In Animal Crossing, prices vary twice a day from Monday to Saturday. If we add the buy price as "day 0", that makes for
13 price points in total, giving us the size of the grid!
"""

# %%
T_train, N, C = train_data.outputs.shape
I = train_data.inputs.shape[-1]

dims = Dimensions(T=T_train, K=6, I=I, C=C, O=1, N=N, G=13)
print(f"Dimensions: {dims}")

# Tasks observe different slots, so they do not share input locations (isotopic_tasks=False).
# We suspect that different clusters behave differently, so we set shared_cluster_hps to False and
# cluster_specific_task_hps to True.
model_config = ModelConfig(
	shared_cluster_hps=False,
	cluster_specific_task_hps=True,
	isotopic_tasks=False,
)
print(f"Model config: {model_config}")

# %% [md]
"""
Let's explore our dataset visually! First, a broad view of the whole train dataset, then a task-specific plot.

Try and change the ID of the plotted task to get a feeling about the data!
"""

# %%
fig, ax = plot_dataset(train_data, dims, figsize=(10, 6), alpha=.15)
draw_weeks(ax, train_wide, color="C0", alpha=.03, linewidth=.7)
fig.suptitle(f"{dims.T} turnip weeks (prices in bells)")
label_slots(fig)
plt.show()

# %%
TASK_ID = 0
fig, ax = plot_dataset(train_data, dims, figsize=(10, 6), alpha=.15, t_id=TASK_ID)
draw_weeks(ax, train_wide[[TASK_ID]], color="C0", alpha=.5, linewidth=1.)
fig.suptitle(f"Task {TASK_ID} (prices in bells)")
label_slots(fig)
plt.show()

# %% [md]
"""
Seeing the whole dataset all at once, it's hard to believe there might be structure to it. The only identifiable trend
is that some weeks follow a very linear decreasing pattern. Let's see if Mimosa can extract more information from this 
dataset.
"""

# %% [md]
"""
## Training the model

We are almost set! But before we fit the model, we have to specify the *structure* of the parameters.

We have 4 parameters to specify:
* cluster_mean: the mean function of our mean-processes, specifying the long-term trend they are centered on
* cluster_kernel: the kernel function of our mean-processes, specifying their "shape" (highly variable, smooth, etc.)
* task_kernel: the kernel function of tasks, specifying the shape of their variation around mean-processes
* noise_kernel: the noise of the points we observed/processes we are learning

The initial values of the hyper-parameters inside the structure do not matter too much, as they will be optimised.
"""

# %%
# Starting guess, in the data's own units: prices sit around ~100 bells and inputs are half-days
# spanning 0..12, hence a mean near 100, variances in bells^2 and length scales of a few slots.
init_params = Parameters(
	cluster_mean=ConstantMean(100.),
	cluster_kernel=VarianceKernel(900.) * SEKernel(length_scale=3.),
	task_kernel=VarianceKernel(400.) * SEKernel(length_scale=2.),
	noise_kernel=WhiteNoiseKernel(noise=25.),
)

# We have to "build" the parameters so that they match the model config and dimensions
# E.g: we wanted cluster-specific HPs but specified only one value. `build_parameters` ensures there will be K distinct
# values to optimise
init_params = build_parameters(init_params, dims, model_config)

# %% [md]
"""
Finally, one last detail: the training grid, which we just set to the union of input points, so 0..12. The mixture
proportions need no setting -- they are read off the responsibilities the model learns (`Mixture.proportions`).
"""

# %%
grid = UnionGrid()(train_data.inputs)

# %% [md]
"""
Here we are! Let's instantiate the model and train it!
"""

# %%
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=dims.K)

hyperposterior, fitted_mixture, fitted_params = model.fit(train_data, grid, init_params, n_iter=100)

print("cluster sizes:", jnp.bincount(fitted_mixture.assignments, length=dims.K))

# %% [md]
"""
With the optimised parameters and mixture coefficients, we can now see what the model has learnt:
* Where the clusters are, and how many tasks each of them holds
* What would be a prediction for a specific task, based on its most probable cluster
* What are the hyper-parameter values learnt during training, and what they tell us about the data
* ...
"""

# %%
# Plot the fitted clusters (mean-processes) over the data
fig, ax = plot_dataset(train_data, dims, mixture=fitted_mixture, figsize=(10, 6), alpha=.1)
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior, figsize=(10, 6), fig=fig, ax=ax)
fig.suptitle("Fitted price patterns (mean-processes), weeks colored by cluster")
label_slots(fig)
plt.show()

# %% [md]
"""
Try changing `t_id` to see a different week and `k_id` to see what the prediction would look like if we assumed
the task was from that specific cluster!

As `train_data` contains mainly weeks that have few missing points, you might see that cluster assignment often doesn't
make a big difference in prediction. We'll see later that it is much more important for predicting zones that are far
away from observed data.
"""

# %%
# Predict one week
predictions = model.predict(train_data, grid, fitted_mixture, fitted_params)  # (T, K, C, O*G)

t_id = 4
k_id = int(fitted_mixture.assignments[t_id])  # week's dominant cluster
prediction = predictions[t_id, k_id, 0]

fig, ax = plot_single_task_prediction(
	train_data, grid, dims, hyperposterior, fitted_mixture, t_id, 0, prediction=prediction, figsize=(10, 6)
)
fig.suptitle(f"Prediction — week {t_id}")
label_slots(fig)
plt.show()

# %%
# Draw samples from that prediction
key, sample_key = jr.split(key)
n_samples = 64
sample_keys = jr.split(sample_key, n_samples)
samples = vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(sample_keys)  # (S, O*G)

fig, ax = plot_single_task_prediction(
	train_data, grid, dims, hyperposterior, fitted_mixture, t_id, 0, samples=samples, figsize=(10, 6)
)
fig.suptitle(f"Prediction samples — week {t_id}")
label_slots(fig)
plt.show()

# %% [md]
"""
## Model selection

Now, how do we know how many clusters we should train for? How do we know which hyper-parameters should be shared or 
distinct? How do we know if a Squared-Exponential kernel was the best choice for tasks/clusters?

Model selection is a field of research of its own. Here, we are going to demonstrate one way of doing it: defining a
use case that we are going to apply on data unseen by the model, with specific metrics. The rationale is that we want
the best model for that specific use-case.

What is the "best" model? With Gaussian processes, there are two things we must account for: **precision** and 
**uncertainty calibration**. With a clustering algorithm, we must also account for mixture quality. The metrics we are 
going to use will therefore focus on these aspects.

Here, our use case is simple: we would like to be able to *predict the future trend of prices **early** in the week*.
Our validation methodology is then clear: provide the trained model with the prices at the start of the week and compare
its predictions to the known prices on the rest of the week.
"""

# %% [md]
"""
No test observation can reach a fit, so we reuse what the fit produced: `hyperposterior` holds
the mean-processes learnt on the training weeks, `mixture_updater` assigns a test week to them from
its early points alone, and `predictor` predicts the whole week from those same points.
"""

# %%
GIVEN_TEST_POINTS = 8  # Including buy price

# Hide everything past the given points, the way missing data is padded.
test_inputs = jnp.broadcast_to(test_data.inputs, test_data.outputs.shape[:2] + test_data.inputs.shape[-1:])
seen = jnp.isin(test_inputs[..., 0], jnp.asarray(SLOT_INPUTS[:GIVEN_TEST_POINTS]))
context = Dataset(inputs=jnp.where(seen[..., None], test_inputs, jnp.nan),
                  outputs=jnp.where(seen[..., None], test_data.outputs, jnp.nan))


prior_mixture = Mixture(responsibilities=jnp.full((dims.T, dims.K), 1 / dims.K))


context_grid = Grid(points=grid.points,
                    mappings=UnionGrid().compute_mappings(grid.points, context.inputs))
test_mixture = model.mixture_updater(
	context, context_grid, fitted_params.task_kernel + fitted_params.noise_kernel,
	hyperposterior, prior_mixture, jitter=model.jitter)

test_predictions = model.predictor(context, context_grid, hyperposterior, fitted_params, jitter=model.jitter)

# %%
t_id = 4  # try 900: a week whose early prices leave the cluster ambiguous
k_id = int(test_mixture.assignments[t_id])  # cluster inferred from the early prices alone

fig, ax = plot_single_task_prediction(
	context, grid, dims, hyperposterior, test_mixture, t_id, 0,
	prediction=test_predictions[t_id, k_id, 0], figsize=(10, 6)
)
ax[0, 0].scatter(SLOT_INPUTS[GIVEN_TEST_POINTS:], test_wide[t_id, GIVEN_TEST_POINTS:],
                 color="crimson", marker="x", label="hidden prices")
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"Test week {t_id} — predicted from its {GIVEN_TEST_POINTS} first prices")
label_slots(fig)
plt.show()

# %%
N_SAMPLES = 128
key, cluster_key, sample_key = jr.split(key, 3)

# One cluster per sample slot, drawn from that week's mixture, then one curve from each: the cloud
# follows the mixture rather than a single cluster.
clusters = jr.categorical(cluster_key, jnp.log(test_mixture.responsibilities), shape=(N_SAMPLES, dims.T)).T  # (T, S)
weeks = jnp.arange(dims.T)[:, None]
test_samples = sample_gp(sample_key,
                         test_predictions.mean[:, :, 0][weeks, clusters],        # (T, S, G)
                         test_predictions.covariance[:, :, 0][weeks, clusters])  # (T, S, G, G)

# %%
fig, ax = plot_single_task_prediction(
	context, grid, dims, hyperposterior, test_mixture, t_id, 0, samples=test_samples[t_id], figsize=(10, 6)
)
ax[0, 0].scatter(SLOT_INPUTS[GIVEN_TEST_POINTS:], test_wide[t_id, GIVEN_TEST_POINTS:],
                 color="crimson", marker="x", label="hidden prices")
ax[0, 0].legend(fontsize=7)
fig.suptitle(f"Test week {t_id} — {N_SAMPLES} samples drawn across clusters")
label_slots(fig)
plt.show()

# %% [md]
"""
We are going to use the following metrics:

* average RMSE to evaluate prediction accuracy
* CIC95 (aka the proportion of points sitting inside the 95% interval) for a quick glance at uncertainty quantification
* mixture agreement between the one on test points only and the one computed with all points
* z-score distribution, for a better look at uncertainty quantification and potential biases

We must account for the fact that *our prediction is multimodal*! The model predicts one GP distribution for every
cluster. If the mixture is not 100% certain about cluster assignment, our metrics must account for the different possible
clusters. This is why metrics are either computed on samples (for which proportions are following cluster probabilities) 
or specific to the majority cluster when there is no sound alternative. 
"""
# %%
# The mixture we would get if the whole week were known: what the early-week mixture is aiming at.
full_grid = Grid(points=grid.points, mappings=UnionGrid().compute_mappings(grid.points, test_inputs))
full_mixture = model.mixture_updater(
	Dataset(inputs=test_inputs, outputs=test_data.outputs), full_grid,
	fitted_params.task_kernel + fitted_params.noise_kernel, hyperposterior, prior_mixture, jitter=model.jitter)

# Grid points and slots coincide, so the hidden end of the week is the tail of both arrays.
truth = test_wide[:, GIVEN_TEST_POINTS:]                             # (T, hidden)
hidden_samples = np.asarray(test_samples)[:, :, GIVEN_TEST_POINTS:]  # (T, S, hidden)
mean, std = hidden_samples.mean(axis=1), hidden_samples.std(axis=1)
lo, hi = np.percentile(hidden_samples, [2.5, 97.5], axis=1)

# Baseline to read the RMSE against: the per-slot average of the training weeks, same for every week.
baseline = np.nanmean(train_wide, axis=0)[GIVEN_TEST_POINTS:]

z_scores = (truth - mean) / std
metrics = {
	"RMSE (bells)": float(np.sqrt(np.mean((mean - truth) ** 2))),
	"baseline RMSE (bells)": float(np.sqrt(np.mean((baseline - truth) ** 2))),
	"CIC95": float(np.mean((truth >= lo) & (truth <= hi))),
	"mixture agreement": float(np.mean(test_mixture.assignments == full_mixture.assignments)),
}

# %%
print(f"Predicting {truth.shape[1]} slots ahead from the {GIVEN_TEST_POINTS} first prices of {dims.T} test weeks:")
for name, value in metrics.items():
	print(f"  {name:<22} {value:6.3f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

# Calibrated uncertainty means z-scores follow a standard normal; heavier tails mean over-confidence.
z_grid = np.linspace(-5, 5, 200)
ax[0].hist(z_scores.ravel(), bins=60, range=(-5, 5), density=True, alpha=.7)
ax[0].plot(z_grid, np.exp(-z_grid ** 2 / 2) / np.sqrt(2 * np.pi), color="black", linestyle="--", label="N(0, 1)")
ax[0].set_xlabel("z-score"), ax[0].set_ylabel("density"), ax[0].legend(fontsize=7)
ax[0].set_title(f"Z-scores of the hidden prices — CIC95 {metrics['CIC95']:.0%}")

# Where the early-week clustering disagrees with the whole-week one, cluster by cluster.
confusion = np.zeros((dims.K, dims.K))
np.add.at(confusion, (np.asarray(full_mixture.assignments), np.asarray(test_mixture.assignments)), 1)
im = ax[1].imshow(confusion / confusion.sum(), cmap="Blues")
ax[1].set_xlabel(f"cluster from the {GIVEN_TEST_POINTS} first prices"), ax[1].set_ylabel("cluster from the whole week")
ax[1].set_title(f"Mixture agreement {metrics['mixture agreement']:.0%}")
fig.colorbar(im, ax=ax[1])
plt.show()

# %% [md]
"""
## No dark magic here

The goal of this notebook is to give an honest look at the model's capabilities. This section is here to reveal the 
limitations that might not be obvious from the results.

First, it is obvious that if you want an actual turnip price predictor for Animal Crossing, this is far from the best
you can use. The code from the game has been reverse-engineered and specific models will have far better performance.
This is just a toy example to demonstrate clustering and model selection.

You can observe it on RMSE which stays high, but also on uncertainty, which is under-estimated: only 80% of the hidden 
prices fall inside the 95% interval, 35% inside the 50% interval. Our samples ignoring observation noise is part of it 
-- adding it back brings CIC95 to 87%. The rest is structural: in-game prices follow patterns with sharp kinks and 
spikes, which smooth SE kernels and a handful of mean-processes can only approximate.

Secondly, the number of test points given to the model at the start of the week was chosen carefully. Providing the 7 
first prices means half of the week has already passed. But reducing that number quickly worsens the mixture
prediction, and thus the predictions themselves. You can try it yourself: see how mixture agreement evolves as you vary 
`GIVEN_TEST_POINTS`. 

Lastly, the dimensions of the dataset make training really easy computationally. Tasks have very few points and are
completely aligned, making the grid very small too. This is the setup Mimosa thrives on: many small, aligned tasks.
On real datasets, training with so many tasks might take far more time and memory, and sometimes require tricks like
task points alignment or stochastic learning.

Nonetheless, we saw how Mimosa was able to extract meaningful patterns from data where it was not obvious, and make 
better-than-average mid-term predictions!
"""
