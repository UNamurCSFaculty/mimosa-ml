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
for name in ("train_swimmers.csv", "test_swimmers.csv"):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))

# %% [markdown]
"""
# Basic multi-task usage of Mimosa

This example walks through the full pipeline on a *real* dataset -- swimming performances -- loaded from
CSV: configure the dimensions, fit a `BasicModel`, then predict a task and sample from that prediction.

The setting is deliberately kept simple: every swimmer is a task and they all follow a single shared
pattern, so `K=1` (no clustering). There is also a single channel and a single output (`C=1`, `O=1`),
so the only structure Mimosa actually uses is the multi-task sharing.

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level2/basic_mt_example.py` script!
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

from kernax import ConstantMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dimensions, ModelConfig, Parameters,
	BasicModel, UnionGrid,
	load_csv, build_parameters, sample_gp,
	plot_dataset, plot_clusters, plot_single_task_prediction,
)
from mimosa.mixture import MixtureUpdater

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi']=300
jax.devices()

# %% [markdown]
"""
We start by loading the train and test sets. Both files share the same format: one row per observed
point, with a `TaskID`, an input coordinate `Input_1` and a single output `Output1_1`. Each swimmer
is a task, and each row is one of that swimmer's measurements along the pool length.
"""

# %% 1. Load the data
train_data = load_csv("data/train_swimmers.csv")
test_data = load_csv("data/test_swimmers.csv")

# %% [markdown]
"""
Mimosa keeps track of the dimensions of the datasets and the shape of the model parameters via two
objects: `Dimensions` and `ModelConfig`. Here we read most of them straight off the loaded data.

For a *simple multi-task* setting we only need to set:
* `K`, the number of clusters, to 1 -- no clustering, every task shares the same mean-process;
* `O`, the number of correlated outputs, to 1 -- no multi-output;
* `C`, the number of channels, to 1 -- no multi-channel.

`T`, `N` and `I` are read from the loaded dataset. `G` is the number of points on the prediction
grid, which we build later as the union of every swimmer's inputs.
"""

# %% 2. Configuration
# The fitting grid is the union of every swimmer's input points, and G is its size. Grid
# construction isn't jit-compatible, so it's done here by the caller, outside of fit/predict, rather
# than owned by the model -- see mimosa.grid.
fitted_grid = UnionGrid(train_data.inputs)
G = len(fitted_grid.points)

# Dimensions: T tasks, K clusters, I input dims, C channels, O correlated outputs, N points
# observed per task, G points in the full grid.
T, N, C = train_data.outputs.shape
I = train_data.inputs.shape[-1]

dims = Dimensions(T=T, K=1, I=I, C=C, O=1, N=N, G=G)

# ModelConfig controls which hyperparameters are shared (across tasks/clusters/channels/outputs) and
# whether tasks/outputs share input locations. Swimmers are measured on overlapping but not identical
# locations, so isotopic_tasks is False.
model_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)
print(f"Dimensions: {dims}")
print(f"Model config: {model_config}")

# %% [markdown]
"""
## Exploring the raw dataset

A first broad view of the whole training set, then a single swimmer. Try changing `t_id` below to
look at a different swimmer!
"""

# %% 3. Plot the raw dataset
fig, ax = plot_dataset(train_data, dims, figsize=(8 * dims.C, 6), alpha=.15)
fig.suptitle("Swimming data (all tasks)")
plt.show()

# %%
t_id = 0
fig, ax = plot_dataset(train_data, dims, figsize=(8 * dims.C, 6), alpha=.15, t_id=t_id, kind="line")
fig.suptitle(f"Swimmer {t_id}")
plt.show()

# %% [markdown]
"""
## Training the model

Let's build our usual 4 parameters!
The initial values do not matter too much -- they are optimised during fitting -- but their structure
does, which is what `build_parameters` takes care of. The values below are expressed in the data's
own units: inputs span ~10-20 and outputs sit around ~65.
"""

# %% 4. Instantiate the model
# n_clusters can differ from the true K above (the model doesn't know it); jitter is the numerical
# stabiliser added before Cholesky factorizations, only increase it if you hit factorization errors.
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=dims.K)

# Starting guess for the parameters to fit. In practice these would be a rough, uninformed guess.
init_params = Parameters(
		cluster_mean=ConstantMean(65.),
		cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=2.),
		task_kernel=VarianceKernel(2.0) * SEKernel(length_scale=1.),
		noise_kernel=WhiteNoiseKernel(noise=1.))

# build_parameters batches the base kernels/mean below to match model_config's sharing structure
# (same helper generate_data uses internally), so their shapes line up with what model.fit expects.
init_params = build_parameters(init_params, dims, model_config)

# %% 5. Fit
# The grid was already built and its size stored in `dims.G` (see section 2).
hyperposterior, fitted_mixture, fitted_params = model.fit(train_data, fitted_grid, init_params, n_iter=10)

# %% 6. Plot the fitted cluster (mean-process)
fig, ax = plot_dataset(train_data, dims, mixture=fitted_mixture, figsize=(8 * dims.C, 6), alpha=.1)
fig, ax = plot_clusters(fitted_grid, dims, hyperposterior=hyperposterior, figsize=(8 * dims.C, 6), fig=fig, ax=ax)
fig.suptitle("Fitted mean-process on the dataset")
plt.show()

# %% [markdown]
"""
## Predicting

Predictions are multimodal: the model returns one Gaussian process per cluster for each task. With a
single cluster here there is only one, so the prediction for a swimmer is directly
`predictions[t_id, 0, 0]`.
"""

# %% 7. Predict
test_grid = UnionGrid(test_data.inputs)
test_mixture = MixtureUpdater()(test_data, test_grid, fitted_params.task_kernel, hyperposterior, fitted_mixture)
predictions = model.predict(test_data, test_grid, test_mixture, fitted_params)  # MultivariateNormal, batched (T, K, C, O*G)

c_id = 0
k_id = int(fitted_mixture.assignments[t_id])  # task's cluster (here always 0, since K=1)
prediction = predictions[t_id, k_id, c_id]

# %% 8. Plot the prediction: observed points, the mean-process, and the predictive mean + confidence interval
fig, ax = plot_single_task_prediction(
	test_data, fitted_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, prediction=prediction, figsize=(8 * dims.C, 6)
)
fig.suptitle(f"Prediction — swimmer {t_id}")
plt.show()

# %% 9. Draw samples from the prediction and plot them alongside it
key, sample_key = jr.split(key)
n_samples = 64
sample_keys = jr.split(sample_key, n_samples)
samples = vmap(lambda k: sample_gp(k, prediction))(sample_keys)  # (S, O*G)

fig, ax = plot_single_task_prediction(
	train_data, fitted_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, samples=samples, figsize=(8 * dims.C, 6)
)
fig.suptitle(f"Prediction samples — swimmer {t_id}")
plt.show()
