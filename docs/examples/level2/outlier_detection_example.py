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
for name in ("soir_profiles.csv",):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))

# %% [markdown]
"""
# Outlier detection

A fitted model tells you more than where the clusters are: it tells you how *surprised* it is by
each task. Tasks it cannot explain are outliers, and scoring them costs nothing beyond the fit you
already ran.

The data here is real: 663 temperature profiles of Venus' atmosphere measured by the SOIR instrument
aboard Venus Express. Each profile is a task -- temperature against pressure -- and each point comes
with its own measurement error, which Mimosa can take as *known* noise. Some profiles are bad
retrievals, and that is what we are after.

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level2/outlier_detection_example.py` script!
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
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from kernax import ConstantMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dimensions, ModelConfig, Parameters, Mixture,
	BasicModel, KMeansGrid,
	load_csv, build_parameters, known_noise_kernel,
	plot_dataset, plot_clusters, plot_task,
)
from mimosa.linalg import cho_factor, cho_solve
from mimosa.nll import tasks_nlls

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi'] = 300
jax.devices()

# %% [markdown]
r"""
The CSV carries two things the other examples' files don't. `Noise1_1` is the measurement error of
each point, which `load_csv` reads into `dataset.known_output_noise`. `LST` is the local solar time
of the profile -- sunrise (0) or sunset (1) -- repeated on each of its rows; `load_csv` ignores any
column that isn't `Input*`, `Output*` or `Noise*`, so we read that one ourselves.

The input is a standardised $-\log_{10}$ of the pressure and the output a standardised temperature,
so the axes below are in standard deviations rather than in millibars and kelvins. Pressure falls as
you climb, so the input axis runs from the deep atmosphere on the left to the top of it on the right.
"""

# %%
dataset = load_csv("data/soir_profiles.csv")

# `load_csv` numbers its tasks by ascending TaskID, so a per-task column read the same way lines up
# with the dataset row for row.
lst = pl.read_csv("data/soir_profiles.csv").group_by("TaskID").agg(pl.col("LST").first()).sort("TaskID")
lst = jnp.asarray(lst["LST"].to_numpy())

print(f"{dataset.outputs.shape[0]} profiles, up to {dataset.outputs.shape[1]} points each")
print(f"known noise: {dataset.known_output_noise is not None}")
print(f"sunrise / sunset: {int((lst == 0).sum())} / {int((lst == 1).sum())}")

# %% [markdown]
"""
Profiles are measured at their own pressure levels, so tasks do not share input locations
(`isotopic_tasks=False`) and a `UnionGrid` would hold one point per distinct level -- **tens of
thousands of them**, far past what a dense grid covariance can hold. `KMeansGrid` fixes the budget
instead: 256 points placed where the data actually is.

Every profile was measured at either sunrise or sunset, giving us a natural `K=2`. The mixture is therefore *given*,
not fitted: *here the clusters are a fact about the measurement, not something to infer from it*.
"""

# %%
key, grid_key = jr.split(key)
grid = KMeansGrid(grid_key, dataset.inputs, n_points=256)

T, N, C = dataset.outputs.shape
I = dataset.inputs.shape[-1]
dims = Dimensions(T=T, K=2, I=I, C=C, O=1, N=N, G=len(grid.points))

model_config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)
print(f"Dimensions: {dims}")

# %%
fig, ax = plot_dataset(dataset, dims, figsize=(8, 6), alpha=.1)
fig.suptitle(f"{dims.T} SOIR temperature profiles")
plt.show()

# %% [markdown]
"""
## Fitting the model

The usual four parameters, with one addition: `known_noise_kernel` turns the per-point measurement
errors into a non-trainable noise kernel, added to the trainable one. Each observation then enters
the task covariance with its own variance, instead of every point sharing a single fitted noise
level.
"""

# %%
init_params = Parameters(
	cluster_mean=ConstantMean(0.),
	cluster_kernel=VarianceKernel(1.) * SEKernel(length_scale=.5),
	task_kernel=VarianceKernel(.25) * SEKernel(length_scale=.5),
	noise_kernel=WhiteNoiseKernel(noise=.05),
)
init_params = build_parameters(init_params, dims, model_config)

# The known errors are variances, one per point, and are not optimised.
init_params = Parameters(
	cluster_mean=init_params.cluster_mean,
	cluster_kernel=init_params.cluster_kernel,
	task_kernel=init_params.task_kernel,
	noise_kernel=init_params.noise_kernel + known_noise_kernel(dataset.known_output_noise, dims, model_config),
)

# %% [markdown]
"""
A `Mixture` is just a matrix of responsibilities, one row per task. Knowing the answer, ours are
hard: a one-hot row per profile, all the weight on its own side of the terminator.

`init_mixture` seeds the fit with it and `freeze_mixture` keeps it there, so the EM loop stops
updating responsibilities and only optimises hyperparameters. The clusters then mean exactly what we
say they mean, which is what makes the scores below comparable across profiles.
"""

# %%
known_mixture = Mixture(responsibilities=jnp.eye(dims.K)[lst])

# `prng_key` still seeds the k-means mixture initialiser -- which `init_mixture` means we never reach.
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=dims.K)

hyperposterior, mixture, params = model.fit(
	dataset, grid, init_params, init_mixture=known_mixture, freeze_mixture=True, n_iter=10
)

# %%
fig, ax = plot_dataset(dataset, dims, mixture=mixture, figsize=(8, 6), alpha=.08)
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior, figsize=(8, 6), fig=fig, ax=ax)
fig.suptitle("Fitted mean-processes, profiles colored by local solar time")
plt.show()

# %% [markdown]
"""
The two mean-processes come out close together, but difference is still real: the curves sit about 1.9
posterior standard deviations apart on average. It is just small next to the variance of the task themselves.
"""

# %% [markdown]
"""
## Scoring the profiles

Two scores, both comparing a profile to the cluster it was assigned to.

The first is the one the model already optimises: the **negative log-likelihood** of a task under a
mean-process, which `mimosa.nll.tasks_nlls` returns for every (task, cluster) pair. It needs each
task's covariance, which is the task kernel plus the noise kernel evaluated on the task's own
points -- exactly what the fit computes internally.
"""

# %%
task_covs = (params.task_kernel + params.noise_kernel)(dataset.clean_inputs, output_ids=dataset.output_ids)
nlls = tasks_nlls(dataset, grid, task_covs, hyperposterior)  # (T, K, C)

# Keep the score under the profile's own cluster -- here, its side of the terminator.
k_ids = mixture.assignments
nll_scores = nlls[jnp.arange(dims.T), k_ids, 0]

# Both scores below are sums over a profile's points, and these profiles hold anywhere between 2 and
# 85 of them -- so a long profile outscores a short one whatever its shape. Dividing by the number of
# observed points is what makes them comparable across profiles.
n_observed = jnp.sum(~jnp.isnan(dataset.outputs[:, :, 0]), axis=1)
nll_scores = nll_scores / n_observed

# %% [markdown]
r"""
The second is the **KL divergence** between what we measured and what the cluster predicts there.
The measurement is a Gaussian centred on the observed temperatures with the known errors on its
diagonal; the cluster's prediction is the mean-process restricted to the profile's own pressures,
widened by the task kernel. Where the NLL asks "how likely is this profile?", the KL asks "how far
apart are these two distributions?", and so it reacts to a profile whose *error bars* are wrong as
well as one whose values are.

$$\mathrm{KL}(\mathcal{N}_0 \| \mathcal{N}_1) = \tfrac{1}{2}\left(\mathrm{tr}(\Sigma_1^{-1}\Sigma_0)
+ (\mu_1-\mu_0)^\top \Sigma_1^{-1} (\mu_1-\mu_0) - d + \ln\frac{|\Sigma_1|}{|\Sigma_0|}\right)$$
"""

# %%
# Profiles hold between 2 and 85 real points, the rest being NaN padding. Masked rows and columns
# are replaced by the identity and the mean difference zeroed, so a padded point contributes
# +1 to the trace and -1 to the dimension count, cancelling out -- the same masking mimosa's own
# likelihoods use.
latent_covs = params.task_kernel(dataset.clean_inputs, output_ids=dataset.output_ids)


def task_kl(outputs, noise, mapping, latent_cov, k_id):
	post = hyperposterior.marginal(mapping)  # mean (K, C, N), covariance (K, C, N, N)

	observed, variances = outputs[:, 0], noise[:, 0]
	mask = jnp.isnan(observed)
	eye = jnp.eye(len(observed))

	cov = jnp.where(mask[None, :] | mask[:, None], eye, post.covariance[k_id, 0] + latent_cov[0, 0])
	variances = jnp.where(mask, 1., variances)
	diff = jnp.where(mask, 0., post.mean[k_id, 0] - observed)

	cov_l = cho_factor(cov)
	trace = jnp.sum(jnp.diagonal(cho_solve(cov_l, eye)) * variances)
	quadratic = jnp.sum(diff * cho_solve(cov_l, diff[:, None])[:, 0])
	log_det = 2 * jnp.sum(jnp.log(jnp.diagonal(cov_l))) - jnp.sum(jnp.log(variances))

	return .5 * (trace + quadratic - len(observed) + log_det)


kl_scores = vmap(task_kl)(dataset.outputs, dataset.known_output_noise, grid.mappings, latent_covs, k_ids)
kl_scores = kl_scores / n_observed  # per point, as for the NLL

# %% [markdown]
"""
## Flagging

Both scores are unbounded and have no natural cut-off, so the threshold is a decision, not a
result. Here we flag the worst 2% on either score; a different budget, or a hard value read off the
plot below, is just as defensible.
"""

# %%
QUANTILE = .98
nll_threshold = jnp.quantile(nll_scores, QUANTILE)
kl_threshold = jnp.quantile(kl_scores, QUANTILE)

outliers = (nll_scores >= nll_threshold) | (kl_scores >= kl_threshold)
print(f"{int(outliers.sum())} profiles flagged out of {dims.T}")

# %%
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(np.asarray(kl_scores)[~outliers], np.asarray(nll_scores)[~outliers],
           s=6, alpha=.5, label="kept")
ax.scatter(np.asarray(kl_scores)[outliers], np.asarray(nll_scores)[outliers],
           s=18, color="tab:red", marker="D", label="flagged")
ax.axvline(float(kl_threshold), color="tab:red", lw=.6, ls="--")
ax.axhline(float(nll_threshold), color="tab:red", lw=.6, ls="--")
ax.set_xscale("log")
ax.set_xlabel("KL divergence")
ax.set_ylabel("negative log-likelihood")
ax.legend(fontsize=8)
fig.suptitle("Per-point KL divergence against per-point NLL")
plt.show()

# %% [markdown]
"""
The two scores rank the profiles similarly, yet at the threshold they flag largely *different* ones
-- which is the point of computing both.

```{note}
A per-point score is only as steady as the number of points behind it. A few of the profiles flagged
here hold fewer than five, where a single bad measurement moves the average on its own. Dropping the
shortest profiles before scoring is a reasonable precaution.
```

Let's look at what actually got flagged, against the mean-processes they were compared to.
"""

# %%
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior, figsize=(8, 6), legend=False)
for t_id in np.flatnonzero(np.asarray(outliers))[:12]:
	plot_task(dataset, dims, int(t_id), 0, fig=fig, ax=ax, color="tab:red", marker="x", alpha=.6)
fig.suptitle(f"12 of the {int(outliers.sum())} flagged profiles, over the fitted mean-processes")
plt.show()

# %% [markdown]
"""
```{note}
Fixing the mixture buys more than fidelity to the instrument: it makes a bad score unambiguous. Had
we let the model cluster, a profile could score badly either because nothing explains it or because
it simply landed in the wrong cluster, and the two are hard to tell apart. Here the cluster is a
fact about the measurement, so a bad score can only mean the profile disagrees with its own side.
```

From here, the obvious next move is to drop the flagged profiles and refit: pass a `Mixture` whose
responsibilities are zeroed on them, and `model.fit` will ignore them while still fitting everything
else. Whether that is an improvement or a way to talk yourself into a cleaner-looking answer depends
on knowing why those profiles are bad, which is a question about the instrument, not about the model.
"""
