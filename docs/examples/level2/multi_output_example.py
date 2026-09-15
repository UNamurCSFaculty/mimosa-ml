# %%
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
for name in ("electricity_tasks_MIMOSA.csv",):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))

# %% [markdown]
r"""
# Predicting Energy Consumption across Multiple Clients and Temporal Windows
This tutorial explores a real-world application that leverages two core components of the Mimosa framework:
*Multi-Output* and *Multi-Task* learning. The objective is to capture dependencies not only across the **input**
dimension but also across **multiple correlated** data sources. To illustrate the Multi-Output Multi-Task (MOMT)
framework, we analyze the *Electricity Hourly* dataset, sourced from the Monash Time Series Forecasting Archive
(available at: [https://zenodo.org/records/4656140](https://zenodo.org/records/4656140)).

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level2/multi_output_example.py` script!
"""

# %% [markdown]
r"""
## Getting started
First, let us initialize the environment by loading the necessary libraries, configurations, and helper functions:
"""

# %%
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", False)
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import equinox as eqx

from kernax import ConvolutionKernel, ZeroMean, VarianceKernel, SEKernel, PeriodicKernel, WhiteNoiseKernel, BlockMean, ConstantMean, ConvolutionKernel, BlockDiagKernel, ICMKernel, LCMKernel

from mimosa import (Dimensions, ModelConfig, GPModel, Parameters, BasicModel, FunctionPredictor, ObservationPredictor,
                    SubdomainRemover, load_csv, build_parameters, build_gp_parameters, MergedGrid)
from mimosa.linalg import find_exact_mappings
from mimosa.data_structures import Dataset, Grid, Hyperposterior, Mixture, MultivariateNormal, GPDataset, GPParameters
from mimosa.grid import MultiOutputUnionGrid, RegularGrid, MergedGrid, UnionGrid
from mimosa.linalg import find_exact_mappings
from mimosa.plot import plot_dataset, plot_clusters, plot_task, plot_single_task_prediction
from mimosa.sampling import sample_gp
from mimosa.synthetic import build_task_kernel
from mimosa.mixture import MixtureUpdater

key = jr.PRNGKey(42)
plt.rcParams["figure.dpi"] = 300
jax.devices()

# %% [markdown]
r"""
The *Electricity* dataset contains hourly power consumption records for 321 clients spanning from 2012 to 2014. 
In our framework, each client represents an **output**, *i.e.* a distinct target variable we aim to predict. 
The temporal dynamics of each client exhibit a recurrent 72-hour pattern, yielding a total of 356 sequential 
windows over the two-year period.

During preprocessing, the continuous signal for each output was segmented into these 356 discrete 72-hour windows. 
Consequently, each window is formulated as an individual **task**, interpreted as a specific realization of an 
underlying latent phenomenon. For this tutorial, we restrict our analysis to two specific outputs (clients 8 and 16). 
Feel free to download the full dataset and experiment with alternative pairs (e.g., clients 4 & 14, 6 & 15, or 11 & 17).

*Note: We advise a reduced number of outputs simultaneously (max 5-6), as the computational complexity of the 
inference scales cubically with the number of outputs. See "No dark magic here" section for more details.*

The preprocessing has already been completed for your convenience. You can load the prepared dataset using the 
following command:
"""

# %%
full_dataset = load_csv("data/electricity_tasks_MIMOSA.csv")

# %% [markdown]
r"""
As with any *Multi-Task* setting in Mimosa, we must specify the number of training tasks (`TRAIN_TASK_COUNT`) and 
the number of tasks reserved for prediction (`PRED_TASK_COUNT`). While you may adjust these parameters, utilizing 
50 training tasks provides a robust demonstration of the *Multi-Output Multi-Task* framework's capabilities while 
remaining time-training friendly.
"""

# %%
TRAIN_TASK_COUNT = 50
PRED_TASK_COUNT = 10

train_dataset = full_dataset[:TRAIN_TASK_COUNT]
pred_dataset = full_dataset[TRAIN_TASK_COUNT:TRAIN_TASK_COUNT+PRED_TASK_COUNT]

# %% [markdown]
r"""
In the *Electricity* dataset, tasks are evaluated on a shared temporal input grid, discretized from hour 0 to hour 72. 
Given that we are modeling 2 **outputs**, the concatenated grid comprises $72 \times 2 = 144$ points.

The `ModelConfig` object defines the parameter-sharing strategy across tasks, clusters, channels, and outputs, as well 
as whether tasks or outputs share identical input locations.
"""

# %%
T_train, N, C = train_dataset.outputs.shape
I = train_dataset.inputs.shape[-1]

train_dims = Dimensions(T=T_train, K=1, I=I, C=C, O=2, N=int(N/2), G=144)
model_config = ModelConfig(isotopic_tasks=False, isotopic_output_in_grid=True, isotopic_output_in_tasks=True,)

# %% [markdown]
r"""
Let us visually explore the dataset. Each **output** (client) is displayed in a separate subplot, while each distinct 
color corresponds to a specific **task** (a 72-hour temporal window).
"""

# %%
_fig, _ax = plot_dataset(train_dataset, train_dims, figsize=(8 * train_dims.C, 6), color_by_task=True)
plt.show()

# %% [markdown]
r"""
Although the **outputs** vary in magnitude and mean, a clear underlying correlation is observable between clients 
consumptions. Furthermore, within each **output** (client), a consistent temporal structure, shared across all 
**tasks**, is highly discernible. In the subsequent experiments, our objective is to exploit the correlations present 
across both **outputs** and **tasks** to generate accurate probabilistic predictions.
"""

# %% [markdown]
r"""
## Interpolation experiment: when *Multi-Output* fills the gap
Suppose client 8 experiences an electrical meter outage between hours 24 and 48 across every task (*i.e.* for both the 
`TRAIN_TASK_COUNT` and `PRED_TASK_COUNT` tasks).
"""

# %%
interp_bounds = ((24, 48),)
interp_output_id = 0
interp_remover = SubdomainRemover(bounds=interp_bounds)
interp_dataset, interp_held_out = interp_remover(train_dataset, o_id=interp_output_id)

_fig, _ax = plot_dataset(interp_dataset, train_dims, figsize=(8 * train_dims.C, 6), color_by_task= True)
plt.show()

# %% [markdown]
r"""
Can we reconstruct the missing consumption of client 8 by leveraging other sources of data? This is where 
*Multi-Output* modeling helps: we use the observed trajectory of client 16 on $[24\text{h}, 48\text{h}]$ to 
inform predictions for client 8 over the exact same window. Feel free to modify the lower and upper bounds 
of the missing window, as well as the output/client on which we mask data.

Before training, we must specify the structure of the parameters. Hyperparameter initialization values are not 
critical, as they will be optimized during training. In this example, we use a Linear Model of Coregionalization 
(LCM) kernel for both the mean process *and* the task process kernels. You can experiment with alternative structures 
such as an Intrinsic Coregionalization Model (ICM) or Convolution Process kernels by adjusting `base_init_params`.
"""

# %%
base_init_params = Parameters(
    cluster_mean=BlockMean(ConstantMean(), n_outputs=train_dims.O, output_hps_in_axes=0),
    cluster_kernel=LCMKernel(
        kernels=[
            VarianceKernel(10.0) * SEKernel(length_scale=2.0),
            VarianceKernel(5.0) * SEKernel(length_scale=0.5)
        ],
        coregionalization_matrices=[
            jnp.ones((train_dims.O, train_dims.O)),
            jnp.ones((train_dims.O, train_dims.O))
        ],
        kappas=[jnp.ones(train_dims.O), jnp.ones(train_dims.O)]
    ) + BlockDiagKernel(WhiteNoiseKernel(noise=20.), n_outputs=train_dims.O, output_hps_in_axes=0),
    task_kernel=LCMKernel(
        kernels=[VarianceKernel(0.5) * SEKernel(length_scale=1.0)],
        coregionalization_matrices=[jnp.ones((train_dims.O, train_dims.O))],
        kappas=[jnp.ones(train_dims.O)]
    ),
    noise_kernel=BlockDiagKernel(WhiteNoiseKernel(noise=5.), n_outputs=train_dims.O, output_hps_in_axes=0)
)

# We "build" the parameters so that they match the model config and dimensions
init_params = build_parameters(base_init_params, train_dims, model_config)

# %% [markdown]
r"""
Next, we have to define the training grid as the union of every task's input points. 
`MultiOutputUnionGrid` is `UnionGrid`'s multi-output counterpart: it needs `model_config` too, to determine whether 
outputs share grid/task input locations.
"""

# %%
train_grid = MultiOutputUnionGrid(train_dataset, model_config, n_outputs=2)
pred_grid = MultiOutputUnionGrid(pred_dataset, model_config, n_outputs=2)
full_grid = MergedGrid(train_grid, pred_grid)

# %% [markdown]
r"""
We can now instantiate and train the model! Feel free to increase `n_iter` to run more iterations of the 
EM algorithm. 
"""

# %%
key, train_key = jr.split(key)

model_interp = BasicModel(prng_key=train_key, n_clusters=train_dims.K, predictor=ObservationPredictor())  # Switch to `FunctionPredictor()` to predict the latent function instead of observations

interp_hyperposterior, interp_mixture, interp_params = model_interp.fit(
    train_dataset,
    full_grid,
    init_params,
    n_iter=25,
)

# %% [markdown]
r"""
Training is complete! We can now inspect what the model has inferred:

- The estimated mean process for each client;

- Task-specific posterior trajectories;

- Learned hyperparameter values and their structural interpretation.

Notice how the prediction of client 8 borrows information from the observations of client 16 on the window [24h, 48h]. The *Multi-Output* component of Mimosa allows outputs to share knowledge via the **covariance matrices** of the model (at both the cluster-mean and task-residual levels). The dashed blue line depicts the cluster hyperposterior mean, while the solid black line shows the task-specific posterior mean.
"""

# %%
# Prediction for task 0
interp_task_id = 0
interp_prediction = model_interp.predict(interp_dataset, train_grid, interp_mixture, interp_params)[interp_task_id, 0, 0]

# Draw samples from that prediction
n_samples = 64
_, _sample_key = jr.split(key)
sample_keys = jr.split(_sample_key, n_samples)
samples = vmap(lambda k: sample_gp(k, interp_prediction))(sample_keys)

_fig2, _ax2 = plot_dataset(train_dataset, train_dims, mixture=interp_mixture, c_id=0, figsize=(8, 10), color_by_task=True, alpha=0.1)
_fig2, _ax2 = plot_single_task_prediction(interp_dataset, train_grid, train_dims, interp_hyperposterior, interp_mixture, 
    interp_task_id, c_id=0, samples=samples, sample_alpha=0.1, fig=_fig2, ax=_ax2)
_fig2, _ax2 = plot_task(interp_held_out, train_dims, interp_task_id, c_id=0, fig=_fig2, ax=_ax2, color="red", marker="x")
_fig2.suptitle(f"Prediction task-specific (samples) for task {interp_task_id}")
plt.show()

# %% [markdown]
r"""
To measure the value of information transferred from client 16, we now fit a single-output *Multi-Task* GP using 
exactly the exact same tasks and missing window. Because this model only accesses observations from client 8, it 
serves as an ablation baseline **without** cross-output information. We define a classic SE kernel for both mean and 
task processes; feel free to choose another design (e.g linear, periodic, rational quadratic ...), even if the 
predictive distribution falling in the missing window will not be really impacted.
"""

# %%
# Keep only output 0 (client 8), using the same training tasks as the MOMT model.
single_output_dataset = Dataset(
    inputs=train_dataset.inputs,
    outputs=train_dataset.outputs[:, :train_dims.N, :],  # Only first N points -> output 0
    known_output_noise=None,
)
T, N, C = single_output_dataset.outputs.shape
single_output_dims = Dimensions(T=T, K=1, I=I, C=C, O=1, N=N, G=N)
single_output_interp_dataset, single_output_held_out = interp_remover(single_output_dataset)

single_output_init_params = Parameters(
    cluster_mean=ZeroMean(),
    cluster_kernel=VarianceKernel(10.0) * SEKernel(length_scale=1.0),
    task_kernel=VarianceKernel(0.5) * SEKernel(length_scale=1.0),
    noise_kernel=WhiteNoiseKernel(noise=0.05),
)
single_output_init_params = build_parameters(single_output_init_params, single_output_dims, model_config)

key, model_key = jr.split(key)
single_output_grid = UnionGrid(single_output_interp_dataset.inputs)
single_output_model = BasicModel(prng_key=model_key, n_clusters=single_output_dims.K, predictor=ObservationPredictor())


single_output_hyperposterior, single_output_mixture, single_output_params = single_output_model.fit(
    single_output_interp_dataset,
    single_output_grid,
    single_output_init_params,
    n_iter=25,
)

# %%
single_output_prediction = single_output_model.predict(single_output_interp_dataset, single_output_grid, single_output_mixture, single_output_params)[interp_task_id, 0, 0]

# Draw samples from that prediction
key, sample_key = jr.split(key)
sample_keys = jr.split(sample_key, 64)  # 64 samples
single_output_samples = vmap(lambda k: sample_gp(k, single_output_prediction))(sample_keys)

_fig, _ax = plot_dataset(single_output_interp_dataset, single_output_dims, c_id=0, figsize=(8, 10), color_by_task=True, alpha=0.1)
_fig, _ax = plot_single_task_prediction(single_output_interp_dataset, single_output_grid, single_output_dims, single_output_hyperposterior, single_output_mixture,
    interp_task_id, c_id=0, samples=single_output_samples, sample_alpha=0.1, fig=_fig, ax=_ax)
_fig, _ax = plot_task(single_output_held_out, single_output_dims, interp_task_id, c_id=0, fig=_fig, ax=_ax, color="red", marker="x")

_fig.suptitle(f"Prediction task-specific (samples) for task {interp_task_id}")
plt.show()

# %% [markdown]
r"""
Notice how the predictive mean quickly reverts to the prior mean, while the uncertainty explodes! When an interval 
lacks data across **all tasks** (training *and* prediction) for a given output, the standard *Multi-Task* framework 
behaves as expected, with a slow drift to the prior mean, with highly increasing variance.
"""

# %% [markdown]
r"""
## Forecasting experiment: when *Multi-Task* rescues the prediction
Suppose both clients experience an electrical meter outage during the final 24 hours of the last ten tasks/windows 
(i.e., for both `interp_output_id = 0` **and** `interp_output_id = 1`, but only across the `PRED_TASK_COUNT` prediction 
tasks). The prediction tasks are displayed in full opacity, whereas the training tasks appear with high transparency 
in the background.
"""

# %%
forecast_bound = ((48, 72),)
forecast_remover = SubdomainRemover(bounds=forecast_bound)
forecast_dataset, forecast_held_out = forecast_remover(pred_dataset)

# %%
plt.close("all")
plt.style.use("ggplot")
plt.rcParams.update({"axes.facecolor": "white", "axes.grid": False, "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "text.color": "black"})

_ggplot_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_fig, _ax = plot_task(train_dataset, train_dims, t_id=0, c_id=0, fig=None, ax=None, figsize=(10, 8), color=_ggplot_colors[0],
    alpha=0.05)

for _axis in _ax.flat:
    _axis.axvspan(*forecast_bound[0], color="gray", alpha=0.2, zorder=0)

for _t in range(1, train_dims.T):
    _c = _ggplot_colors[_t % len(_ggplot_colors)]
    _fig, _ax = plot_task(train_dataset, train_dims, t_id=_t, fig=_fig, ax=_ax, color=_c, alpha=0.05)

for _t in range(pred_dataset.inputs.shape[0]):
    _c = _ggplot_colors[_t % len(_ggplot_colors)]
    _fig, _ax = plot_task(forecast_dataset, train_dims, t_id=_t, fig=_fig, ax=_ax, color=_c, alpha=1.0)

for _row in range(train_dims.O):
    for _col in range(train_dims.C):
        _current_ax = _ax[_row, _col]
        _current_ax.set_xlabel("Hours")
        _current_ax.set_ylabel("Electricity consumption")
        _current_ax.set_title(f"Client {8 if _row == 0 else 16}")
plt.show()


# %% [markdown]
r"""
Can we reconstruct the missing consumption for the last ten tasks across both clients? This is where *Multi-Task* 
modeling helps: for both outputs, we leverage the observed trajectories from the training windows on [48h, 72h] to 
inform the prediction tasks over that exact same interval. Feel free to modify the lower and upper bounds of the 
missing window.

The parameter structure remains identical to the one used in the interpolation problem, so we do not need to redefine 
it. Let's simply train the model!
"""

# %%
model_forecast = BasicModel(prng_key=train_key, n_clusters=train_dims.K, predictor=ObservationPredictor())

forecast_hyperposterior, forecast_mixture, forecast_params = model_forecast.fit(
    train_dataset,
    train_grid,
    init_params,
    n_iter=25,
)

# %% [markdown]
r"""
Once the model is trained, we compute the predictive distributions for all held-out tasks. Feel free to modify `_forecast_task_id` to display the results for the prediction task of your choice.
"""

# %%
_forecast_task_id = 1

# The prediction tasks were never seen during the fit: they get their own grid over the very same
# points, and the fitted mean-processes assign them a cluster before we predict.
forecast_grid = MultiOutputUnionGrid(forecast_dataset, model_config, n_outputs=train_dims.O)
forecast_pred_mixture = MixtureUpdater()(forecast_dataset, forecast_grid, forecast_params.task_kernel, forecast_hyperposterior, forecast_mixture)
prediction = model_forecast.predict(forecast_dataset, forecast_grid, forecast_pred_mixture, forecast_params)[_forecast_task_id, 0, 0]  # MultivariateNormal, batched (T, K, C, O*G)

# Draw samples from that prediction
_n_samples = 64
_, _sample_key = jr.split(key)
_sample_keys = jr.split(_sample_key, _n_samples)
_samples = vmap(lambda k: sample_gp(k, prediction))(_sample_keys)

_fig, _ax = plot_dataset(train_dataset, train_dims, mixture=forecast_mixture, c_id=0, figsize=(8, 10), color_by_task=True, alpha=0.1)
_fig, _ax = plot_single_task_prediction(forecast_dataset, forecast_grid, train_dims, forecast_hyperposterior, forecast_pred_mixture,
    _forecast_task_id, c_id=0, samples=_samples, sample_alpha=0.1, fig=_fig, ax=_ax)
_fig, _ax = plot_task(forecast_held_out, train_dims, _forecast_task_id, c_id=0, fig=_fig, ax=_ax, color="red", marker="x")
_fig.suptitle(f"Prediction task-specific for task {_forecast_task_id}")

plt.show()

# %% [markdown]
r"""
Notice how the predictive distribution of the target task borrows information from the observations of the training tasks over the [48h, 72h] window. The *Multi-Task* component of Mimosa allows tasks to share knowledge via the model's **mean process**. During intervals with unobserved timestamps for a prediction task, Mimosa's prediction quickly converges toward the mean process inferred from the training data. Uncertainty increases, but significantly less than it would with a standard Gaussian Process. The dashed blue line depicts the cluster hyperposterior mean, while the solid black line shows the task-specific posterior mean.

To measure the value of the information transferred from other tasks, we now fit a single-task *Multi-Output* GP using the exact same outputs and missing window. Because this model only accesses observations from a single prediction task, it serves as an ablation baseline without any cross-task information transfer. You will have to redefine the `Dimensions` of the problem, the `ModelConfig` and the baseline `Grid` in order to train your baseline model.
"""

# %%
baseline_task_id = 1
baseline_dataset, baseline_held_out = forecast_remover(pred_dataset, t_id=baseline_task_id)

baseline_gp_dataset = GPDataset(inputs=pred_dataset.inputs[baseline_task_id], outputs=baseline_dataset.outputs[baseline_task_id],)

baseline_params = build_gp_parameters(
        GPParameters(
            mean=BlockMean(ZeroMean(), n_outputs=train_dims.O, output_hps_in_axes=None),
            kernel=LCMKernel(
                kernels=[VarianceKernel(10.0) * SEKernel(length_scale=1.0)],
                coregionalization_matrices=[jnp.ones((train_dims.O, 1))],
                kappas=[jnp.ones(train_dims.O)],
            ),
            noise_kernel=BlockDiagKernel(
                WhiteNoiseKernel(noise=0.05),
                n_outputs=train_dims.O,
                output_hps_in_axes=None,
            ),
        ),
        n_channels=train_dims.C,
    )
baseline_model = GPModel()
baseline_params = baseline_model.fit(baseline_gp_dataset, baseline_params)
baseline_prediction = baseline_model.predict(baseline_gp_dataset, train_grid, baseline_params,)

# %%
plt.close("all")
_x = np.asarray(train_grid.points[:, 0])
_mean = np.asarray(jnp.squeeze(baseline_prediction.mean))
_cov = np.asarray(jnp.squeeze(baseline_prediction.covariance))
_std = np.sqrt(np.diag(_cov))

_fig, _ax = plot_dataset(baseline_dataset, train_dims, t_id=baseline_task_id, c_id=0, figsize=(8, 6),legend=False,)

for _output_id in range(train_dims.O):
    _pred = slice(_output_id * len(_x), (_output_id + 1) * len(_x))
    _current_ax = _ax[_output_id, 0]
    _current_ax.plot(_x, _mean[_pred], color="black")
    _current_ax.fill_between(_x, _mean[_pred] - 1.96 * _std[_pred], _mean[_pred] + 1.96 * _std[_pred], color="black", alpha=0.2)

plot_task(baseline_held_out, train_dims, baseline_task_id, c_id=0, fig=_fig, ax=_ax, color="red", marker="x")
plt.show()

# %% [markdown]
r"""
Notice how the predictive mean quickly reverts to the prior mean, while the uncertainty explodes! When an interval lacks data across **all outputs** (both client 8 & client 16) for a single task, the standard *Multi-Output* framework behaves exactly as expected: the prediction slowly drifts back to the prior mean, accompanied by a sharply increasing variance.
"""

# %% [markdown]
r"""
## No dark magic here

The goal of this notebook is to provide an honest look at the model’s capabilities. This section highlights limitations that might not be immediately obvious from the previous results.

First, knowledge transfer between outputs relies entirely on cross-series dependencies. If outputs are nearly independent or only weakly correlated, the shared information will not be sufficient to recover a satisfying predictive distribution. Similarly, if an observation gap occurs across **all outputs** and **all tasks** over the exact same input interval, the model cannot magically reconstruct the missing trajectories; it degrades to a single-output, single-task baseline, reverting to the prior mean with high predictive uncertainty.

Moreover, there is no "one-size-fits-all" Multi-Output kernel. Depending on your problem, you may prefer an ICM, an LCM, or a Convolution Process kernel. We highly advise experimenting with different kernel designs to best model your data. For Convolution kernels in particular, a thoughtful initialization helps avoid poor local optima (e.g., setting the prior mean equal to the empirical mean of each output, or pre-optimizing hyperparameters to avoid a cold start).

Finally, the training time of a *Multi-Output Multi-Task* model scales **cubically** with the number of outputs. The complexity is $\mathcal{O}(O^3 \cdot T \cdot N^3)$, where $O$ is the number of outputs, $T$ is the number of tasks, and $N$ is the size of the `MultiOutputUnionGrid`. In this tutorial, we only accounted for clients 8 and 16 as our outputs, whereas the full *Electricity* dataset contains 321 clients. Training the exact same model on all 321 clients would take several **days** (or simply crash due to the sheer size of the covariance matrices) unless sparse approximations are used.

When designing your pipeline with Mimosa, you must carefully ask yourself: *what should represent an output, and what should represent a task?* Whether you treat a data source as an **output** or as a **task** will drastically impact both the modeling assumptions and the inference time of your model.
"""
