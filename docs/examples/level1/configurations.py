# %% tags=["remove-cell"]
import importlib.util, os, subprocess, sys
from pathlib import Path

if importlib.util.find_spec("mimosa") is None:
    # When running in Colab, you can select a GPU for execution and un-comment the next line
    # subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jax[cuda]"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mimosa-ml"], check=True)

# The docs build runs each notebook from its own level folder, while the data folder is shared at
# docs/examples/data. On Colab the notebook runs from /content, where neither exists.
if Path("../data").is_dir():
    os.chdir("..")
Path("data").mkdir(exist_ok=True)

# %% [markdown]
r"""
# Configurations

`ModelConfig` has seven flags. Each is shown below on synthetic data: same dimensions, same base
kernels, one flag changed at a time.

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level1/configurations.py` script!
"""

# %%
import dataclasses

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from kernax import ConstantMean, VarianceKernel, SEKernel, WhiteNoiseKernel, BlockMean, BlockDiagKernel, ICMKernel

from mimosa import Dimensions, ModelConfig, Parameters, generate_data, build_parameters, plot_dataset
from mimosa.data_structures import validate_model_config

key = jr.PRNGKey(0)
plt.rcParams["figure.dpi"] = 150
jax.devices()

# %% [markdown]
r"""
## Setup

Hyperparameters are set by hand, not sampled, so each flag's effect is obvious. `base_params` holds
the values used wherever a hyperparameter is *shared*.
"""

# %%
dims = Dimensions(T=6, K=2, I=1, C=2, O=1, N=100, G=300)

base_params = Parameters(
	cluster_mean=ConstantMean(0.0),
	cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=1.0),
	task_kernel=VarianceKernel(1.0) * SEKernel(length_scale=0.4),
	noise_kernel=WhiteNoiseKernel(noise=0.05),
)

# %% [markdown]
r"""
## Baseline

Everything shared. Tasks sample their own input locations (`isotopic_tasks=False`) and differ only
by their own GP draw.
"""

# %%
config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

dataset, _, _, mixture, _, _, _ = generate_data(key, dims, base_params, config, input_range=[(-2.5, 2.5)])

fig, _ = plot_dataset(dataset, dims, mixture=mixture, figsize=(6 * dims.C, 4))
fig.suptitle("baseline — everything shared")
plt.show()

# %% [markdown]
r"""
## `shared_task_hps=False`

Each task gets its own task-kernel hyperparameters. `build_parameters` gives the length scale one
value per task; setting them by hand needs `skip_parameter_build=True`, so `generate_data` uses the
parameters as given instead of rebuilding them.
"""

# %%
config = ModelConfig(
	shared_task_hps=False,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

params = build_parameters(base_params, dims, config)
params = dataclasses.replace(
	params, task_kernel=params.task_kernel.replace(length_scale=jnp.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0]))
)

dataset, _, _, mixture, _, _, _ = generate_data(
	key, dims, params, config, input_range=[(-2.5, 2.5)], skip_parameter_build=True
)

fig, _ = plot_dataset(dataset, dims, mixture=mixture, figsize=(6 * dims.C, 4))
fig.suptitle("shared_task_hps=False — task length scale 0.05 → 2.0")
plt.show()

# %% [markdown]
r"""
## `shared_cluster_hps=False`

Each mean-process gets its own mean and kernel: one cluster flat and low, the other wiggly and high.
"""

# %%
config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=False,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

params = build_parameters(base_params, dims, config)
params = dataclasses.replace(
	params,
	cluster_mean=params.cluster_mean.replace(constant=jnp.array([-8.0, 8.0])),
	cluster_kernel=params.cluster_kernel.replace(length_scale=jnp.array([3.0, 0.3])),
)

dataset, _, _, mixture, _, _, _ = generate_data(
	key, dims, params, config, input_range=[(-2.5, 2.5)], skip_parameter_build=True
)

fig, _ = plot_dataset(dataset, dims, mixture=mixture, figsize=(6 * dims.C, 4))
fig.suptitle("shared_cluster_hps=False — cluster means -8 / +8, length scales 3.0 / 0.3")
plt.show()

# %% [markdown]
r"""
## `shared_channel_hps=False`

Each channel gets its own hyperparameters. Left panel is wiggly, right one is smooth.
"""

# %%
config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=False,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

params = build_parameters(base_params, dims, config)
params = dataclasses.replace(
	params, task_kernel=params.task_kernel.replace(length_scale=jnp.array([0.05, 1.5]))
)

dataset, _, _, mixture, _, _, _ = generate_data(
	key, dims, params, config, input_range=[(-2.5, 2.5)], skip_parameter_build=True
)

fig, _ = plot_dataset(dataset, dims, mixture=mixture, figsize=(6 * dims.C, 4))
fig.suptitle("shared_channel_hps=False — channel length scales 0.05 / 1.5")
plt.show()

# %% [markdown]
r"""
## `cluster_specific_task_hps=True`

Task-kernel hyperparameters vary by cluster assignment: tasks of one cluster deviate from their
mean-process more sharply than tasks of the other.
"""

# %%
config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=True,
	isotopic_tasks=False,
)

params = build_parameters(base_params, dims, config)
params = dataclasses.replace(
	params, task_kernel=params.task_kernel.replace(length_scale=jnp.array([0.05, 1.5]))
)

dataset, _, _, mixture, _, _, _ = generate_data(
	key, dims, params, config, input_range=[(-2.5, 2.5)], skip_parameter_build=True
)

fig, _ = plot_dataset(dataset, dims, mixture=mixture, figsize=(6 * dims.C, 4))
fig.suptitle("cluster_specific_task_hps=True — task length scales 0.05 / 1.5 per cluster")
plt.show()

# %% [markdown]
r"""
## `isotopic_tasks=True`

Structural, not a hyperparameter: every task now shares the same set of input locations.
"""

# %%
config = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=True,
)

dataset, _, _, mixture, _, _, _ = generate_data(key, dims, base_params, config, input_range=[(-2.5, 2.5)])

print("inputs shape:", dataset.inputs.shape, "— one row of locations, shared by every task")

fig, _ = plot_dataset(dataset, dims, mixture=mixture, figsize=(6 * dims.C, 4))
fig.suptitle("isotopic_tasks=True — all tasks on the same input locations")
plt.show()

# %% [markdown]
r"""
## Reading a hyperparameter back

`build_parameters` add batch axis to hyperparameters depending on the config flags. The task kernel is wrapped
channel first, then cluster, then task — so the **task** axis is outermost:

	task_kernel.<hp_name>[task][cluster][channel]

The cluster mean and cluster kernel have no task axis:

	cluster_kernel.<hp_name>[cluster][channel]

A shared axis has size 1, so index it with `0`. When you combine multiple kernel, e.g: 
`VarianceKernel(...) * SEKernel(...)`, you can navigate the whole structure using `.left` and `.right`.
"""

# %%
config = ModelConfig(
	shared_task_hps=False,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

params = build_parameters(base_params, dims, config)
params = dataclasses.replace(
	params, task_kernel=params.task_kernel.replace(length_scale=jnp.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0]))
)

for t in range(dims.T):
	print(f"task {t} — length_scale =", params.task_kernel.right.length_scale[t][0][0])

print("cluster 0 mean constant =", params.cluster_mean.constant[0][0])

# %% [markdown]
r"""
Attribute access reads straight through the batch axes, so every value can be read at once. The
shape is `(task, cluster, channel)`, with `1` wherever the hyperparameter is shared.
"""

# %%
print("all task length scales:", params.task_kernel.right.length_scale.shape)
print(params.task_kernel.right.length_scale.squeeze())

# %% [markdown]
r"""
## Multi-output configurations

Two flags say whether outputs share input locations: `isotopic_output_in_grid` (in the grid) and
`isotopic_output_in_tasks` (in the observed points). Three of their four combinations are valid.

Multi-output kernels must already handle the output structure — here `ICMKernel`, `BlockMean` and
`BlockDiagKernel`.
"""

# %%
mo_dims = Dimensions(T=6, K=2, I=1, C=1, O=2, N=100, G=300)

mo_params = Parameters(
	cluster_mean=BlockMean(ConstantMean(0.0), n_outputs=mo_dims.O, output_hps_in_axes=0),
	cluster_kernel=ICMKernel(VarianceKernel(5.0) * SEKernel(length_scale=1.0), n_outputs=mo_dims.O, n_latent=mo_dims.O),
	task_kernel=ICMKernel(VarianceKernel(1.0) * SEKernel(length_scale=0.4), n_outputs=mo_dims.O, n_latent=mo_dims.O),
	noise_kernel=BlockDiagKernel(WhiteNoiseKernel(noise=0.05), n_outputs=mo_dims.O, output_hps_in_axes=0),
)

# %% [markdown]
r"""
### Both isotopic

Outputs share grid *and* observed locations: every input point carries a value for every output.
"""

# %%
config = ModelConfig(isotopic_tasks=False, isotopic_output_in_grid=True, isotopic_output_in_tasks=True)

dataset, grid, _, mixture, _, _, _ = generate_data(key, mo_dims, mo_params, config, input_range=[(-2.5, 2.5)])

print("grid:", len(grid.points), "points — task inputs:", dataset.inputs.shape)

fig, _ = plot_dataset(dataset, mo_dims, mixture=mixture, figsize=(6, 4 * mo_dims.O))
fig.suptitle("isotopic grid, isotopic tasks")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
### Isotopic grid, heterotopic tasks

One shared grid, but each output samples its own points from it — the usual case when outputs are
measured at different times.
"""

# %%
config = ModelConfig(isotopic_tasks=False, isotopic_output_in_grid=True, isotopic_output_in_tasks=False)

dataset, grid, _, mixture, _, _, _ = generate_data(key, mo_dims, mo_params, config, input_range=[(-2.5, 2.5)])

print("grid:", len(grid.points), "points — task inputs:", dataset.inputs.shape)

fig, _ = plot_dataset(dataset, mo_dims, mixture=mixture, figsize=(6, 4 * mo_dims.O))
fig.suptitle("isotopic grid, heterotopic tasks")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
### Both heterotopic

Each output gets its own grid, over its own input range — here output 0 on $[-2.5, 2.5]$ and
output 1 on $[0, 5]$.
"""

# %%
config = ModelConfig(isotopic_tasks=False, isotopic_output_in_grid=False, isotopic_output_in_tasks=False)

dataset, grid, _, mixture, _, _, _ = generate_data(
	key, mo_dims, mo_params, config, input_range=[(-2.5, 2.5), (0.0, 5.0)]
)

print("grid:", len(grid.points), "points — task inputs:", dataset.inputs.shape)

fig, _ = plot_dataset(dataset, mo_dims, mixture=mixture, figsize=(6, 4 * mo_dims.O))
fig.suptitle("heterotopic grid, heterotopic tasks")
plt.tight_layout()
plt.show()

# %% [markdown]
r"""
### The invalid combination

Points cannot share locations across outputs if they were drawn from a grid that doesn't.
`validate_model_config` rejects it.
"""

# %%
try:
	validate_model_config(ModelConfig(isotopic_output_in_grid=False, isotopic_output_in_tasks=True), mo_dims)
except ValueError as e:
	print("ValueError:", e)
