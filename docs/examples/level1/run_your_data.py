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
# Run your own data

A customisable version of {doc}`/examples/level1/basic_example`: same pipeline, but reading a CSV instead of
generating data. Every cell marked **TODO** is one you are expected to edit.

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level1/run_your_data.py` script!

## CSV format

One wide file, one row per (task, input coordinate):

```
TaskID,Input1,Output1_1,Output1_2
0,-2.399328859060403,-5.680749645326175,0.5690175502546957
0,-2.3322147651006713,-5.776689121226701,0.8544802736370158
```

| column | meaning |
| --- | --- |
| `TaskID` | which task the row belongs to. Any label; tasks are indexed in order of appearance. |
| `Input<i>` | input coordinates, one column per input dimension (`I` columns). |
| `Output<o>_<c>` | observed value, output `o`, channel `c`, both 1-based. |
| `Noise<o>_<c>` | *(optional)* known observation noise for that cell. All or none. |

Any other column is ignored. Tasks may have different numbers of rows, and rows need not be sorted.

Two sentinels, which are **not** the same thing:

* **empty cell** — that coordinate is not part of output `o`'s input axis for that task (heterotopic
  outputs). The point does not exist for that output.
* **`nan`** — the coordinate exists for that output, but the value is unobserved (missing data).

A foreign CSV with only `nan`s and no empty cells is read as isotopic outputs, which is safe: NaN
outputs are masked during fitting and prediction either way.

For one file per output, or for CSVs that don't follow the `Output<o>_<c>` naming, see `load_csv`'s
and `load_single_csv`'s docstrings (`output_groups`).
"""

# %%
import jax

jax.config.update("jax_enable_x64", True)
import jax.random as jr
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
	Dimensions, ModelConfig, Parameters,
	BasicModel, UnionGrid,
	generate_data, save_csv, load_csv, build_parameters, sample_gp,
	plot_dataset, plot_clusters, plot_single_task_prediction,
)

key = jr.PRNGKey(42)
plt.rcParams['figure.dpi'] = 300

# %% [markdown]
"""
## Loading

**TODO** — delete the cell below and point `CSV_PATH` at your own file. It only exists so this page
has something to read: it writes a toy dataset (32 tasks, 50 points, 1 input, 2 channels) in the
format above.
"""

# %%
demo_dims = Dimensions(T=32, K=2, I=1, C=2, O=1, N=50, G=150)
demo_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(5.0) * SEKernel(length_scale=.5),
	task_kernel=VarianceKernel(1.0) * SEKernel(length_scale=.4),
	noise_kernel=WhiteNoiseKernel(noise=.05),
)
key, gen_key = jr.split(key)
demo_dataset, *_ = generate_data(gen_key, demo_dims, demo_params, ModelConfig(), input_range=[(-2.5, 2.5)])
save_csv("data/dummy.csv", demo_dataset)

# %%
CSV_PATH = "data/dummy.csv"  # TODO: your file

dataset = load_csv(CSV_PATH)
print(f"inputs {dataset.inputs.shape}, outputs {dataset.outputs.shape}")

# %% [markdown]
"""
`Dimensions` describes the data; `T`, `N`, `C` and `I` are read off the loaded arrays, the rest is
yours to choose.

* `K` — how many clusters to look for. Start at 2-3; see {doc}`/examples/level2/turnip_example` for picking it
  empirically.
* `O` — number of correlated outputs. 1 unless your file has `Output2_*` columns.
* `G` — size of the training grid. With `UnionGrid` below it is the number of distinct input
  locations in the file.
"""

# %%
T, N, C = dataset.outputs.shape
I = dataset.inputs.shape[-1]
K, O = 2, 1  # TODO

grid = UnionGrid(dataset.inputs)  # TODO: RegularGrid / KMeansGrid for a smaller grid
G = grid.points.shape[0]

dims = Dimensions(T=T, K=K, I=I, C=C, O=O, N=N // O, G=G)
print(dims)

# %% [markdown]
"""
Always look at the data before fitting — a wrong `C`/`O` split or a bad unit shows up here.
"""

# %%
fig, ax = plot_dataset(dataset, dims, figsize=(8 * dims.C, 6), alpha=.4)
fig.suptitle(f"{dims.T} tasks from {CSV_PATH}")
plt.show()

# %% [markdown]
"""
## Configuring the model

`ModelConfig` says which hyperparameters are shared. Sharing everything is the cheapest and most
stable starting point; loosen one flag at a time. See {doc}`/examples/level1/configurations` for what each one costs.
"""

# %%
model_config = ModelConfig(  # TODO
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	# True iff every task observes the exact same input locations, which is exactly what `load_csv`
	# signals by collapsing `inputs`' leading axis to 1 -- so read it off the data rather than guess.
	isotopic_tasks=dataset.inputs.shape[0] == 1,
)

# %% [markdown]
"""
The starting parameters are a guess, but they must be **in your data's units**: a variance is in
output units squared, a length scale in input units. Order-of-magnitude right is enough; optimisation
does the rest.
"""

# %%
init_params = Parameters(  # TODO
	cluster_mean=ZeroMean(),                                    # ConstantMean(y_mean) if not centred
	cluster_kernel=VarianceKernel(2.0) * SEKernel(length_scale=1.),
	task_kernel=VarianceKernel(.5) * SEKernel(length_scale=.5),
	noise_kernel=WhiteNoiseKernel(noise=.5),
)
init_params = build_parameters(init_params, dims, model_config)

# %% [markdown]
"""
## Fitting
"""

# %%
key, model_key = jr.split(key)
model = BasicModel(prng_key=model_key, n_clusters=dims.K)

hyperposterior, mixture, fitted_params = model.fit(dataset, grid, init_params, n_iter=50)  # TODO: n_iter

print("cluster sizes:", jnp.bincount(mixture.assignments, length=dims.K))

# %%
fig, ax = plot_dataset(dataset, dims, mixture=mixture, figsize=(8 * dims.C, 6), alpha=.1)
fig, ax = plot_clusters(grid, dims, hyperposterior=hyperposterior, figsize=(8 * dims.C, 6), fig=fig, ax=ax)
fig.suptitle("Fitted clusters, tasks colored by assignment")
plt.show()

# %% [markdown]
"""
## Predicting

One Gaussian process per task *and* cluster. Keeping the task's most probable cluster is the usual
choice; set `k_id` by hand to see what another cluster would predict.
"""

# %%
predictions = model.predict(dataset, grid, mixture, fitted_params)  # (T, K, C, O*G)

t_id, c_id = 0, 0  # TODO
k_id = int(mixture.assignments[t_id])
prediction = predictions[t_id, k_id, c_id]

fig, ax = plot_single_task_prediction(
	dataset, grid, dims, hyperposterior, mixture, t_id, c_id, prediction=prediction, figsize=(8 * dims.C, 6)
)
fig.suptitle(f"Prediction — task {t_id}, channel {c_id}")
plt.show()

# %%
key, sample_key = jr.split(key)
sample_keys = jr.split(sample_key, 64)
samples = vmap(lambda k: sample_gp(k, prediction))(sample_keys)  # (S, O*G)

fig, ax = plot_single_task_prediction(
	dataset, grid, dims, hyperposterior, mixture, t_id, c_id, samples=samples, figsize=(8 * dims.C, 6)
)
fig.suptitle(f"Prediction samples — task {t_id}, channel {c_id}")
plt.show()

# %% [markdown]
"""
## Where to go next

* prediction outside the observed locations → end of {doc}`/examples/level1/basic_example`
* perform a true train-test split on real-world data → end of {doc}`/examples/level2/basic_mt_example`
* perform model selection, e.g: find an appropriate number of clusters → {doc}`/examples/level2/turnip_example`
* learn corelation between multiple outputs → {doc}`/examples/level2/multi_output_example`
* run on multiple independant channels in parallel → {doc}`/examples/level2/multi_channel_example`
"""
