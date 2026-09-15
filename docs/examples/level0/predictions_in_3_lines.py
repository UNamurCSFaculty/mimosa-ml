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
for name in ("train_swimmers_256.csv", "test_swimmers_128.csv"):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))

# %% [markdown]
"""
# Level 0 — predictions in 3 lines

Two CSVs — swimmers to learn from, swimmers to predict — and three lines of code. `TrainTestPipeline`
fixes every modelling choice to its most common setting, so there is nothing to configure.

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level0/predictions_in_3_lines.py` script!
"""

# %%
import jax

# Like every other tutorial: Gaussian processes factorise covariance matrices, which needs float64.
jax.config.update("jax_enable_x64", True)
import jax.random as jr
import matplotlib.pyplot as plt

from mimosa.pipelines import TrainTestPipeline

# %% [markdown]
"""
## The three lines
"""

# %%
mimosa = TrainTestPipeline(n_clusters=2, prng_key=jr.PRNGKey(42))
mimosa.load_train_data("data/train_swimmers_256.csv").load_test_data("data/test_swimmers_128.csv").fit()
predictions = mimosa.predict()

# %% [markdown]
"""
That is a full multi-task GP: 256 training swimmers clustered into 2 groups, and a predictive
distribution for each of the 128 test swimmers — in their own units, on a grid of 200 evenly-spaced
points.
"""

# %%
fig, ax = mimosa.plot()
plt.show()

# %%
fig, ax = mimosa.plot(t_id=0)
plt.show()

# %% [markdown]
"""
## What you do *not* get

`TrainTestPipeline` buys its three lines by **deciding everything for you**. It is single-output and
single-channel, its kernels are squared-exponential with a fixed starting guess, every
hyperparameter is shared across tasks and clusters, the grids are chosen for you, and the fitted
hyperparameters come back in *scaled* units rather than your data's. So it cannot:

* say how many clusters your data actually has, or let you compare fits — {doc}`/examples/level2/turnip_example`;
* use a different kernel, or start the optimisation from a guess in your own units —
  {doc}`/examples/level1/basic_example`;
* loosen what is shared across tasks, clusters and channels — {doc}`/examples/level1/configurations`;
* read *your* CSV, with its own columns, dimensions and missing values — {doc}`/examples/level1/run_your_data`;
* predict several correlated outputs, several channels, or non-Gaussian observations —
  {doc}`/tutorials_level2`;
* sample the prediction, keep its multi-modality, or predict anywhere you like —
  {doc}`/examples/level1/basic_example`.
  
But more importantly: Mimosa is *much more than just a predictive framework*! Getting a better understanding of
the underlying components of the framework is often required to run much more complex but much more *interesting*
experiments.

Still, hopefully this example may have convinced you that the underlying framework can be interesting to you!
Maybe, you were even able to run this full pipeline on you own data! If you want to explore the framework
one step at a time, start at {doc}`/examples/level1/basic_example`. You'll be able to run full experiment in no time!
"""
