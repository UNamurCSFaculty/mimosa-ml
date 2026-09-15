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
for name in ("car_trajectories_aligned.csv", "car_trajectories_2c_2o.csv"):
    if not Path("data", name).exists():
        urlretrieve(f"{DATA_URL}/{name}", Path("data", name))

# %% [markdown]
r"""
# **Concepts behind Mimosa**
This notebook introduces the core components of the Mimosa framework and illustrates them through concrete examples.

Use the "launch" button to run it interactively in Colab or clone the repository and
run the `examples/level1/concepts_behind_mimosa.py` script!

---

Before diving into Mimosa, you should identify the various **sources of information** in your raw data.
Real-world problems often require capturing dependencies that go beyond standard input dimensions
(which standard Gaussian Processes already address) to account for multiple, correlated data sources.
This notebook is here to help you identify what should be a *Task*, an *Input*, an *Output* and a *Channel*,
so that you can take full advantage of the modeling capabilities of Mimosa.

The mindmap below outlines the architectural building blocks of the framework.
"""

# %% [markdown]
r"""
![The Mimosa framework](../../images/Mimosa_Framework.svg)
"""

# %% [markdown]
r"""
## An all-in-one example to navigate the model

To understand how these concepts interact, let's say you want to **track vehicle trajectories
in a roundabout** with multiple entry/exit points.

The trajectories come from [openDD](https://arxiv.org/abs/2007.08463). `load_csv` reads them straight into a
`Dataset`: 284 cars, 128 points each, two channels (X and Y, rescaled to the unit square) sharing
one time axis.
"""

# %%
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from mimosa import Dimensions, KMeansMixtureInitialiser, load_csv, plot_dataset

key = jr.PRNGKey(2026)
plt.rcParams["animation.html"] = "jshtml"  # lets an animation render itself, with no encoding dance
jax.devices()

# %%
dataset = load_csv("data/car_trajectories_aligned.csv")
N = dataset.inputs.shape[1]
dims = Dimensions(T=dataset.outputs.shape[0], K=6, I=1, C=2, O=1, N=N, G=N)
x, y = np.asarray(dataset.outputs[..., 0]), np.asarray(dataset.outputs[..., 1])

# %% [markdown]
r"""
Every vehicle crossing this intersection represents a distinct **task**, as its trajectory is a unique realization
of a broader driving behavior.
"""

# %%
task_id = 6
task_fig, task_ax = plt.subplots(figsize=(7, 6))
task_ax.plot(x.T, y.T, color="0.85", linewidth=0.6)  # the other cars, for context
task_ax.plot(x[task_id], y[task_id], color=(0.00, 0.60, 0.55), linewidth=2.5)
task_ax.scatter(x[task_id, 0], y[task_id, 0], color="green", s=55, zorder=3, label="start")
task_ax.scatter(x[task_id, -1], y[task_id, -1], color="red", s=55, zorder=3, label="end")
task_ax.set_aspect("equal", adjustable="box")
task_ax.set(xlabel="x coordinate", ylabel="y coordinate", title=f"One task in the roundabout: vehicle {task_id}")
task_ax.legend()
plt.show()

# %% [markdown]
r"""
Since drivers are limited to a finite set of standard routes, their trajectories naturally group into distinct
**clusters**. Consequently, every individual trajectory (task) is simply a *slight variation* of its assigned
cluster's core pattern.

Mimosa clusters tasks through a `Mixture`, and `KMeansMixtureInitialiser` is the very object its models
use to build one: soft k-means over each task's per-channel summary statistics.
"""

# %%
mixture = KMeansMixtureInitialiser(prng_key=key, n_clusters=dims.K)(dataset)
palette = plt.get_cmap("tab10")

cluster_fig, cluster_ax = plt.subplots(figsize=(8, 6))
lines = [cluster_ax.plot([], [], color=palette(k), linewidth=1.2, alpha=0.8)[0] for k in mixture.assignments]
cluster_ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()), xlabel="x coordinate",
               ylabel="y coordinate", title="Multiple trajectories of vehicles in the roundabout")
cluster_ax.set_aspect("equal", adjustable="box")

# Reveal the cars a few at a time, spread across the dataset to keep the routes diverse.
revealed = np.linspace(0, dims.T - 1, 40, dtype=int)
animation = FuncAnimation(
    cluster_fig,
    lambda frame: [line.set_data(x[t], y[t]) for t, line in enumerate(lines) if t <= revealed[frame]],
    frames=len(revealed), interval=180,
)
plt.close(cluster_fig)  # the animation is the output; this drops the duplicate still frame
animation

# %% [markdown]
r"""
Furthermore, each trajectory is plotted in a 2D plane using $(x, y)$ coordinates. We shouldn't assume a specific
correlation between the two *a priori*: cars can go in any direction in general. The link between $x$ and $y$ lives
in *the mixture*. To represent variables of interest whose values are not directly correlated but which can help
finding an appropriate mixture, Mimosa uses **channels**. This is an analogy of color channels in the convolution
layers of neural networks: computation paths that are parallel to each other.
"""

# %%
channels_fig, _ = plot_dataset(dataset, dims, mixture=mixture, kind="line", figsize=(8 * dims.C, 4), alpha=0.3)
channels_fig.suptitle("The same roundabout tasks viewed through two channels")
plt.show()

# %% [markdown]
r"""
Finally, suppose our goal is to predict, along those same axes, both the car previous movement and where it points.
Its position offset $(dx, dy)$ and its heading $(\cos \theta, \sin \theta)$ are two distinct **outputs**. Because
a car in a roundabout mostly points where it is going, these two outputs are highly correlated, illustrating how
the framework leverages relationships across all dimensions.

`car_trajectories_2c_2o.csv` stores both: its `Output1_*`/`Output2_*` columns give `load_csv` everything it needs
to return a two-output `Dataset` directly. Channel 1 gathers the x-ish components ($dx$, $\cos \theta$),
channel 2 the y-ish ones ($dy$, $\sin \theta$):
"""

# %%
mo_dataset = load_csv("data/car_trajectories_2c_2o.csv")
mo_N = mo_dataset.inputs.shape[1]
mo_dims = Dimensions(T=mo_dataset.outputs.shape[0], K=6, I=1, C=2, O=2, N=mo_N, G=mo_N)

# n_outputs=2: tasks are summarised per output, so two cars that differ in one output only stay distinguishable.
mo_mixture = KMeansMixtureInitialiser(prng_key=key, n_clusters=mo_dims.K, n_outputs=mo_dims.O)(mo_dataset)

outputs_fig, outputs_ax = plot_dataset(mo_dataset, mo_dims, mixture=mo_mixture, kind="line",
                                       figsize=(8 * mo_dims.C, 4 * mo_dims.O), alpha=0.3)
for row, output_name in enumerate(("position offset", "heading")):
    for col, channel_name in enumerate(("x component", "y component")):
        outputs_ax[row, col].set_title(f"output {row} ({output_name}), channel {col} ({channel_name})")
outputs_fig.suptitle("Channels and outputs for the roundabout tasks")
plt.show()

# %% [markdown]
r"""
## **Where does Mimosa fit in?**

The relationships between the different components of the Mimosa framework are displayed below. Please note that while each **output** of the model can comprise several **channels**, we have omitted them from the graphical representation for the sake of clarity.
"""

# %% [markdown]
r"""
![Graphical model of MOMTClust](../../images/Graphical_model_MOMTClust.png)
"""

# %% [markdown]
r"""
Let $\mathcal{X}$ be the input domain of dimension $\mathcal{I} \in \mathbb{N}$, $\mathcal{O}$ the set of **outputs**, $\mathcal{T}$ the set of **tasks** and $\mathcal{K}$ the set of **clusters**. For all $\mathbf{x} \in \mathcal{X}$, for all $k \in \mathcal{K}$, for all $t \in \mathcal{T}$,

\begin{equation}
    y_t(\mathbf{x}) = \mu_k(\mathbf{x}) + f_t(\mathbf{x}) + \epsilon_t(\mathbf{x})
\end{equation}

where
- $\mu_k(\cdot) \sim \mathcal{GP}\Big(m_k(.), k_{\theta_k}(\cdot, \cdot)\Big)$ is a shared mean process, with prior mean function $m_k(.)$ and prior covariance kernel $k_{\theta_k}(\cdot, \cdot)$ defined on $\mathcal{X} \times \mathcal{X}$ with values in the set of matrices $\mathcal{M}_{|\mathcal{O}|, |\mathcal{O}|}(\mathbb{R})$ and parametrized by $\theta_k$. $\forall k \in \mathcal{K}, \theta_k = \left\{ \left\{S_{k,o}\right\}_{o \in \mathcal{O}}, \left\{ \{l_{k,o,i}\}_{i=1}^{\mathcal{I}} \right\}_{o \in \mathcal{O}}, \left\{l_{k,i}\right\}_{i=1}^{\mathcal{I}} \right\}$.

- $f_t(\cdot) \sim \mathcal{GP} \Big(\mathbf0, k_{\theta_t}(\cdot, \cdot)\Big)$ is the task-specific process of task $t$, with prior covariance kernel $k_{\theta_t}(\cdot, \cdot)$ defined on $\mathcal{X} \times \mathcal{X}$ with values in $\mathcal{M}_{|\mathcal{O}|, |\mathcal{O}|}(\mathbb{R})$ and parametrized by $\theta_t$. $\forall t \in \mathcal{T}, \theta_t = \left\{ \left\{S_{t,o}\right\}_{o \in \mathcal{O}}, \left\{ \{l_{t,o,i}\}_{i=1}^{\mathcal{I}} \right\}_{o \in \mathcal{O}}, \left\{l_{t,i}\right\}_{i=1}^{\mathcal{I}} \right\}$.

- $\forall t \in \mathcal{T}, Z_t = (Z_{t,1}, \dots, Z_{t,|\mathcal{K}}|)^T$ is the latent cluster-assignment vector; $Z_t \sim \mathcal{M}(1, \mathbf{\pi})$, with $\pi= (\pi_1, \dots, \pi_{|\mathcal{K}|}))^T$ the vector of mixing proportions.


- $\epsilon_t(\cdot) \sim \mathcal{GP}\Big(\mathbf0, k_{\sigma_t^2}(\cdot, \cdot)\Big)$ is the error term, with prior covariance $k_{\sigma_t^2}(\cdot, \cdot)$ defined as a white noise kernel defined on $\mathcal{X} \times \mathcal{X}$ with values in $\mathcal{M}_{|\mathcal{O}|, |\mathcal{O}|}(\mathbb{R})$. This means that,  $\forall (\mathbf{x}, \mathbf{x}') \in \mathcal{X} \times \mathcal{X}, \quad \forall(o,o') \in \mathcal{O} \times \mathcal{O}, \quad  \Big(k_{\sigma_t^2}(\mathbf{x}, \mathbf{x}')\Big)_{o,o'} = \sigma_{t,o}^2 \in \mathbb{R}_+ \text{ if } \mathbf{x} = \mathbf{x}' \text{ and } o=o', 0 \text{ otherwise.}$
    Each output has its own specific noise.


We operate in a *Multi-Output* setting, so both $k_{\theta_k}(\cdot,\cdot)$ and $k_{\theta_t}(\cdot,\cdot)$ are defined as *Multi-Output* kernels. In the graphical model, we use the Convolution Process formalism to define them, but the model architecture remains the same with ICM and LCM approaches.
"""

# %% [markdown]
r"""
Depending on the specific problem you want to tackle, you can leverage different building blocks within Mimosa:
- **Single variable, shared pattern** : If you only have one target variable to predict and assume that your observed 
individuals share a common underlying behavior, the *Multi-Task* framework is your go-to tool (see the 
[swimmer performance example](../level2/basic_mt_example.ipynb)). 
In *Multi-Task*, the common mean process $\mu_0(.)$ enables information sharing across tasks, overcoming the 
limitations of classic GPs when making predictions far from a target task's observed data. For mathematical details, 
please refer to Arthur Leroy, Pierre Latouche, Benjamin Guedj, and Servane Gey. MAGMA: inference and prediction using 
multi-task Gaussian processes with common mean. *Machine Learning*, 111(5):1821–1849, May 2022.

  <br>

- **Single variable, grouped patterns**: If you still have a single target variable, but the trajectories of your 
individuals naturally separate into distinct subgroups, the *Multi-Task Clustering* framework (see the 
[turnip prices example](../level2/turnip_example.ipynb)) has you covered. *Multi-Task Clustering* is an 
extension of the *Multi-Task* approach: it introduces multiple mean GPs 
(instead of a single mean GP shared by all tasks in the dataset), each associated with a specific cluster. For further 
information, please refer to Arthur Leroy, Pierre Latouche, Benjamin Guedj, and Servane Gey. Cluster-specific 
predictions with multi-task Gaussian processes. *Journal of Machine Learning Research*, 24(5):1–49, 2023.

  <br>

- **Multidimensional variable, clustered patterns**: If the target variable you aim to predict is multidimensional, 
and your observed individuals can be grouped into distinct clusters (similar to the *Multi-Task Clustering* approach), 
the *Multi-Channel* framework (see the [car trajectories example](../level2/multi_channel_example.ipynb)) 
will help you share information across these dimensions. 
In *Multi-Channel*, the clustering mixture is used to transfer information from one channel to another (while 
channels remain conditionally independent given the mixture). The formal mathematical formulation is currently 
pending publication.

  <br>

- **Multiple variables, shared patterns**: If you need to predict multiple correlated target variables, and assume a 
shared underlying pattern across individuals for each of them, the *Multi-Output Multi-Task* framework (see 
the [electricity consumption example](../level2/multi_output_example.ipynb)) is exactly what you need. By 
combining the parsimonious underlying mean of the *Multi-Task* approach with 
the expressive covariance structure of *Multi-Output* models, the *Multi-Output Multi-Task* framework leverages 
distinct and complementary components of the GP framework. The formal mathematical formulation is currently pending 
publication.
"""
