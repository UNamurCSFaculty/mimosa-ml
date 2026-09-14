import jax
jax.config.update("jax_enable_x64", True)
import jax.random as jr
from jax import vmap
from kernax import ZeroMean, VarianceKernel, SEKernel, WhiteNoiseKernel

from mimosa import (
    Dimensions, ModelConfig, Parameters,
    BasicModel, UnionGrid, generate_data, sample_gp,
    plot_dataset, plot_clusters, plot_single_task_prediction,
)

key = jr.PRNGKey(0)

# 20 tasks, 2 clusters, 1D inputs/outputs, 25 points/task
dims = Dimensions(T=20, K=2, I=1, C=1, O=1, N=25, G=50)
config = ModelConfig(isotopic_tasks=False)

# Ground-truth parameters used to generate the synthetic dataset
init_params = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=1.0),
	task_kernel=VarianceKernel(0.1) * SEKernel(length_scale=2.0),
	noise_kernel=WhiteNoiseKernel(noise=0.01),
)

key, subkey = jr.split(key)
dataset, grid, hyperprior, mixture, parameters, cluster_means, tasks = generate_data(
	subkey, dims, init_params, config, input_range=[(-2.5, 2.5)])

# Fit a model on the generated dataset, starting from the same parameters
fit_grid = UnionGrid()(dataset.inputs)
model = BasicModel(jr.PRNGKey(1), n_clusters=dims.K)
hyperposterior, fitted_mixture, fitted_params = model.fit(dataset, fit_grid, parameters, n_iter=50)

# Predict the posterior distribution of every task, in every cluster
predictions = model.predict(dataset, fit_grid, fitted_mixture, fitted_params)
# Plot the fitted mean-processes over the dataset
fig, ax = plot_dataset(dataset, dims, mixture=mixture, alpha=.3)
fig, ax = plot_clusters(fit_grid, dims, hyperposterior=hyperposterior, fig=fig, ax=ax, legend=False)
fig.show()

# Draw samples from task 0's predictive distribution and plot them
t_id, c_id = 0, 0
k_id = int(fitted_mixture.assignments[t_id])  # task's dominant cluster
prediction = predictions[t_id, k_id, c_id]

key, sample_key = jr.split(key)
sample_keys = jr.split(sample_key, 64)
samples = vmap(lambda k: sample_gp(k, prediction.mean, prediction.covariance))(sample_keys)

fig, ax = plot_single_task_prediction(
	dataset, fit_grid, dims, hyperposterior, fitted_mixture, t_id, c_id, samples=samples)
fig.show()
