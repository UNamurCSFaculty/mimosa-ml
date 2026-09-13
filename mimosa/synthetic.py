"""
Generate synthetic multi-task, multi-cluster datasets from GP priors (`generate_data`), and remove
data points at random to simulate missingness (`RandomDataRemover`).
"""

from abc import abstractmethod
import dataclasses
import jax.random as jr
import jax.numpy as jnp
from jax import vmap, Array
import equinox as eqx
from kernax import BatchModule, InputSpecificParamModule, WhiteNoiseKernel, AbstractKernel, AbstractMean, AbstractModule
from kernax.parametrisations import NonTrainableParametrisation
from kernax.hp_sampling import sample_hps_from_uniform_priors

from mimosa.data_structures import (
	Dimensions,
	Parameters,
	GPParameters,
	ParameterPriors,
	ModelConfig,
	Hyperprior,
	Mixture,
	Dataset,
	Grid,
	MultivariateNormal,
	DataRemovalConfig,
)
from mimosa.grid import RegularGrid
from mimosa.linalg import find_exact_mappings
from mimosa.sampling import sample_gp
from mimosa.constants import DEFAULT_JITTER

__all__ = [
	"generate_grid",
	"sample_inputs",
	"build_mean",
	"build_mean_kernel",
	"build_task_kernel",
	"known_noise_kernel",
	"build_parameters",
	"build_gp_parameters",
	"sample_parameters_from_priors",
	"generate_data",
	"AbstractDataRemover",
	"RandomDataRemover",
]


def generate_grid(dims: Dimensions, config: ModelConfig, bounds: list[tuple[float, float]]) -> Grid:
	"""
	Build a regular grid spanning `bounds` in every dimension, with as close to `G` points as possible.

	Parameters
	----------
	dims
		Dimensions of the dataset to generate, containing dims.G, dims.I and dims.O.
	config
		Model configuration, used for its `isotopic_output_in_grid` field when dims.O > 1.
	bounds
		Min and max value of the grid, applied to every dimension.

	Returns
	-------
	Grid of points, `output_ids` and `n_outputs`, with no mappings. Its points have shape `(~G, I)`
	if `isotopic_output_in_grid`, and `(O * ~G, I)` otherwise, where `~G` is
	`round(dims.G ** (1 / dims.I)) ** dims.I` -- the nearest `dims.I`-th power to `dims.G`, which may
	fall below it (see `generate_data`'s Notes).
	"""
	if config.isotopic_output_in_grid and len(bounds) > 1:
		raise ValueError("Cannot have different output bounds for isotopic outputs grid.")
	if not config.isotopic_output_in_grid and len(bounds) != dims.O:
		raise ValueError(f"Cannot build heterotopic grid for {dims.O} outputs with only {len(bounds)} bounds.")

	grid_size = max(round(dims.G ** (1 / dims.I)), 1)
	# Each output's bounds span every input dimension, so one `RegularGrid` per output block.
	blocks = [RegularGrid(bounds=(b,) * dims.I, n_points=grid_size).compute_points() for b in bounds]
	points = jnp.concat(blocks, axis=0)

	output_ids = None if config.isotopic_output_in_grid else jnp.repeat(jnp.arange(dims.O), grid_size**dims.I)
	return Grid(points=points, mappings=None, output_ids=output_ids, n_outputs=dims.O)


def sample_inputs(key: Array, grid: Grid, dims: Dimensions, config: ModelConfig) -> tuple[Array, None | Array, Array]:
	"""
	Sample `dims.N` input points per task (or per output) from `grid`, without replacement, and
	compute each sampled point's index (mapping) in `grid`.

	Sampling structure follows `config.isotopic_tasks`/`config.isotopic_output_in_tasks`/
	`config.isotopic_output_in_grid`: shared or distinct sampled points across tasks, and across outputs.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	grid
		Grid of points to sample from.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `isotopic_tasks`/`isotopic_output_in_tasks`/
		`isotopic_output_in_grid` fields.

	Returns
	-------
	inputs
		Sampled input points.
	output_ids
		Output ids of each input point.
	mappings
		Index of each sampled point in `grid`.
	"""
	# `generate_grid` rounds to a regular mesh, so a block holds `round(dims.G ** (1/dims.I)) ** dims.I`
	# points, which is `dims.G` only when `dims.G` is an exact `dims.I`-th power.
	G = len(grid.points) if config.isotopic_output_in_grid else len(grid.points) // dims.O
	output_mapping_offset = jnp.repeat(jnp.arange(dims.O), dims.N) * G

	if config.isotopic_output_in_grid:
		if config.isotopic_output_in_tasks:
			output_ids = None
			if config.isotopic_tasks:
				# Sample inputs once and broadcast to every task
				inputs = jr.choice(key, grid.points, (dims.N,), replace=False)[None, ...]
				mappings = find_exact_mappings(grid.points, inputs[0])[None, ...]

			else:
				# Vmap on multiple PRNG keys to sample distinct inputs for every task
				inputs = vmap(lambda k: jr.choice(k, grid.points, (dims.N,), replace=False))(jr.split(key, dims.T))
				mappings = vmap(lambda i: find_exact_mappings(grid.points, i))(inputs)

			if dims.O > 1:
				mappings = jnp.tile(mappings, dims.O) + output_mapping_offset

		else:
			if dims.O == 1:
				raise ValueError("Cannot have heterotopic outputs with only one output.")

			output_ids = jnp.repeat(jnp.arange(dims.O, dtype=int), dims.N)

			if config.isotopic_tasks:
				# Vmap on multiple PRNG keys to sample distinct inputs for each output, then broadcast to every task
				inputs = vmap(lambda k: jr.choice(k, grid.points, (dims.N,), replace=False))(jr.split(key, dims.O))
				mappings = vmap(lambda i: find_exact_mappings(grid.points, i))(inputs)

				inputs = inputs.reshape(dims.N * dims.O, dims.I)[None, ...]
				mappings = (mappings.reshape(dims.N * dims.O) + output_mapping_offset)[None, ...]

			else:
				inputs = vmap(lambda k: jr.choice(k, grid.points, (dims.N,), replace=False))(
					jr.split(key, dims.T * dims.O)
				)
				mappings = vmap(lambda i: find_exact_mappings(grid.points, i))(inputs)

				inputs = inputs.reshape(dims.T, dims.N * dims.O, dims.I)
				mappings = mappings.reshape(dims.T, dims.N * dims.O) + output_mapping_offset

	else:
		if config.isotopic_output_in_tasks:
			raise ValueError("Cannot have heterotopic task inputs for each output sampled from an isotopic grid.")
		else:
			output_ids = jnp.repeat(jnp.arange(dims.O, dtype=int), dims.N)

			if config.isotopic_tasks:
				mappings = vmap(lambda k: jr.choice(k, jnp.arange(G), (dims.N,), replace=False))(jr.split(key, dims.O))
				mappings = (mappings.reshape(dims.N * dims.O) + output_mapping_offset)[
					None, ...
				]  # Broadcast to every tasks
				inputs = grid.points[mappings[0]][None, ...]  # Broadcast to every tasks

			else:
				mappings = vmap(lambda k: jr.choice(k, jnp.arange(G), (dims.N,), replace=False))(
					jr.split(key, dims.O * dims.T)
				)
				mappings = (mappings.reshape(dims.T, dims.N * dims.O) + output_mapping_offset).reshape(
					dims.T, dims.O * dims.N
				)
				inputs = grid.points[mappings]

	return inputs, output_ids, mappings


def build_mean(mean: AbstractMean, dims: Dimensions, config: ModelConfig) -> AbstractModule:
	"""
	Batch `mean` across channel dimensions and mean-processes, according to `config`'s HP-sharing
	flags, for synthetic data generation.

	`mean` should be the "base" mean, i.e. the one used if all HPs were shared.

	Parameters
	----------
	mean
		Base mean function to batch.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `shared_channel_hps`/`shared_cluster_hps` fields.

	Returns
	-------
	Batched mean function, with independent hyperparameters per channel/mean-process where configured.
	"""
	# multi-channel HPs
	if not config.shared_channel_hps:
		mean = BatchModule(mean, batch_size=dims.C, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean = BatchModule(mean, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if not config.shared_cluster_hps:
		mean = BatchModule(mean, batch_size=dims.K, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean = BatchModule(mean, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	return mean


def build_mean_kernel(mean_kernel: AbstractKernel, dims: Dimensions, config: ModelConfig) -> AbstractModule:
	"""
	Batch `mean_kernel` across channel dimensions and mean-processes, according to `config`'s
	HP-sharing flags, for synthetic data generation.

	`mean_kernel` should be the "base" kernel, i.e. the one used if all HPs were shared. If
	`dims.O > 1`, it should already be wrapped in a `BlockKernel` to handle the multi-output
	structure (this function doesn't manage output-related config).

	Parameters
	----------
	mean_kernel
		Base kernel to batch.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `shared_channel_hps`/`shared_cluster_hps` fields.

	Returns
	-------
	Batched kernel, with independent hyperparameters per channel/mean-process where configured.
	"""
	# multi-channel HPs
	if not config.shared_channel_hps:
		mean_kernel = BatchModule(mean_kernel, batch_size=dims.C, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean_kernel = BatchModule(mean_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if not config.shared_cluster_hps:
		mean_kernel = BatchModule(mean_kernel, batch_size=dims.K, batch_in_axes=0, batch_over_inputs=False)
	else:
		mean_kernel = BatchModule(mean_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	return mean_kernel


def build_task_kernel(task_kernel: AbstractKernel, dims: Dimensions, config: ModelConfig) -> AbstractModule:
	"""
	Batch `task_kernel` across channel dimensions, mean-processes and tasks, according to `config`'s
	HP-sharing flags, for synthetic data generation. Used for both the task and noise kernels.

	`task_kernel` should be the "base" kernel, i.e. the one used if all HPs were shared. If
	`dims.O > 1`, it should already be wrapped in a `BlockKernel` to handle the multi-output
	structure (this function doesn't manage output-related config).

	Parameters
	----------
	task_kernel
		Base kernel to batch.
	dims
		Dimensions of the dataset to generate.
	config
		Model configuration, used for its `shared_channel_hps`, `cluster_specific_task_hps`,
		`shared_task_hps` and `isotopic_tasks` fields.

	Returns
	-------
	Batched kernel, with independent hyperparameters per channel/mean-process/task where configured.
	"""
	# multi-channel HPs
	if not config.shared_channel_hps:
		task_kernel = BatchModule(task_kernel, batch_size=dims.C, batch_in_axes=0, batch_over_inputs=False)
	else:
		task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# cluster HPs
	if config.cluster_specific_task_hps:
		task_kernel = BatchModule(task_kernel, batch_size=dims.K, batch_in_axes=0, batch_over_inputs=False)
	else:
		task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)

	# task HPs
	if config.shared_task_hps:
		if config.isotopic_tasks:
			task_kernel = BatchModule(task_kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)
		else:
			task_kernel = BatchModule(task_kernel, batch_size=dims.T, batch_in_axes=None, batch_over_inputs=True)
	else:
		if config.isotopic_tasks:
			task_kernel = BatchModule(task_kernel, batch_size=dims.T, batch_in_axes=0, batch_over_inputs=False)
		else:
			task_kernel = BatchModule(task_kernel, batch_size=dims.T, batch_in_axes=0, batch_over_inputs=True)

	return task_kernel


def known_noise_kernel(variances: Array, dims: Dimensions, config: ModelConfig) -> AbstractModule:
	"""
	Build a non-trainable, per-point white-noise kernel carrying `variances`.

	Add it to a `build_task_kernel`-batched noise kernel (`noise_kernel + known_noise_kernel(...)`) so
	that each observation's own known variance enters the task covariance. Batched channel, then
	cluster, then task, matching `build_task_kernel`'s axis order; only the task axis carries data.

	Parameters
	----------
	variances
		Per-point variances, shape `(T, N, C)` (NaN padding read as 0). A NaN would poison every
		hyperparameter gradient; padded points are masked downstream anyway.
	dims
		Dimensions of the dataset to fit.
	config
		Model configuration, used for its `isotopic_tasks` and `isotopic_output_in_tasks` fields.

	Returns
	-------
	Kernel evaluating to a diagonal covariance of shape `(T, 1, C, N, N)`.

	Raises
	------
	NotImplementedError
		If `variances` spans more rows than there are input points, i.e. several correlated outputs
		share their input locations: the noise then needs a block kernel.
	"""
	n_points = dims.N if config.isotopic_output_in_tasks else dims.O * dims.N
	if variances.shape[1] != n_points:
		raise NotImplementedError(
			"Multi-output datasets sharing their input locations are not supported: the known noise "
			f"spans {variances.shape[1]} rows, but the kernel only sees {n_points} input points."
		)

	kernel = InputSpecificParamModule(
		WhiteNoiseKernel(noise=0.0, noise_parametrisation=NonTrainableParametrisation()), n_points
	)
	kernel = BatchModule(kernel, batch_size=dims.C, batch_in_axes=0, batch_over_inputs=False)
	kernel = BatchModule(kernel, batch_size=1, batch_in_axes=None, batch_over_inputs=False)
	kernel = BatchModule(kernel, batch_size=dims.T, batch_in_axes=0, batch_over_inputs=not config.isotopic_tasks)
	return kernel.replace(noise=jnp.moveaxis(jnp.nan_to_num(variances), -1, 1))


def build_parameters(parameters: Parameters, dims: Dimensions, config: ModelConfig) -> Parameters:
	"""
	Batch every field of `parameters` (cluster mean/kernel, task/noise kernel) according to
	`config`'s hyperparameter-sharing flags.

	`parameters` should hold the "base" mean/kernels, i.e. the ones used if all HPs were shared. Self
	-contained: unlike `build_mean`/`build_mean_kernel`/`build_task_kernel`, doesn't require calling
	each one separately, so it can be reused outside `generate_data` (e.g. to build a model's initial
	parameters from a `ModelConfig`).

	Parameters
	----------
	parameters
		Base cluster mean/kernel and task/noise kernel to batch.
	dims
		Dimensions of the dataset to generate or fit.
	config
		Model configuration, used for its HP-sharing flags (see `build_mean`/`build_mean_kernel`/
		`build_task_kernel`).

	Returns
	-------
	`parameters` with every field batched, with independent hyperparameters per channel/cluster/task
	where configured.
	"""
	return Parameters(
		cluster_mean=build_mean(parameters.cluster_mean, dims, config),
		cluster_kernel=build_mean_kernel(parameters.cluster_kernel, dims, config),
		task_kernel=build_task_kernel(parameters.task_kernel, dims, config),
		noise_kernel=build_task_kernel(parameters.noise_kernel, dims, config),
	)


def build_gp_parameters(parameters: GPParameters, n_channels: int, shared_channel_hps: bool = True) -> GPParameters:
	"""
	Batch every field of `parameters` (GP mean, kernel, noise kernel) over the channel axis.

	`parameters` should hold the "base" mean/kernels, i.e. the ones used if all HPs were shared. If
	`O > 1`, they should already be wrapped in a multi-output mean/kernel (`BlockMean`,
	`BlockDiagKernel`, `ICMKernel`, ...) to handle the output structure.

	Parameters
	----------
	parameters
		Base GP mean, kernel and noise kernel to batch.
	n_channels
		Number of channels, i.e. `C`.
	shared_channel_hps
		If True, a single set of hyperparameters is held behind `n_channels` identical copies; if
		False, each channel carries its own.

	Returns
	-------
	`parameters` with every field batched over the channel axis, with independent hyperparameters per
	channel if `shared_channel_hps` is False.

	Notes
	-----
	Shared hyperparameters are expanded to `n_channels` copies rather than kept as the single copy
	`build_mean`/`build_task_kernel` leave (`batch_size=1`): `mimosa.prediction.predict_gp` vmaps the
	mean's channel axis against `outputs`' `C` columns, so a mean of leading size 1 cannot be paired
	with them.
	"""
	axes = None if shared_channel_hps else 0

	def batch(module: AbstractModule) -> AbstractModule:
		return BatchModule(module, batch_size=n_channels, batch_in_axes=axes, batch_over_inputs=False)

	return GPParameters(
		mean=batch(parameters.mean),
		kernel=batch(parameters.kernel),
		noise_kernel=batch(parameters.noise_kernel),
	)


def sample_parameters_from_priors(key: Array, parameters: Parameters, priors: ParameterPriors) -> Parameters:
	"""
	Sample every field of `parameters` (cluster mean/kernel, task/noise kernel) uniformly from
	`priors`.

	`parameters` should already be batched (e.g. via `build_parameters`), so that hyperparameters
	are sampled independently wherever `priors` and the batching structure allow. Self-contained: can
	be reused outside `generate_data` (e.g. to sample a model's initial parameters).

	Parameters
	----------
	key
		`jax.random` PRNG key.
	parameters
		Batched cluster mean/kernel and task/noise kernel whose hyperparameters are resampled.
	priors
		Min/max bounds for each parameter of `parameters`, used to sample its hyperparameters.

	Returns
	-------
	`parameters` with every field's hyperparameters resampled from `priors`.
	"""
	subkey1, subkey2, subkey3, subkey4 = jr.split(key, 4)
	return Parameters(
		cluster_mean=sample_hps_from_uniform_priors(subkey1, parameters.cluster_mean, priors.cluster_mean_priors),
		cluster_kernel=sample_hps_from_uniform_priors(subkey2, parameters.cluster_kernel, priors.cluster_kernel_priors),
		task_kernel=sample_hps_from_uniform_priors(subkey3, parameters.task_kernel, priors.task_kernel_priors),
		noise_kernel=sample_hps_from_uniform_priors(subkey4, parameters.noise_kernel, priors.noise_kernel_priors),
	)


def generate_data(
	key: Array,
	dims: Dimensions,
	parameters: Parameters,
	config: ModelConfig,
	priors: ParameterPriors | None = None,
	input_range: None | list[tuple[float, float]] = None,
	jitter: Array = DEFAULT_JITTER,
	skip_parameter_build: bool = False,
) -> tuple[Dataset, Grid, Hyperprior, Mixture, Parameters, Array, MultivariateNormal]:
	"""
	Generate a synthetic multi-task, multi-cluster dataset from GP priors.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	dims
		Dimensions of the dataset to generate.
	parameters
		Cluster mean/kernel and task/noise kernels, used as priors to sample the cluster processes.
	config
		Model configuration: hyperparameter-sharing structure (task/cluster/channel/output) and
		input-sampling structure (isotopic tasks/outputs).
	priors
		Min/max bounds for each parameter of `parameters`, used to sample its hyperparameters.
		If None, hyperparameters are left unchanged.
	input_range
		Min and max value for input points of every output. Applied to every input dimension. Default is (-50, 50) for every output.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.
	skip_parameter_build
		If True, `parameters` is used as-is, without calling `build_parameters` on it. Pass an
		already-batched `Parameters` (e.g. `build_parameters(...)` whose hyperparameters were then
		set per task/cluster/channel) to generate data from chosen hyperparameters rather than from
		a single base value or from `priors`. `parameters` must already match `dims` and `config`.

	Returns
	-------
	dataset
		Generated inputs and outputs.
	grid
		Grid of points and each task's mapping onto it.
	hyperprior
		Prior distribution over each mean-process's values at the grid points.
	mixture
		Cluster proportions and hard task-to-cluster assignments.
	parameters
		Sampled Parameters (cluster mean/kernel, task/noise kernels) used for generation.
	cluster_means
		Sampled mean-process values at the grid points. Shape `(K, C, O*G)`.
	tasks
		Task processes' mean and covariance, evaluated at each task's sampled input points.

	Notes
	-----
	When `dims.I > 1` (aka multi-dimensional inputs), `len(grid.points)` is not guaranteed to be the provided `dims.G`.
	This is because we use a regular mesh-grid with the same number of points along each dimension. For example, if you
	want G=90 grid points on a 2D plane, you will actually obtain a grid with 9*9 = 81 points. For this reason, we recommend
	setting `dims.G` to "some integer to the power of `dims.I`" to get the exact expected grid length.
	"""
	if input_range is None:
		if config.isotopic_output_in_grid:
			input_range = [(-50, 50)]
		else:
			input_range = [(-50, 50) for _ in range(dims.O)]

	if dims.I > 1:
		grid_size = max(round(dims.G ** (1 / dims.I)), 1)
		if grid_size**dims.I != dims.G:
			raise ValueError(
				f"dims.G={dims.G} is not an integer to the power of dims.I={dims.I}. "
				f"Closest valid values are {grid_size**dims.I} and {(grid_size + 1) ** dims.I}."
			)

	# Step 1: generate the grid
	grid = generate_grid(dims, config, input_range)

	# Step 2: sample the input grid
	inputs, output_ids, mappings = sample_inputs(key, grid, dims, config)  # Varying shapes

	# Step 3: batch kernels, unless the caller already did
	if not skip_parameter_build:
		parameters = build_parameters(parameters, dims, config)

	# Step 4: sample HPs from priors
	if priors is not None:
		key, subkey = jr.split(key)
		parameters = sample_parameters_from_priors(subkey, parameters, priors)

	# Step 5: sample mean processes for each cluster from the mean and mean kernel, evaluated on the grid
	# mean has shape (K, C, O*G), cov has shape (K, C, O*G, O*G)
	hyperprior = Hyperprior(
		mean=parameters.cluster_mean(grid.points, output_ids=grid.output_ids),
		covariance=parameters.cluster_kernel(grid.points, output_ids=grid.output_ids),
	)

	if config.shared_channel_hps:
		sample_channels = vmap(lambda k, m, c: sample_gp(k, m[0], c[0], jitter=jitter), in_axes=(0, None, None))
		if config.shared_cluster_hps:
			sample_clusters = vmap(lambda k, m, c: sample_channels(k, m[0], c[0]), in_axes=(0, None, None))
		else:
			sample_clusters = vmap(lambda k, m, c: sample_channels(k, m, c), in_axes=(0, 0, 0))
	else:
		sample_channels = vmap(lambda k, m, c: sample_gp(k, m, c, jitter=jitter), in_axes=(0, 0, 0))
		if config.shared_cluster_hps:
			sample_clusters = vmap(lambda k, m, c: sample_channels(k, m[0], c[0]), in_axes=(0, None, None))
		else:
			sample_clusters = vmap(lambda k, m, c: sample_channels(k, m, c), in_axes=(0, 0, 0))
	key, subkey = jr.split(key)
	subkeys = jr.split(subkey, (dims.K, dims.C))

	cluster_means = sample_clusters(subkeys, hyperprior.mean, hyperprior.covariance)  # Shape (K, C, O*G)

	# Step 6: assign tasks to clusters
	responsibilities = jnp.eye(dims.K)[
		jnp.array(jnp.floor(jnp.arange(dims.T) / dims.T * dims.K), dtype=int)
	]  # Shape (T, K)
	mixture = Mixture(responsibilities=responsibilities)

	# Step 7: sample task processes for each task from the task kernel, evaluated on the task inputs
	task_means_on_grid = cluster_means[jnp.argmax(mixture.responsibilities, axis=1), ...]  # Shape (T, C, O*G)
	if config.isotopic_tasks:
		task_means = vmap(lambda t_m, m: t_m[:, m], in_axes=(0, None))(
			task_means_on_grid, mappings[0]
		)  # Shape (T, C, O*N)
	else:
		task_means = vmap(lambda t_m, m: t_m[:, m], in_axes=(0, 0))(task_means_on_grid, mappings)  # Shape (T, C, O*N)

	if output_ids is not None:
		# output_ids is shared across tasks here, but the algorithm should support it varying per
		# task too -- so its leading axis mirrors `inputs`' own (1 if isotopic_tasks, T otherwise),
		# rather than adding a dedicated sharing flag.
		dataset_output_ids = jnp.broadcast_to(
			output_ids, ((1 if config.isotopic_tasks else dims.T),) + output_ids.shape
		)
	else:
		dataset_output_ids = None

	if config.isotopic_tasks:
		task_covs = parameters.task_kernel(inputs[0], output_ids=output_ids) + parameters.noise_kernel(
			inputs[0], output_ids=output_ids
		)
	else:
		task_covs = parameters.task_kernel(inputs, output_ids=dataset_output_ids) + parameters.noise_kernel(
			inputs, output_ids=dataset_output_ids
		)
	# Shape (T, K, C, O*N, O*N), with T=1 if shared_task_hps, K=1 if not cluster_specific_task_hps and C=1 if shared_channel_hps

	if config.cluster_specific_task_hps:
		# Select covariance from the "right" cluster for each task
		task_covs = task_covs[
			jnp.arange(len(task_covs)), jnp.argmax(mixture.responsibilities, axis=1)
		]  # Shape (T, C, O*N, O*N) with T=1 if shared_task_hps and C=1 if shared_channel_hps
	else:
		task_covs = task_covs[:, 0, ...]  # Shape (T, C, O*N, O*N) with T=1 if shared_task_hps and C=1 if shared_channel_hps

	tasks = MultivariateNormal(mean=task_means, covariance=task_covs)

	if config.shared_channel_hps:
		sample_channels = vmap(lambda k, m, c: sample_gp(k, m, c[0], jitter=jitter), in_axes=(0, 0, None))
		if config.isotopic_tasks and config.shared_task_hps:
			sample_tasks = vmap(lambda k, m, c: sample_channels(k, m, c[0]), in_axes=(0, 0, None))
		else:
			sample_tasks = vmap(lambda k, m, c: sample_channels(k, m, c), in_axes=(0, 0, 0))
	else:
		sample_channels = vmap(lambda k, m, c: sample_gp(k, m, c, jitter=jitter), in_axes=(0, 0, 0))
		if config.isotopic_tasks and config.shared_task_hps:
			sample_tasks = vmap(lambda k, m, c: sample_channels(k, m, c[0]), in_axes=(0, 0, None))
		else:
			sample_tasks = vmap(lambda k, m, c: sample_channels(k, m, c), in_axes=(0, 0, 0))
	key, subkey = jr.split(key)
	subkeys = jr.split(subkey, (dims.T, dims.C))

	outputs = sample_tasks(subkeys, task_means, task_covs).mT  # Shape (T, O*N, C)

	dataset = Dataset(inputs=inputs, outputs=outputs, output_ids=dataset_output_ids)
	grid = dataclasses.replace(grid, mappings=mappings)

	return dataset, grid, hyperprior, mixture, parameters, cluster_means, tasks


class AbstractDataRemover(eqx.Module):
	"""
	Base class for modules that remove data points from a Dataset generated by `generate_data`.
	"""

	@abstractmethod
	def __call__(
		self, key: Array, dataset: Dataset, config: DataRemovalConfig, grid: Grid | None = None
	) -> Dataset | tuple[Dataset, Grid]:
		"""
		Remove data points from `dataset`, according to `config`.

		Parameters
		----------
		key
			`jax.random` PRNG key.
		dataset
			Dataset to remove points from.
		config
			Removal configuration.
		grid
			Kept for API parity, passed through unchanged: a missing point is marked by NaN in
			`dataset.outputs` only, inputs/grid are never touched.

		Returns
		-------
		Dataset with points removed, or `(Dataset, Grid)` if `grid` was given.
		"""
		...


class RandomDataRemover(AbstractDataRemover):
	"""
	Removes data points at random, per `DataRemovalConfig`. A missing point is marked by NaN in
	`dataset.outputs`; this masking is read downstream in `mimosa.hyperpost`, `mimosa.nll` and
	`mimosa.prediction`.
	"""

	def __call__(
		self, key: Array, dataset: Dataset, config: DataRemovalConfig, grid: Grid | None = None
	) -> Dataset | tuple[Dataset, Grid]:
		"""
		See `AbstractDataRemover.__call__`.
		"""
		T, ON, C = dataset.outputs.shape
		# `outputs`' point axis is O contiguous blocks of N (see `sample_inputs`). `output_ids` is None
		# exactly when outputs share input locations, and `inputs` then holds a single block of N.
		N = dataset.inputs.shape[1] if dataset.output_ids is None else ON // (int(dataset.output_ids.max()) + 1)
		O = ON // N

		if config.same_missing_across_outputs and dataset.output_ids is not None:
			raise ValueError(
				"Cannot share missingness across outputs when they do not share input "
				"locations. Set `same_missing_across_outputs=False`, or generate the "
				"dataset with `isotopic_output_in_tasks=True`."
			)

		# 1: random remove_mask, shape (T, O, N, C). "Which k of N points are removed" is drawn without
		# replacement by ranking iid uniform scores per row and keeping the k lowest ranks: no python loop,
		# no vmap, works whether the row is (task,), (task, output) or (task, output, channel). Rows are
		# drawn once and broadcast over the axes missingness is shared along.
		o = 1 if config.same_missing_across_outputs else O
		shape = (T, o, N) if config.same_missing_across_channels else (T, o, C, N)
		key_scores, key_counts = jr.split(key)
		ranks = jnp.argsort(jnp.argsort(jr.uniform(key_scores, shape), axis=-1), axis=-1)
		counts = (
			jr.randint(key_counts, shape[:-1], 0, config.max_missing + 1)
			if config.random_missing_count
			else jnp.full(shape[:-1], config.max_missing)
		)
		selected = ranks < counts[..., None]  # shape (T, o, N) or (T, o, C, N)

		selected = (
			selected[..., None] if config.same_missing_across_channels else jnp.moveaxis(selected, -2, -1)
		)  # (T, o, C, N) -> (T, o, N, C)
		remove_mask = jnp.broadcast_to(selected, (T, O, N, C)).reshape(T, ON, C)

		# 2: remove outputs (and known noise, if any) in one line
		outputs = jnp.where(remove_mask, jnp.nan, dataset.outputs)
		outputs_known_noise = (
			None if dataset.known_output_noise is None else jnp.where(remove_mask, jnp.nan, dataset.known_output_noise)
		)

		dataset = Dataset(
			inputs=dataset.inputs,
			outputs=outputs,
			known_output_noise=outputs_known_noise,
			output_ids=dataset.output_ids,
		)
		return (dataset, grid) if grid is not None else dataset
