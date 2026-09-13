"""
Update and initialise the soft-clustering Mixture of tasks into mean-processes.
"""

from abc import abstractmethod

from jax import Array
from jax.nn import softmax
import jax.numpy as jnp
import equinox as eqx
from kernax import AbstractKernel

from mimosa.nll import tasks_nlls
from mimosa.kmeans import soft_kmeans
from mimosa.data_structures import Dataset, Grid, Hyperposterior, Mixture
from mimosa.constants import DEFAULT_JITTER

__all__ = ["MixtureInitialiser", "KMeansMixtureInitialiser", "update_mixture", "MixtureUpdater"]


class MixtureInitialiser(eqx.Module):
	"""
	Base class for initialising a Mixture from a Dataset.
	"""

	@abstractmethod
	def __call__(self, dataset: Dataset) -> Mixture:
		"""
		Initialise a Mixture from `dataset`.

		Parameters
		----------
		dataset
			Dataset to initialise a Mixture for.

		Returns
		-------
		Initial Mixture.
		"""
		...


def _summary_statistics(outputs: Array) -> Array:
	"""
	Min, max, mean and std of each task's observations, per channel, ignoring missing points.

	Parameters
	----------
	outputs
		Output values. Shape `(T, N, C)`. NaN marks a missing (or masked-out) observation.

	Returns
	-------
	One row per task, holding that task's four statistics for each channel. Shape `(T, 4*C)`.

	Notes
	-----
	Concatenated along the feature axis, *not* stacked-then-reshaped. `jnp.stack` would put the
	statistic on the leading axis, and reshaping a `(4, T, C)` array to `(T, 4*C)` reinterprets the
	buffer rather than transposing it -- each row would then hold one statistic belonging to four
	different tasks.
	"""
	return jnp.concatenate(
		(
			jnp.nanmin(outputs, axis=1),
			jnp.nanmax(outputs, axis=1),
			jnp.nanmean(outputs, axis=1),
			jnp.nanstd(outputs, axis=1),
		),
		axis=-1,
	)


class KMeansMixtureInitialiser(MixtureInitialiser):
	"""
	Initialise a Mixture by soft k-means clustering of per-task summary statistics.

	Each task is summarised by the min, max, mean and standard deviation of its observations, per
	channel and -- when `n_outputs > 1` -- per output. Summarising per output matters for
	multi-output data: pooling the outputs into one summary makes two tasks that differ only in one
	output look nearly identical, which is exactly the situation a multi-output model exists for.

	Attributes
	----------
	prng_key
		`jax.random` PRNG key.
	n_clusters
		Number of mean-processes to initialise responsibilities for.
	n_outputs
		Number of correlated outputs to summarise separately. `1` (the default) summarises the whole
		task at once. When the Dataset's outputs share their input locations (`output_ids is None`),
		the count is read off the shapes instead and this attribute is not needed.
	n_restarts
		Number of k-means restarts to run, keeping the best. See `soft_kmeans`.
	"""

	prng_key: Array
	n_clusters: int
	n_outputs: int
	n_restarts: int

	def __init__(self, prng_key, n_clusters: int, n_outputs: int = 1, n_restarts: int = 8):
		self.prng_key = prng_key
		self.n_clusters = n_clusters
		self.n_outputs = n_outputs
		self.n_restarts = n_restarts

	def _output_ids(self, dataset: Dataset) -> tuple[int, None | Array]:
		"""
		Number of outputs to summarise separately, and which output each observation belongs to.

		`(1, None)` means "summarise the task as a whole". The count must be static (it decides how
		many feature blocks are built), so it comes from the shapes when the outputs share input
		locations, and from `n_outputs` otherwise -- `dataset.output_ids`' *values* are traced under
		`jit` and cannot be inspected here.
		"""
		if dataset.output_ids is None:
			# Outputs share input locations, so `outputs` is block-major over them: (T, O*N, C)
			# against inputs of (#T, N, I). O is then a ratio of static shapes.
			n_outputs = dataset.outputs.shape[1] // dataset.inputs.shape[1]
			if n_outputs == 1:
				return 1, None
			points_per_output = dataset.inputs.shape[1]
			return n_outputs, jnp.repeat(jnp.arange(n_outputs), points_per_output)

		if self.n_outputs == 1:
			return 1, None
		return self.n_outputs, dataset.output_ids

	def __call__(self, dataset: Dataset) -> Mixture:
		"""
		Cluster tasks by soft k-means over each task's per-output, per-channel min, max, mean and std.

		See `MixtureInitialiser.__call__`.
		"""
		n_outputs, output_ids = self._output_ids(dataset)

		if n_outputs == 1:
			features = _summary_statistics(dataset.outputs)  # (T, 4*C)
		else:
			output_ids = jnp.broadcast_to(output_ids, dataset.outputs.shape[:2])[..., None]
			features = jnp.concatenate(
				[_summary_statistics(jnp.where(output_ids == o, dataset.outputs, jnp.nan)) for o in range(n_outputs)],
				axis=-1,
			)  # (T, 4*O*C)

		# A task with no surviving observation for some output would otherwise contribute a NaN
		# feature, which poisons every distance in the k-means rather than just that one coordinate.
		features = jnp.nan_to_num(features)

		# Standardise before clustering: `soft_kmeans`' stiffness acts on raw squared distances, so
		# unstandardised features tie the clustering to the data's units. On a unit-scale dataset the
		# distances shrink until every responsibility goes uniform and the centers collapse onto the
		# global mean -- one cluster holding every task.
		features = (features - features.mean(axis=0)) / jnp.maximum(features.std(axis=0), 1e-9)

		_, resp = soft_kmeans(self.prng_key, features, self.n_clusters, n_restarts=self.n_restarts)
		return Mixture(responsibilities=resp)


def update_mixture(
	dataset: Dataset,
	grid: Grid,
	task_kernel: AbstractKernel,
	hyperposterior: Hyperposterior,
	mixture: Mixture,
	jitter: Array = DEFAULT_JITTER,
) -> Mixture:
	"""
	Update the tasks' responsibilities towards each mean-process, given the current hyperposterior.

	Parameters
	----------
	dataset
		Dataset whose tasks are being clustered.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	task_kernel
		Task covariance kernel (including noise).
	hyperposterior
		Current posterior distribution over each mean-process's values at the grid points.
	mixture
		Current mixture, whose proportions weight the likelihoods as a prior.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	Mixture with updated responsibilities, proportional to each task's likelihood under each mean-process.
	"""
	if dataset.inputs.shape[0] == 1:
		output_ids = dataset.output_ids[0] if dataset.output_ids is not None else None
		task_llhs = jnp.sum(
			tasks_nlls(
				dataset,
				grid,
				task_kernel(dataset.clean_inputs[0], output_ids=output_ids),
				hyperposterior,
				jitter=jitter,
			),
			axis=-1,
		)
	else:
		task_llhs = jnp.sum(
			tasks_nlls(
				dataset,
				grid,
				task_kernel(dataset.clean_inputs, output_ids=dataset.output_ids),
				hyperposterior,
				jitter=jitter,
			),
			axis=-1,
		)
	return Mixture(responsibilities=softmax(jnp.log(mixture.proportions[None, :]) - task_llhs, axis=1))


class MixtureUpdater(eqx.Module):
	"""
	Callable wrapper around `update_mixture`, as an `equinox.Module`.
	"""

	def __call__(
		self,
		dataset: Dataset,
		grid: Grid,
		task_kernel: AbstractKernel,
		hyperposterior: Hyperposterior,
		mixture: Mixture,
		jitter: Array = DEFAULT_JITTER,
	) -> Mixture:
		"""
		See `update_mixture`.
		"""
		return update_mixture(dataset, grid, task_kernel, hyperposterior, mixture, jitter=jitter)
