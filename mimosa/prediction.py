"""
Predict outputs at the grid points, for every task and every mean-process, by Gaussian process
conditioning of each task on its observed values and on the mean-process hyperposterior.
"""

from abc import abstractmethod

import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, Array
import equinox as eqx

from mimosa.linalg import cho_factor
from mimosa.data_structures import (
	Parameters,
	GPParameters,
	Dataset,
	GPDataset,
	Grid,
	Hyperposterior,
	PredictionMeanBlocks,
	PredictionCovBlocks,
	MultivariateNormal,
)
from mimosa.constants import DEFAULT_JITTER

__all__ = [
	"predict_task_channel",
	"predict_task_in_cluster",
	"predict_clusters",
	"predict",
	"predict_gp",
	"Predictor",
	"FunctionPredictor",
	"ObservationPredictor",
	"GPPredictor",
]


def predict_task_channel(
	output_obs: Array,
	post_mean_blocks: PredictionMeanBlocks,
	cov_blocks: PredictionCovBlocks,
	jitter: Array = DEFAULT_JITTER,
) -> MultivariateNormal:
	"""
	Predict a single task's channel at the grid points, under a single mean-process, by GP
	conditioning on the task's observed values.

	Handles padded data: missing points are read from NaNs in `output_obs`.

	Parameters
	----------
	output_obs
		Observed values of this channel, for this task. Shape `(O*N,)`.
	post_mean_blocks
		Mean-process hyperposterior mean at the task's observed points and at the grid points,
		for this channel.
	cov_blocks
		Combined mean-process and task covariance (observed/grid/cross blocks), for this channel.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over this task's channel at the grid points.
	"""
	padding_mask_1D = ~jnp.isnan(output_obs)[:, None]
	padding_mask_2D = padding_mask_1D & padding_mask_1D.T

	gamma_obs = jnp.where(padding_mask_2D, cov_blocks.cov_obs, jnp.eye(len(cov_blocks.cov_obs)))
	gamma_crossed = jnp.where(padding_mask_1D, cov_blocks.cov_crossed, 0.0)

	L = cho_factor(gamma_obs, jitter=jitter)
	z = lax.linalg.triangular_solve(L, gamma_crossed, left_side=True, lower=True)
	y = lax.linalg.triangular_solve(
		L, (jnp.nan_to_num(output_obs) - post_mean_blocks.mean_obs)[:, None], left_side=True, lower=True
	)[:, 0]

	pred_mean = post_mean_blocks.mean_grid + (z.T @ y)
	pred_cov = cov_blocks.cov_grid - (z.T @ z)

	return MultivariateNormal(mean=pred_mean, covariance=pred_cov)


def predict_task_in_cluster(
	output_obs: Array,
	post_mean_blocks: PredictionMeanBlocks,
	cov_blocks: PredictionCovBlocks,
	jitter: Array = DEFAULT_JITTER,
) -> MultivariateNormal:
	"""
	Predict a single task's channels at the grid points, under a single mean-process, vmapped
	across channels.

	See `predict_task_channel`.

	Parameters
	----------
	output_obs
		Observed values of this task, for every channel. Shape `(O*N, C)`.
	post_mean_blocks
		Mean-process hyperposterior mean at the task's observed points and at the grid points,
		batched over channel dimensions.
	cov_blocks
		Combined mean-process and task covariance (observed/grid/cross blocks), batched over
		channel dimensions.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over this task's channels at the grid points, batched over channel dimensions.
	"""
	if cov_blocks.cov_obs.shape[0] == 1:
		return vmap(predict_task_channel, in_axes=(0, 0, None, None))(
			output_obs.T, post_mean_blocks, cov_blocks[0], jitter
		)
	else:
		return vmap(predict_task_channel, in_axes=(0, 0, 0, None))(output_obs.T, post_mean_blocks, cov_blocks, jitter)


def predict_clusters(
	task_outputs: Array,
	mappings: Array,
	hyperposterior: Hyperposterior,
	task_cov_blocks: PredictionCovBlocks,
	jitter: Array = DEFAULT_JITTER,
) -> MultivariateNormal:
	"""
	Predict a single task's outputs at the grid points, under every mean-process, vmapped across
	mean-processes.

	See `predict_task_in_cluster`.

	Parameters
	----------
	task_outputs
		Observed values of this task, for every channel. Shape `(O*N, C)`.
	mappings
		Index of this task's observed points in the grid.
	hyperposterior
		Posterior distribution over every mean-process's values at the grid points.
	task_cov_blocks
		Task covariance (including noise), split into observed/grid/cross blocks, batched over
		mean-processes (or shared, if `task_cov_blocks.cov_obs.shape[0] == 1`).
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over this task's channels at the grid points, batched over mean-processes
	(and channel dimensions).
	"""
	post_obs = hyperposterior.marginal(mappings)

	cov_blocks = PredictionCovBlocks(
		cov_obs=post_obs.covariance + task_cov_blocks.cov_obs,
		cov_grid=hyperposterior.covariance + task_cov_blocks.cov_grid,
		cov_crossed=hyperposterior.cross_covariance(mappings) + task_cov_blocks.cov_crossed,
	)

	post_mean_blocks = PredictionMeanBlocks(mean_obs=post_obs.mean, mean_grid=hyperposterior.mean)

	if cov_blocks.cov_obs.shape[0] == 1:
		return vmap(predict_task_in_cluster, in_axes=(None, 0, None, None))(
			task_outputs, post_mean_blocks, cov_blocks[0], jitter
		)
	return vmap(predict_task_in_cluster, in_axes=(None, 0, 0, None))(task_outputs, post_mean_blocks, cov_blocks, jitter)


def _cross_grid(grid: Grid, output_ids: None | Array) -> tuple[Array, None | Array]:
	"""
	Grid points/output_ids to cross-covary against points labelled by `output_ids`.

	kernax's multi-output kernels require *both* sides of a 2-argument call to carry `output_ids`,
	or neither -- so when `grid.points` is an isotopic, unlabelled pool (`grid.output_ids is None`)
	but `output_ids` isn't, tile+label the grid side to match it (one copy per output). Otherwise
	(`output_ids is None`, or `grid.output_ids` is already set) `grid.points`/`grid.output_ids` are
	already consistent and are returned unchanged.

	Parameters
	----------
	grid
		Grid whose points are being cross-covaried against `output_ids`-labelled points.
	output_ids
		Output ids of the points `grid` is being cross-covaried against (e.g. `dataset.output_ids`).

	Returns
	-------
	points, output_ids
		`grid.points`/`grid.output_ids`, expanded and labelled if needed.
	"""
	if output_ids is None or grid.output_ids is not None:
		return grid.points, grid.output_ids
	points = jnp.tile(grid.points, (grid.n_outputs, 1))
	ids = jnp.repeat(jnp.arange(grid.n_outputs), len(grid.points))
	return points, ids


def predict(
	dataset: Dataset,
	grid: Grid,
	hyperposterior: Hyperposterior,
	parameters: Parameters,
	noisy: bool = False,
	jitter: Array = DEFAULT_JITTER,
) -> MultivariateNormal:
	"""
	Predict every task's outputs at the grid points, under every mean-process.

	See `predict_clusters`.

	Parameters
	----------
	dataset
		Dataset whose tasks are predicted.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	hyperposterior
		Posterior distribution over every mean-process's values at the grid points.
	parameters
		Model parameters (task/noise kernels) used to compute the task covariances.
	noisy
		Whether to include observation noise at the predicted points. Noise at a predicted point is
		independent of the noise on the observations, so it enters `cov_grid` only and never
		`cov_crossed` -- including where a grid point coincides with an observed input.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over every task's outputs at the grid points, batched over tasks,
	mean-processes and channel dimensions.

	Notes
	-----
	When using an `InputSpecificParamModule` to contain point-specific noise, setting `noisy=True`
	will result in an internal Kernax error, as we cannot make a noisy prediction on points for which
	we don't know the noise.
	"""
	if grid.mappings is None:
		raise ValueError("`grid.mappings` is `None`. Mappings must be computed to make a prediction.")

	cross_points, cross_ids = _cross_grid(grid, dataset.output_ids)

	if dataset.inputs.shape[0] == 1:
		extended_grid = grid.points
		output_ids = dataset.output_ids[0] if dataset.output_ids is not None else None

		task_cov_blocks = PredictionCovBlocks(
			cov_obs=parameters.task_kernel(dataset.clean_inputs[0], output_ids=output_ids)
			+ parameters.noise_kernel(dataset.clean_inputs[0], output_ids=output_ids),
			cov_grid=parameters.task_kernel(extended_grid, output_ids=grid.output_ids)
			+ (parameters.noise_kernel(extended_grid, output_ids=grid.output_ids) if noisy else 0.0),
			cov_crossed=parameters.task_kernel(
				dataset.clean_inputs[0], cross_points, output_ids=output_ids, output_ids2=cross_ids
			),
		)
	else:
		extended_grid = jnp.broadcast_to(grid.points, dataset.inputs.shape[:1] + grid.points.shape)
		extended_cross_points = jnp.broadcast_to(cross_points, dataset.inputs.shape[:1] + cross_points.shape)
		# extended_grid/extended_cross_points gain a leading T axis, so any output_ids passed
		# alongside them must too.
		extended_grid_ids = (
			None
			if grid.output_ids is None
			else jnp.broadcast_to(grid.output_ids, dataset.inputs.shape[:1] + grid.output_ids.shape)
		)
		extended_cross_ids = (
			None if cross_ids is None else jnp.broadcast_to(cross_ids, dataset.inputs.shape[:1] + cross_ids.shape)
		)

		task_cov_blocks = PredictionCovBlocks(
			cov_obs=parameters.task_kernel(dataset.clean_inputs, output_ids=dataset.output_ids)
			+ parameters.noise_kernel(dataset.clean_inputs, output_ids=dataset.output_ids),
			cov_grid=parameters.task_kernel(extended_grid, output_ids=extended_grid_ids)
			+ (parameters.noise_kernel(extended_grid, output_ids=extended_grid_ids) if noisy else 0.0),
			cov_crossed=parameters.task_kernel(
				dataset.clean_inputs,
				extended_cross_points,
				output_ids=dataset.output_ids,
				output_ids2=extended_cross_ids,
			),
		)

	mappings = grid.mappings[0] if dataset.inputs.shape[0] == 1 else grid.mappings

	task_cov_blocks, task_cov_axes = task_cov_blocks.over_tasks

	return vmap(
		predict_clusters,
		in_axes=(0, 0 if mappings.ndim == 2 else None, None, task_cov_axes, None),
	)(dataset.outputs, mappings, hyperposterior, task_cov_blocks, jitter)


def predict_gp(
	dataset: GPDataset,
	grid: Grid,
	parameters: GPParameters,
	noisy: bool = False,
	jitter: Array = DEFAULT_JITTER,
) -> MultivariateNormal:
	"""
	Predict a single Gaussian process's outputs at the grid points, by GP conditioning on its
	observed values.

	Parameters
	----------
	dataset
		Observations of the Gaussian process, conditioned on.
	grid
		Grid of points to predict at. Only `points`, `output_ids` and `n_outputs` are read: with no
		task axis there is nothing to map onto the grid, so `mappings` may be left out.
	parameters
		GP mean, kernel and noise kernel, batched over channels (see
		`mimosa.synthetic.build_gp_parameters`).
	noisy
		Whether to include observation noise at the predicted points. Noise at a predicted point is
		independent of the noise on the observations, so it enters `cov_grid` only and never
		`cov_crossed` -- including where a grid point coincides with an observed input.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Predicted distribution over the Gaussian process's channels at the grid points, of shapes
	`(C, O*G)` and `(C, O*G, O*G)`.
	"""
	x, obs_ids = dataset.inputs, dataset.output_ids
	cross_points, cross_ids = _cross_grid(grid, obs_ids)

	return predict_task_in_cluster(
		dataset.outputs,
		PredictionMeanBlocks(
			mean_obs=parameters.mean(x, output_ids=obs_ids),
			mean_grid=parameters.mean(grid.points, output_ids=grid.output_ids),
		),
		PredictionCovBlocks(
			cov_obs=parameters.kernel(x, output_ids=obs_ids) + parameters.noise_kernel(x, output_ids=obs_ids),
			cov_grid=parameters.kernel(grid.points, output_ids=grid.output_ids)
			+ (parameters.noise_kernel(grid.points, output_ids=grid.output_ids) if noisy else 0.0),
			cov_crossed=parameters.kernel(x, cross_points, output_ids=obs_ids, output_ids2=cross_ids),
		),
		jitter,
	)


class Predictor(eqx.Module):
	"""
	Base class for callable wrappers around `predict`, as `equinox.Module` subclasses.

	Subclasses set `noisy`, which selects what is predicted: the latent function, or an observation
	of it at the predicted points.
	"""

	@property
	@abstractmethod
	def noisy(self) -> bool:
		"""
		Whether the prediction includes observation noise at the predicted points. See `predict`.
		"""

	def __call__(
		self,
		dataset: Dataset,
		grid: Grid,
		hyperposterior: Hyperposterior,
		parameters: Parameters,
		jitter: Array = DEFAULT_JITTER,
	) -> MultivariateNormal:
		"""
		See `predict`.
		"""
		return predict(dataset, grid, hyperposterior, parameters, self.noisy, jitter)


class FunctionPredictor(Predictor):
	r"""
	Computes the prediction for the function $f_t(x_t)$ of each task in `dataset`.

	As we predict the *function*, prediction doesn't include noise at predicted points, conditioned
	on noisy observations.
	"""

	noisy = False


class ObservationPredictor(Predictor):
	r"""
	Computes the prediction for the observations $y_t = f_t(x_t) + \varepsilon$ of each task in
	`dataset`.

	As we predict an *observation*, prediction includes observation noise at predicted points.

	Will not work with an `InputSpecificParamModule` in `noise_kernel`.
	"""

	noisy = True


class GPPredictor(eqx.Module):
	r"""
	Callable wrapper around `predict_gp`, as an `equinox.Module`.

	Attributes
	----------
	noisy
		Whether to predict an observation $y(x^*) = f(x^*) + \varepsilon$ rather than the latent
		function $f(x^*)$, i.e. whether to include observation noise at the predicted points. See
		`predict_gp`.
	"""

	noisy: bool = eqx.field(static=True, default=False)

	def __call__(
		self,
		dataset: GPDataset,
		grid: Grid,
		parameters: GPParameters,
		jitter: Array = DEFAULT_JITTER,
	) -> MultivariateNormal:
		"""
		See `predict_gp`.
		"""
		return predict_gp(dataset, grid, parameters, self.noisy, jitter)
