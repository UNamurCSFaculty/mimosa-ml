"""
Negative log-likelihoods used to optimise cluster and task hyperparameters: a standard
multivariate-normal NLL (`mvn_nll`), and its Magma-algorithm variant with a trace-correction
term accounting for the mean-process's posterior uncertainty (`magma_nll`).
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import vmap, Array
import equinox as eqx

from mimosa.linalg import cho_factor, cho_solve
from mimosa.data_structures import Dataset, Grid, Hyperposterior, Hyperprior, Mixture, Parameters
from mimosa.constants import DEFAULT_JITTER

__all__ = [
	"single_channel_mvn_nll",
	"mvn_nll",
	"single_channel_trace_correction",
	"trace_correction",
	"magma_nll",
	"clusters_nlls",
	"tasks_nlls",
	"elbo",
	"ClusterNLL",
	"TaskNLL",
]


def single_channel_mvn_nll(value: Array, mean: Array, cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of a multivariate normal distribution, for a single channel.

	Handles padded data: missing points are read from NaNs in `value`.

	Parameters
	----------
	value
		Observed values for this channel. Shape `(O*N,)`.
	mean
		Mean of the distribution, for this channel. Shape `(O*N,)`.
	cov
		Covariance of the distribution, for this channel. Shape `(O*N, O*N)`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood. Scalar.
	"""
	nan_mask = jnp.isnan(value)  # (O*N,)

	cov = jnp.where(nan_mask[None, :] | nan_mask[:, None], jnp.eye(cov.shape[-1]), cov)
	cov_l = cho_factor(cov, jitter=jitter)  # Shape (O*N, O*N)
	diff = jnp.where(nan_mask, 0.0, value - mean)  # Shape (O*N,)
	y = cho_solve(cov_l, diff[:, None])[:, 0]  # Shape (O*N,)

	data_fit = jnp.sum(diff * y)
	penalty = 2 * jnp.sum(jnp.log(jnp.diagonal(cov_l)))
	constant = (value.shape[0] - jnp.sum(nan_mask)) * jnp.log(2 * jnp.pi)

	return 0.5 * (data_fit + penalty + constant)


def mvn_nll(values: Array, mean: Array, cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of a multivariate normal distribution, vmapped across channels.

	See `single_channel_mvn_nll`.

	Parameters
	----------
	values
		Observed values for each channel. Shape `(O*N, C)`.
	mean
		Mean of the distribution. Shape `(C, O*N)`, with `C=1` if `shared_channel_hps`.
	cov
		Covariance of the distribution. Shape `(C, O*N, O*N)`, with `C=1` if `shared_channel_hps`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of each channel. Shape `(C,)`.
	"""
	mean_ax = None if mean.shape[0] == 1 else 0
	cov_ax = None if cov.shape[0] == 1 else 0
	f = vmap(single_channel_mvn_nll, in_axes=(0, mean_ax, cov_ax, None))
	return f(values.T, mean[0] if mean_ax is None else mean, cov[0] if cov_ax is None else cov, jitter)


def single_channel_trace_correction(value: Array, cov: Array, post_cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Trace correction term that adapts the negative log-likelihood of a MVN to the Magma algorithm,
	for a single channel: `0.5 * trace(post_cov @ inv(cov))`.

	Handles padded data: missing points are read from NaNs in `value`.

	Parameters
	----------
	value
		Observed values for this channel, used only for their NaN pattern (missing points). Shape `(O*N,)`.
	cov
		Covariance of the task or mean process, for this channel. Shape `(O*N, O*N)`.
	post_cov
		Posterior covariance of a specific mean process, for this channel. Shape `(O*N, O*N)`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Trace correction term. Scalar.
	"""
	nan_mask = jnp.isnan(value)  # (O*N,)
	nan_mask_2d = nan_mask[None, :] | nan_mask[:, None]
	eye = jnp.eye(cov.shape[-1])

	post_cov = jnp.where(nan_mask_2d, eye, post_cov)
	post_cov_l = cho_factor(post_cov, jitter=jitter)  # Shape (O*N, O*N)
	cov = jnp.where(nan_mask_2d, eye, cov)
	cov_l = cho_factor(cov, jitter=jitter)  # Shape (O*N, O*N)

	v = jsp.linalg.solve_triangular(cov_l, post_cov_l, lower=True)
	return 0.5 * (jnp.sum(v**2) - jnp.sum(nan_mask))


def trace_correction(values: Array, cov: Array, post_cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Trace correction term that adapts the negative log-likelihood of a MVN to the Magma algorithm,
	vmapped across channels.

	See `single_channel_trace_correction`.

	Parameters
	----------
	values
		Observed values for each channel, used only for their NaN pattern (missing points). Shape `(O*N, C)`.
	cov
		Covariance of the task or mean process. Shape `(C, O*N, O*N)`, with `C=1` if `shared_channel_hps`.
	post_cov
		Posterior covariance of a specific mean process. Shape `(C, O*N, O*N)`, with `C=1` if `shared_channel_hps`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Trace correction term of each channel. Shape `(C,)`.
	"""
	cov_ax = None if cov.shape[0] == 1 else 0
	post_ax = None if post_cov.shape[0] == 1 else 0
	f = vmap(single_channel_trace_correction, in_axes=(0, cov_ax, post_ax, None))
	return f(values.T, cov[0] if cov_ax is None else cov, post_cov[0] if post_ax is None else post_cov, jitter)


def magma_nll(values: Array, mean: Array, cov: Array, post_cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Full negative log-likelihood of a mean process in the Magma algorithm: `mvn_nll` plus the
	trace-correction term.

	Parameters
	----------
	values
		Observed values for each channel. Shape `(O*N, C)`.
	mean
		Posterior mean of a specific mean process. Shape `(C, O*G)`, with `C=1` if `shared_channel_hps`.
	cov
		Covariance of the task or mean process. Shape `(C, O*N, O*N)`, with `C=1` if `shared_channel_hps`.
	post_cov
		Posterior covariance of a specific mean process. Shape `(C, O*G, O*G)`, with `C=1` if `shared_channel_hps`.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of each channel. Shape `(C,)`.
	"""
	return mvn_nll(values, mean, cov, jitter=jitter) + trace_correction(values, cov, post_cov, jitter=jitter)


def clusters_nlls(hyperposterior: Hyperposterior, hyperprior: Hyperprior, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Negative log-likelihood of every mean-process, for each channel, under its prior.

	Parameters
	----------
	hyperposterior
		Posterior distribution over each mean-process's values at the grid points.
	hyperprior
		Prior distribution over each mean-process's values at the grid points.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of every mean-process, for each channel. Shape `(K, C)`.
	"""
	# A prior shared across mean-processes is read in place rather than expanded to the posterior's
	# `(K, C)` batch shape; one shared across channels is left for `magma_nll` to read the same way.
	# `Hyperprior` annotates `mean` and `covariance` with the same `*B`, so one axis covers the pair.
	prior_ax = None if hyperprior.mean.shape[0] == 1 else 0
	prior = hyperprior[0] if prior_ax is None else hyperprior

	f = vmap(magma_nll, in_axes=(0, prior_ax, prior_ax, 0, None))
	return f(hyperposterior.mean.mT, prior.mean, prior.covariance, hyperposterior.covariance, jitter)


def tasks_nlls(
	dataset: Dataset, grid: Grid, task_covs: Array, hyperposterior: Hyperposterior, jitter: Array = DEFAULT_JITTER
) -> Array:
	"""
	Negative log-likelihood of every task, under each mean-process, for each channel.

	Parameters
	----------
	dataset
		Dataset whose tasks' likelihoods are computed.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	task_covs
		Task covariance (including noise) of every task. Shape `(T, K, C, O*N, O*N)`, with `T=1` if
		`shared_task_hps`, `K=1` if `shared_cluster_hps` and `C=1` if `shared_channel_hps`.
	hyperposterior
		Posterior distribution over each mean-process's values at the grid points.
	jitter
		Diagonal jitter added before Cholesky factorization, for numerical stability.

	Returns
	-------
	Negative log-likelihood of every task, under each mean-process, for each channel. Shape `(T, K, C)`.
	"""
	task_ax = None if task_covs.shape[0] == 1 else 0
	cluster_ax = None if task_covs.shape[1] == 1 else 0

	def task_nll(outputs, mappings, task_cov):
		post = hyperposterior.marginal(mappings)  # (K, C, O*N)
		f = vmap(lambda p, t_c: magma_nll(outputs, p.mean, t_c, p.covariance, jitter), in_axes=(0, cluster_ax))
		return f(post, task_cov[0] if cluster_ax is None else task_cov)

	mappings = grid.mappings[0] if dataset.inputs.shape[0] == 1 else grid.mappings
	f = vmap(task_nll, in_axes=(0, None if mappings.ndim == 1 else 0, task_ax))
	return f(dataset.outputs, mappings, task_covs[0] if task_ax is None else task_covs)


def elbo(
	dataset: Dataset,
	grid: Grid,
	mixture: Mixture,
	parameters: Parameters,
	hyperposterior: Hyperposterior,
	jitter: Array = DEFAULT_JITTER,
) -> Array:
	"""
	Evidence lower bound of the model.

	The quantity the VEM loop maximises: it increases at every iteration (up to numerical error), so
	its relative gain is what `mimosa.models.EarlyStoppingModel` watches to decide convergence.

	Built from the same pieces the loop itself uses: `clusters_nlls` for the mean-processes under
	their prior, `tasks_nlls` weighted by the responsibilities (exactly the loss
	`mimosa.optimisers.optimise_tasks` minimises), plus the entropies of the mixture and of the
	hyperposterior.

	Since the mixture is updated *after* the parameters, the ELBO is meant to be evaluated on the
	carry as it leaves an iteration, i.e. new hyperparameters against the E-step's hyperposterior.

	Parameters
	----------
	dataset
		Dataset being fitted.
	grid
		Grid of points and mappings of `dataset`'s inputs onto it.
	mixture
		Current mixture.
	parameters
		Current model parameters (mean, kernels).
	hyperposterior
		Current hyperposterior, consistent with `parameters`.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.

	Returns
	-------
	ELBO. Scalar.
	"""
	hyperprior = Hyperprior(
		mean=parameters.cluster_mean(grid.points, output_ids=grid.output_ids),
		covariance=parameters.cluster_kernel(grid.points, output_ids=grid.output_ids),
	)
	cluster_term = clusters_nlls(hyperposterior, hyperprior, jitter=jitter).sum()

	task_kernel = parameters.task_kernel + parameters.noise_kernel
	if dataset.inputs.shape[0] == 1:
		output_ids = dataset.output_ids[0] if dataset.output_ids is not None else None
		task_covs = task_kernel(dataset.clean_inputs[0], output_ids=output_ids)
	else:
		task_covs = task_kernel(dataset.clean_inputs, output_ids=dataset.output_ids)

	task_term = (
		tasks_nlls(dataset, grid, task_covs, hyperposterior, jitter=jitter) * mixture.responsibilities[..., None]
	).sum()

	# Entropy of the mixture: sum_i sum_k tau_ik * log(pi_k / tau_ik). A task with zero
	# responsibility towards a cluster contributes nothing, but would evaluate log(0) first, so the
	# ratio is only formed where both terms are positive.
	responsibilities, proportions = mixture.responsibilities, mixture.proportions[None, :]
	defined = (responsibilities > 0) & (proportions > 0)
	log_frac = jnp.where(
		defined,
		jnp.log(jnp.where(defined, proportions, 1.0)) - jnp.log(jnp.where(defined, responsibilities, 1.0)),
		0.0,
	)
	mixture_entropy = jnp.sum(responsibilities * log_frac)

	# Entropy of each mean-process: sum(log(diag(chol(post_cov)))), i.e. 0.5 * logdet, summed over
	# mean-processes and channels.
	post_cov_l = cho_factor(hyperposterior.covariance, jitter=jitter)
	post_entropy = jnp.sum(jnp.log(jnp.diagonal(post_cov_l, axis1=-2, axis2=-1)))

	return -cluster_term - task_term + mixture_entropy + post_entropy


class ClusterNLL(eqx.Module):
	"""
	Callable wrapper around `clusters_nlls`, as an `equinox.Module`.
	"""

	def __call__(self, hyperposterior: Hyperposterior, hyperprior: Hyperprior, jitter: Array = DEFAULT_JITTER) -> Array:
		"""
		See `clusters_nlls`.
		"""
		return clusters_nlls(hyperposterior, hyperprior, jitter)


class TaskNLL(eqx.Module):
	"""
	Callable wrapper around `tasks_nlls`, as an `equinox.Module`.
	"""

	def __call__(
		self,
		dataset: Dataset,
		grid: Grid,
		task_covs: Array,
		hyperposterior: Hyperposterior,
		jitter: Array = DEFAULT_JITTER,
	) -> Array:
		"""
		See `tasks_nlls`.
		"""
		return tasks_nlls(dataset, grid, task_covs, hyperposterior, jitter)
