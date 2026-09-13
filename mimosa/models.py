"""
Top-level models, combining grid construction, mixture initialisation/update, hyperposterior
computation, hyperparameter optimisation, and prediction into a single fit/predict interface.
"""

from abc import abstractmethod
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jax_tqdm import loop_tqdm

from mimosa.hyperpost import Hyperpost
from mimosa.nll import ClusterNLL, TaskNLL
from mimosa.optimisers import ClusterOptimiser, TaskOptimiser, optimise_gp
from mimosa.mixture import KMeansMixtureInitialiser, MixtureInitialiser, MixtureUpdater
from mimosa.prediction import FunctionPredictor, GPPredictor, Predictor
from mimosa.data_structures import (
	Dataset,
	GPDataset,
	Grid,
	Mixture,
	Parameters,
	GPParameters,
	MultivariateNormal,
	Hyperposterior,
)
from mimosa.constants import DEFAULT_JITTER

__all__ = ["AbstractModel", "BasicModel", "GPModel"]


class AbstractModel(eqx.Module):
	"""
	Base class for models exposing a `fit`/`predict` interface.
	"""

	@abstractmethod
	def fit(self, *args, **kwargs):
		"""
		Fit the model's parameters and mixture to a Dataset.
		"""
		pass

	@abstractmethod
	def predict(self, *args, **kwargs):
		"""
		Predict outputs at the grid points for a fitted model.
		"""
		pass


class BasicModel(AbstractModel):
	"""
	Default model pipeline: k-means mixture initialisation, LBFGS-optimised cluster and task
	hyperparameters.

	Unlike grid construction (see `mimosa.grid.GridBuilder`), every step here is jit-compatible, so
	`fit`/`predict` are jitted end-to-end. The `Grid` itself is a required argument rather than an
	attribute: building it (e.g. via `mimosa.grid.UnionGrid`) isn't always jit-compatible, so it must
	be computed by the caller outside of `fit`/`predict`.

	Attributes
	----------
	mixture_initialiser
		Initialises the tasks' mixture responsibilities.
	mixture_updater
		Updates the tasks' mixture responsibilities during fitting.
	hyperpost
		Computes the hyperposterior over each mean-process's values at the grid points.
	cluster_nll
		Negative log-likelihood used to optimise the cluster mean and kernel hyperparameters.
	task_nll
		Negative log-likelihood used to optimise the task and noise kernel hyperparameters.
	cluster_optimiser
		Optimiser for the cluster mean and kernel hyperparameters.
	task_optimiser
		Optimiser for the task and noise kernel hyperparameters.
	predictor
		Computes predictions from a fitted model: `mimosa.prediction.FunctionPredictor` for the
		latent function, `mimosa.prediction.ObservationPredictor` to include observation noise.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.
	"""

	mixture_initialiser: MixtureInitialiser
	mixture_updater: MixtureUpdater
	hyperpost: Hyperpost
	cluster_nll: ClusterNLL
	task_nll: TaskNLL
	cluster_optimiser: ClusterOptimiser
	task_optimiser: TaskOptimiser
	predictor: Predictor
	jitter: Array

	def __init__(
		self,
		prng_key: Array,
		n_clusters: int,
		jitter: Array = DEFAULT_JITTER,
		n_outputs: int = 1,
		predictor: Predictor = FunctionPredictor(),
	):
		"""
		Parameters
		----------
		prng_key
			`jax.random` PRNG key, used to initialise the mixture via k-means++.
		n_clusters
			Number of mean-processes in the mixture.
		jitter
			Diagonal jitter added before Cholesky factorizations, for numerical stability.
		n_outputs
			Number of correlated outputs, used only by the k-means mixture initialisation, which
			summarises each task per output rather than pooling them. Only needed when the outputs
			do *not* share their input locations (`dataset.output_ids is not None`); otherwise the
			count is read off the Dataset's shapes. See `mimosa.mixture.KMeansMixtureInitialiser`.
		predictor
			Whether `predict` returns the latent function (`FunctionPredictor`, the default) or an
			observation of it, including observation noise (`ObservationPredictor`).
		"""
		self.mixture_initialiser = KMeansMixtureInitialiser(prng_key, n_clusters, n_outputs)
		self.mixture_updater = MixtureUpdater()
		self.hyperpost = Hyperpost()
		self.cluster_nll = ClusterNLL()
		self.task_nll = TaskNLL()
		self.cluster_optimiser = ClusterOptimiser(
			solver=optx.LBFGS(atol=1e-3, rtol=1e-3),
			nll=self.cluster_nll,
		)
		self.task_optimiser = TaskOptimiser(
			solver=optx.LBFGS(atol=1e-3, rtol=1e-3),
			nll=self.task_nll,
		)
		self.predictor = predictor
		self.jitter = jitter

	@eqx.filter_jit
	def fit(
		self,
		dataset: Dataset,
		grid: Grid,
		parameters: Parameters,
		init_hyperposterior: Hyperposterior | None = None,
		init_mixture: Mixture | None = None,
		freeze_hyperposterior: bool = False,
		freeze_mixture: bool = False,
		freeze_cluster_parameters: bool = False,
		freeze_task_parameters: bool = False,
		n_iter: int = 50,
	) -> tuple[Hyperposterior, Mixture, Parameters]:
		"""
		Fit the model's cluster/task hyperparameters and mixture responsibilities to a Dataset.

		Initialises the mixture and hyperposterior, then alternates, for `n_iter` iterations:
		1) optimising the parameters (M-step)
		2) computing the hyperposterior (E-step)
		3) updating the mixture (E-step)

		It starts on the M-step as the pre-loop initialisation already acts like a first E-step.

		Parameters
		----------
		dataset
			Dataset to fit the model to.
		grid
			Grid of points and mappings of `dataset`'s inputs onto it, e.g. from
			`mimosa.grid.UnionGrid`.
		parameters
			Initial model parameters (mean, kernels).
		init_hyperposterior
			Hyperposterior to start from. Defaults to computing one from the initial mixture and
			`parameters`. Required when `freeze_hyperposterior` is True.
		init_mixture
			Mixture to start from. Defaults to `mixture_initialiser` (k-means), or, when
			`init_hyperposterior` is given, to one E-step against it from uniform proportions.
		freeze_hyperposterior
			Keep `init_hyperposterior` fixed for the whole fit, skipping the E-step's hyperposterior
			update.
		freeze_mixture
			Keep the initial mixture fixed for the whole fit.
		freeze_cluster_parameters
			Keep `parameters.cluster_mean` and `parameters.cluster_kernel` fixed.
		freeze_task_parameters
			Keep `parameters.task_kernel` and `parameters.noise_kernel` fixed.
		n_iter
			Number of fitting iterations.

		Returns
		-------
		hyperposterior
			Fitted hyperposterior, consistent with the returned parameters.
		mixture
			Fitted mixture.
		parameters
			Fitted model parameters.
		"""
		if freeze_hyperposterior and init_hyperposterior is None:
			raise ValueError("Must specify `init_hyperposterior` if `freeze_hyperposterior` is True.")

		if init_mixture is not None:
			mixture = init_mixture
		elif init_hyperposterior is not None:
			T, K = len(dataset.outputs), self.mixture_initialiser.n_clusters
			mixture = self.mixture_updater(
				dataset,
				grid,
				parameters.task_kernel + parameters.noise_kernel,
				init_hyperposterior,
				Mixture(responsibilities=jnp.ones((T, K)) / K),  # Only for uniform mixture proportions
				jitter=self.jitter,
			)
		else:
			mixture = self.mixture_initialiser(dataset)

		if init_hyperposterior is not None:
			hyperposterior = init_hyperposterior
		else:
			hyperposterior = self.hyperpost(dataset, grid, mixture, parameters, jitter=self.jitter)

		@loop_tqdm(n_iter, desc=f"Training model for {n_iter} iterations:")
		def step(i, args):
			hyperposterior, mixture, parameters = args

			# As init is basically a preliminary e-step, we start with the m-step
			# --- M-step ---
			# Seeded from the carry, so a frozen half keeps its incoming value.
			cluster_mean, cluster_kernel = parameters.cluster_mean, parameters.cluster_kernel
			task_kernel, noise_kernel = parameters.task_kernel, parameters.noise_kernel

			if not freeze_cluster_parameters:
				cluster_mean, cluster_kernel = self.cluster_optimiser(
					parameters.cluster_mean, parameters.cluster_kernel, hyperposterior, grid, jitter=self.jitter
				).value

			if not freeze_task_parameters:
				optim_task = self.task_optimiser(
					parameters.task_kernel + parameters.noise_kernel,
					dataset,
					grid,
					hyperposterior,
					mixture,
					jitter=self.jitter,
				).value
				task_kernel, noise_kernel = optim_task.left, optim_task.right

			parameters = Parameters(
				cluster_mean=cluster_mean,
				cluster_kernel=cluster_kernel,
				task_kernel=task_kernel,
				noise_kernel=noise_kernel,
			)

			# --- E-step ---
			if not freeze_hyperposterior:
				hyperposterior = self.hyperpost(dataset, grid, mixture, parameters, jitter=self.jitter)

			if not freeze_mixture:
				mixture = self.mixture_updater(
					dataset,
					grid,
					parameters.task_kernel + parameters.noise_kernel,
					hyperposterior,
					mixture,
					jitter=self.jitter,
				)

			return hyperposterior, mixture, parameters

		return jax.lax.fori_loop(0, n_iter, step, (hyperposterior, mixture, parameters))

	@eqx.filter_jit
	def predict(self, dataset: Dataset, grid: Grid, mixture: Mixture, parameters: Parameters) -> MultivariateNormal:
		"""
		Predict outputs at the grid points, for a fitted model.

		Parameters
		----------
		dataset
			Dataset to condition predictions on.
		grid
			Grid of points and mappings of `dataset`'s inputs onto it, e.g. from
			`mimosa.grid.UnionGrid`.
		mixture
			Fitted mixture.
		parameters
			Fitted model parameters.

		Returns
		-------
		Predicted distribution over each task's outputs at the grid points, for every mean-process.
		"""
		hyperposterior = self.hyperpost(dataset, grid, mixture, parameters, jitter=self.jitter)
		return self.predictor(dataset, grid, hyperposterior, parameters, jitter=self.jitter)


class GPModel(AbstractModel):
	"""
	Vanilla (single-task) Gaussian process: LBFGS-optimised hyperparameters, then exact GP
	conditioning.

	`fit` is a single minimisation of the GP marginal likelihood returning the fitted `GPParameters`.

	Works with correlated outputs and channels too.

	Unlike `BasicModel`, the `Grid` is needed only to `predict`: with no task to align, nothing is
	mapped onto it during the fit.

	Attributes
	----------
	solver
		Optimistix minimiser for the GP's hyperparameters.
	predictor
		Computes predictions from a fitted model: `mimosa.prediction.GPPredictor`, whose `noisy` field
		selects the latent function or an observation of it.
	jitter
		Diagonal jitter added before Cholesky factorizations, for numerical stability.
	"""

	solver: optx.AbstractMinimiser
	predictor: GPPredictor
	jitter: Array

	def __init__(
		self,
		solver: optx.AbstractMinimiser = optx.LBFGS(atol=1e-3, rtol=1e-3),
		predictor: GPPredictor = GPPredictor(),
		jitter: Array = DEFAULT_JITTER,
	):
		"""
		Parameters
		----------
		solver
			Optimistix minimiser for the GP's hyperparameters.
		predictor
			Whether `predict` returns the latent function (`GPPredictor()`, the default) or an
			observation of it, including observation noise (`GPPredictor(noisy=True)`).
		jitter
			Diagonal jitter added before Cholesky factorizations, for numerical stability.
		"""
		self.solver = solver
		self.predictor = predictor
		self.jitter = jitter

	@eqx.filter_jit
	def fit(self, dataset: GPDataset, parameters: GPParameters) -> GPParameters:
		"""
		Fit the GP's mean, kernel and noise hyperparameters by maximum likelihood.

		Parameters
		----------
		dataset
			Observations of the Gaussian process to fit.
		parameters
			Initial GP mean, kernel and noise kernel, batched over channels (see
			`mimosa.synthetic.build_gp_parameters`).

		Returns
		-------
		Fitted GP parameters.
		"""
		return optimise_gp(parameters, dataset, solver=self.solver, jitter=self.jitter).value

	@eqx.filter_jit
	def predict(self, dataset: GPDataset, grid: Grid, parameters: GPParameters) -> MultivariateNormal:
		"""
		Predict the GP's outputs at the grid points, for a fitted model.

		Parameters
		----------
		dataset
			Observations of the Gaussian process to condition the prediction on.
		grid
			Grid of points to predict at, e.g. from `mimosa.grid.RegularGrid`. Its `mappings` are not
			needed here.
		parameters
			Fitted GP parameters.

		Returns
		-------
		Predicted distribution over the Gaussian process's channels at the grid points.
		"""
		return self.predictor(dataset, grid, parameters, jitter=self.jitter)
