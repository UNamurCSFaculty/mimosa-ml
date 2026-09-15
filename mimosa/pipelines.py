"""
The highest-level API mimosa exposes: two datasets, a number of clusters, a PRNG key, nothing else.

`BasicModel` already hides the EM loop, but a caller still has to know about grids, `Dimensions`,
`ModelConfig`, parameter batching, which grid fitting uses versus prediction, how to re-derive the
mixture on unseen tasks, and that hyperparameters live in the data's own units. `TrainTestPipeline`
below fixes every one of those choices to its most common setting and keeps the user in the units
they came in with:

* zero-mean, squared-exponential cluster and task kernels, white observation noise;
* single output, single channel, hyperparameters shared across tasks/clusters/channels, no
  cluster-specific task hyperparameters, heterotopic tasks;
* inputs min-max scaled to [0, 1] and outputs standardised, both from the *training* data, so the
  fixed initial hyperparameter guess is on scale and LBFGS converges;
* every object handed back -- grid points, mean-processes, predictions -- de-normalised.

The scaling is what makes a *fixed* initial guess defensible: on unit-box inputs and unit-variance
outputs, a length scale of 0.2 and a variance of 1 are informative for any dataset.

This module is deliberately *not* re-exported by the package root: it is a convenience layer over the
flat API rather than part of it. Import it explicitly, `from mimosa.pipelines import
TrainTestPipeline`, as a reminder that anything it does not cover is `mimosa.BasicModel`'s job.
"""

from pathlib import Path

from jax import Array
from kernax import SEKernel, VarianceKernel, WhiteNoiseKernel, ZeroMean

from mimosa.data_structures import (
	Dataset,
	Dimensions,
	Grid,
	Hyperposterior,
	Mixture,
	ModelConfig,
	MultivariateNormal,
	Parameters,
)
from mimosa.grid import MergedGrid, RegularGrid, UnionGrid
from mimosa.io import load_csv
from mimosa.mappings import ExactInputMapper
from mimosa.mixture import MixtureInitialiser, MixtureUpdater
from mimosa.models import BasicModel
from mimosa.plot import plot_clusters, plot_dataset, plot_single_task_prediction
from mimosa.prediction import FunctionPredictor, Predictor
from mimosa.synthetic import build_parameters
from mimosa.utils import Scaler

__all__ = [
	"DEFAULT_MODEL_CONFIG",
	"DEFAULT_PARAMETERS",
	"TrainTestPipeline",
]


# Every hyperparameter is shared, tasks are heterotopic, and there is a single output. The model is
# single-output only, so `isotopic_output_in_*` must stay True (a 1-output grid cannot be
# heterotopic -- `validate_model_config` rejects it).
DEFAULT_MODEL_CONFIG = ModelConfig(
	shared_task_hps=True,
	shared_cluster_hps=True,
	shared_channel_hps=True,
	cluster_specific_task_hps=False,
	isotopic_tasks=False,
)

# Inputs live in [0, 1] and outputs have unit variance, so these are informative for any dataset: a
# length scale of 0.2 is a fifth of the observed input range. `fit` batches them to the data's
# dimensions with `build_parameters`.
DEFAULT_PARAMETERS = Parameters(
	cluster_mean=ZeroMean(),
	cluster_kernel=VarianceKernel(1.0) * SEKernel(length_scale=0.2),
	task_kernel=VarianceKernel(0.5) * SEKernel(length_scale=0.2),
	noise_kernel=WhiteNoiseKernel(noise=0.1),
)


class TrainTestPipeline:
	"""
	One-call multi-task GP clustering and prediction: load two datasets, fit, predict, plot.

	Every modelling choice is fixed to its most common setting (see the module docstring). Anything
	beyond that -- multi-output, per-task hyperparameters, a different kernel, a custom grid -- is
	what `mimosa.BasicModel` is for.

	Attributes
	----------
	train_data, test_data
		The datasets as the user provided them, in their original units.
	parameters
		Fitted hyperparameters. In *scaled* units: they describe the normalised problem, so a length
		scale here is a fraction of the training input range.
	mixture
		Fitted soft-clustering of the training tasks.
	test_mixture
		Responsibilities of the *test* tasks, re-derived at prediction time against the fitted
		mean-processes.
	hyperposterior
		Mean-processes over the prediction grid, de-normalised. Available after `predict`.
	predictions
		Per test task, per cluster, per channel predictive distribution over the prediction grid,
		de-normalised. Shape `(T_test, K, C, G)`. Available after `predict`.
	grid
		Prediction grid, de-normalised. The x-axis every returned distribution lives on.
	model
		The `mimosa.BasicModel` doing the work, assembled from the constructor's arguments.

	Examples
	--------
	>>> m = TrainTestPipeline(n_clusters=2, prng_key=jr.PRNGKey(0))  # doctest: +SKIP
	>>> m.load_train_data("train.csv").load_test_data("test.csv").fit()  # doctest: +SKIP
	>>> predictions = m.predict()  # doctest: +SKIP
	>>> m.plot(t_id=0)  # doctest: +SKIP
	"""

	def __init__(
		self,
		n_clusters: int,
		prng_key: Array,
		n_iter: int = 50,
		scaler: Scaler | None = None,
		parameters: Parameters = DEFAULT_PARAMETERS,
		model_config: ModelConfig = DEFAULT_MODEL_CONFIG,
		mixture_initialiser: MixtureInitialiser | None = None,
		mixture_updater: MixtureUpdater = MixtureUpdater(),
		predictor: Predictor = FunctionPredictor(),
	):
		"""
		Every argument past `n_iter` defaults to the setting the module docstring describes; they are
		here so one choice can be swapped without giving up the rest of the pipeline.

		Parameters
		----------
		n_clusters
			Number of mean-processes to cluster the tasks into.
		prng_key
			`jax.random` PRNG key, used to initialise the mixture.
		n_iter
			Number of EM iterations run by `fit`.
		scaler
			Scaling of both datasets. By default `load_train_data` fits one on the training data;
			pass a `Scaler` to reuse a scaling fitted elsewhere, and the training data leaves it
			untouched.
		parameters
			Initial hyperparameters, in *scaled* units and unbatched -- `fit` batches them to the
			data's dimensions. The default is informative on any dataset precisely because the
			inputs are on [0, 1] and the outputs have unit variance.
		model_config
			Which hyperparameters are shared across tasks, clusters and channels. Only meaningful to
			change alongside `parameters`, since the two have to agree on what is batched.
		mixture_initialiser
			Initialises the training tasks' responsibilities. Defaults to k-means over per-task
			summary statistics, seeded with `prng_key`.
		mixture_updater
			Updates responsibilities: at every E-step of `fit`, and once on the test tasks in
			`predict`.
		predictor
			Whether `predict` returns the latent function (`FunctionPredictor`, the default) or an
			observation of it, including observation noise (`ObservationPredictor`).
		"""
		self.n_clusters = n_clusters
		self.prng_key = prng_key
		self.n_iter = n_iter
		self.initial_parameters = parameters
		self.model_config = model_config
		self.mixture_updater = mixture_updater

		self.model = BasicModel(
			prng_key=prng_key,
			n_clusters=n_clusters,
			predictor=predictor,
			mixture_initialiser=mixture_initialiser,
			mixture_updater=mixture_updater,
		)

		self.train_data: Dataset | None = None
		self.test_data: Dataset | None = None
		self.scaler: Scaler | None = scaler
		self.parameters: Parameters | None = None
		self.mixture: Mixture | None = None
		self.test_mixture: Mixture | None = None
		self.hyperposterior: Hyperposterior | None = None
		self.predictions: MultivariateNormal | None = None
		self.grid: Grid | None = None

	# ----------------------------------------- loading ---------------------------------------- #

	@staticmethod
	def _as_dataset(source: str | Path | Dataset) -> Dataset:
		"""
		A Dataset passes through; a path is read with `mimosa.load_csv`.
		"""
		return source if isinstance(source, Dataset) else load_csv(source)

	def load_train_data(self, source: str | Path | Dataset) -> "TrainTestPipeline":
		"""
		Load the tasks to fit on, and fit the input/output scaling on them.

		Parameters
		----------
		source
			Path to a CSV in `mimosa.load_csv`'s format, or an already-built Dataset.

		Returns
		-------
		Self, so calls can be chained.
		"""
		self.train_data = self._as_dataset(source)
		if self.train_data.outputs.shape[-1] != 1 or self.train_data.output_ids is not None:
			raise ValueError(
				"TrainTestPipeline is single-output, single-channel. Use mimosa.BasicModel for anything else."
			)
		# A scaler given to the constructor was fitted on purpose; don't overwrite it.
		if self.scaler is None:
			self.scaler = Scaler.fit(self.train_data)
		return self

	def load_test_data(self, source: str | Path | Dataset) -> "TrainTestPipeline":
		"""
		Load the tasks to predict, scaled with the *training* statistics.

		Parameters
		----------
		source
			Path to a CSV in `mimosa.load_csv`'s format, or an already-built Dataset.

		Returns
		-------
		Self, so calls can be chained.
		"""
		if self.scaler is None:
			raise ValueError("Call load_train_data first: the scaling is fitted on the training data.")
		self.test_data = self._as_dataset(source)
		return self

	# ------------------------------------------ fitting --------------------------------------- #

	def _dimensions(self, dataset: Dataset, n_grid_points: int) -> Dimensions:
		"""
		Dimensions of a Dataset against a grid of `n_grid_points` points.
		"""
		T, N, C = dataset.outputs.shape
		return Dimensions(T=T, K=self.n_clusters, I=dataset.inputs.shape[-1], C=C, O=1, N=N, G=n_grid_points)

	def fit(self) -> "TrainTestPipeline":
		"""
		Fit the mean-processes, the hyperparameters and the training tasks' clustering.

		Fitting happens on the union of the training tasks' input points -- the grid the
		hyperposterior is exact on -- and in scaled units.

		Returns
		-------
		Self, so calls can be chained.
		"""
		if self.train_data is None:
			raise ValueError("Call load_train_data first.")

		train = self.scaler.scale(self.train_data)
		self._fit_grid = UnionGrid(train.inputs)
		self._dims = self._dimensions(train, len(self._fit_grid.points))

		initial = build_parameters(self.initial_parameters, self._dims, self.model_config)

		self._fit_hyperposterior, self.mixture, self.parameters = self.model.fit(
			train, self._fit_grid, initial, n_iter=self.n_iter
		)
		return self

	# ---------------------------------------- predicting -------------------------------------- #

	def predict(self, n_points: int = 200) -> MultivariateNormal:
		"""
		Predict every test task, on an evenly-spaced grid spanning the training input range.

		The prediction grid merges three point sets: the training tasks' inputs (so the
		hyperposterior keeps the information the fit found there), the test tasks' inputs (so each
		test task can be conditioned on its own observations), and `n_points` evenly-spaced points
		(so the returned curves are smooth). The result is then marginalised back onto the
		evenly-spaced points alone.

		The fitted hyperparameters are reused as they are: they are shared across tasks, so a test
		task is described by the same kernel a training task is, and nothing has to be re-optimised.
		What *is* re-derived is each test task's responsibilities, which cannot be known before
		seeing it.

		Parameters
		----------
		n_points
			Number of evenly-spaced points to predict on, per input dimension.

		Returns
		-------
		Predictive distribution of every test task under every mean-process, in the outputs'
		original units. Shape `(T_test, K, C, n_points)`. Each task's responsibilities towards the
		clusters are in `test_mixture`.
		"""
		if self.parameters is None:
			raise ValueError("Call fit first.")
		if self.test_data is None:
			raise ValueError("Call load_test_data first.")

		train, test = self.scaler.scale(self.train_data), self.scaler.scale(self.test_data)
		I = train.inputs.shape[-1]

		# `Dimensions` forbids a task holding more points than the grid, so the smooth grid must be
		# at least as dense as the busiest test task.
		n_points = max(n_points, test.outputs.shape[1])

		# Inputs were scaled onto [0, 1], so that is the range to predict over.
		regular = RegularGrid(train.inputs, bounds=((0.0, 1.0),) * I, n_points=n_points)
		merged = MergedGrid(UnionGrid(train.inputs), UnionGrid(test.inputs), regular)
		smooth = merged.sources[2]  # index of the evenly-spaced points in the merged pool

		# `merged` carries the *training* mappings (they come from the first source), which is what
		# the hyperposterior needs. Conditioning a test task additionally needs its own inputs
		# located in the same pool, so re-map them onto the shared points.
		test_grid = Grid(points=merged.points, mappings=ExactInputMapper()(merged.points, test.inputs))

		# Mean-processes over the whole pool, still conditioned on the training tasks only.
		hyperposterior = self.model.hyperpost(train, merged, self.mixture, self.parameters)

		# Each test task's responsibilities, from its own observations against those mean-processes.
		# The fitted mixture enters only through its proportions, which act as the prior.
		self.test_mixture = self.mixture_updater(
			test,
			test_grid,
			self.parameters.task_kernel + self.parameters.noise_kernel,
			hyperposterior,
			self.mixture,
		)

		predictions = self.model.predictor(test, test_grid, hyperposterior, self.parameters)

		self.grid = self.scaler.unscale_grid(regular)
		self.hyperposterior = self.scaler.unscale_distribution(hyperposterior.marginal(smooth))
		self.predictions = self.scaler.unscale_distribution(predictions.marginal(smooth))
		self._pred_dims = self._dimensions(test, len(regular.points))
		return self.predictions

	# ----------------------------------------- plotting --------------------------------------- #

	def plot(self, t_id: int | None = None, figsize: tuple[float, float] = (8, 6)):
		"""
		Plot the fit, or a single test task's prediction.

		Parameters
		----------
		t_id
			None (default) plots the training tasks coloured by their fitted cluster, with the
			mean-processes over them. An int plots that test task's observations, every
			mean-process, and the task's predictive mean and confidence interval under its most
			likely cluster -- which needs `predict` to have been called.
		figsize
			Size of the figure.

		Returns
		-------
		fig, ax
			The figure and its 2D array of axes.
		"""
		if self.mixture is None:
			raise ValueError("Call fit first.")

		if t_id is None:
			fig, ax = plot_dataset(self.train_data, self._dims, mixture=self.mixture, figsize=figsize, alpha=0.15)
			fig, ax = plot_clusters(
				self.scaler.unscale_grid(self._fit_grid),
				self._dims,
				hyperposterior=self.scaler.unscale_distribution(self._fit_hyperposterior),
				fig=fig,
				ax=ax,
			)
			fig.suptitle(f"Fitted mean-processes ({self.n_clusters} clusters)")
			return fig, ax

		if self.predictions is None:
			raise ValueError("Call predict first to plot a test task.")

		k_id = int(self.test_mixture.assignments[t_id])
		fig, ax = plot_single_task_prediction(
			self.test_data,
			self.grid,
			self._pred_dims,
			self.hyperposterior,
			self.test_mixture,
			t_id,
			c_id=0,
			prediction=self.predictions[t_id, k_id, 0],
			figsize=figsize,
		)
		fig.suptitle(f"Prediction — test task {t_id} (cluster {k_id})")
		return fig, ax
