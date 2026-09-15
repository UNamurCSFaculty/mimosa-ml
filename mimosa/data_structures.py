"""
Data structures used throughout the package: dimensions and configuration, model parameters and priors,
datasets and grids, and prediction outputs.

Structures are plain dataclasses, or `equinox.Module` when they must be passed to jitted functions.
"""

from dataclasses import dataclass, fields
from typing import Literal
from jaxtyping import Array, Float, Int, jaxtyped
import jax.numpy as jnp
import jax.tree_util as jtu
from beartype import beartype as typechecker
import equinox as eqx
from kernax import MeanLike, KernelLike

from mimosa.mappings import ExactInputMapper, InputMapper

__all__ = [
	"IdArg",
	"Dimensions",
	"ModelConfig",
	"DataRemovalConfig",
	"validate_model_config",
	"Parameters",
	"GPParameters",
	"ParameterPriors",
	"Dataset",
	"GPDataset",
	"Grid",
	"MultivariateNormal",
	"Hyperprior",
	"Hyperposterior",
	"Mixture",
	"PredictionMeanBlocks",
	"PredictionCovBlocks",
]


@dataclass(frozen=True)
class Dimensions:
	"""
	Dimensions of the data that the model will generate, learn, or predict.

	Attributes
	----------
	T
		Number of tasks.
	K
		Number of clusters in the mixture.
	I
		Dimensionality of input points.
	C
		Number of channels, aka dimensionality of output points.
	O
		Number of correlated outputs.
	N
		Number of points observed by the largest task.
	G
		Number of points in the full grid.

	Raises
	------
	ValueError
		If any dimension is non-positive, if there are more clusters than tasks,
		or more points per task than in the grid.
	"""

	T: int
	K: int
	I: int
	C: int
	O: int
	N: int
	G: int

	def __post_init__(self):
		# Check positivity
		def all_positive():
			return all(getattr(self, field.name) > 0 for field in fields(self))

		if not all_positive():
			raise ValueError("All dimensions must be positive integers.")

		# Check for contradicting dimensions
		if self.K > self.T:
			raise ValueError("Cannot have more clusters than tasks.")
		if self.N > self.G:
			raise ValueError("Cannot have more points in tasks than in the full grid")


@dataclass(frozen=True)
class ModelConfig:
	"""
	Configuration of the model, used to know which parameters are batched.

	Attributes
	----------
	shared_task_hps
		If True, task-kernel hyperparameters are shared across tasks; if False, each task has its own.
	shared_cluster_hps
		If True, cluster-kernel hyperparameters are shared across mean-processes; if False, each mean-process has its own.
	shared_channel_hps
		If True, hyperparameters are shared across channel dimensions; if False, each channel has its own.
	cluster_specific_task_hps
		If True, task-kernel hyperparameters may additionally vary by cluster assignment, independently of `shared_task_hps`.
	isotopic_tasks
		If True, all tasks share the same input locations.
	isotopic_output_in_tasks
		If True, all outputs in tasks share the same input locations.
	isotopic_output_in_grid
		If True, all outputs in the grid share the same input locations.
	"""

	shared_task_hps: bool = True
	shared_cluster_hps: bool = True
	shared_channel_hps: bool = True
	cluster_specific_task_hps: bool = True
	isotopic_tasks: bool = True
	isotopic_output_in_tasks: bool = True
	isotopic_output_in_grid: bool = True


@dataclass(frozen=True)
class DataRemovalConfig:
	"""
	Configuration for the random removal of data points in a Dataset.

	Attributes
	----------
	max_missing
		Maximum number of missing points per task, per output (and per channel, when missingness is
		not shared across channels).
	random_missing_count
		If True, the number of missing points is drawn randomly in [0, `max_missing`]; if False, it is fixed to `max_missing`.
	same_missing_across_channels
		If True, a missing point is missing for every channel; if False, missingness is drawn independently per channel.
	same_missing_across_outputs
		If True, a missing point is missing for every output; if False, missingness is drawn independently per output.
		True is only meaningful when outputs share input locations (`ModelConfig.isotopic_output_in_tasks`);
		`RandomDataRemover` raises otherwise.
	"""

	max_missing: int
	random_missing_count: bool = False
	same_missing_across_channels: bool = True
	same_missing_across_outputs: bool = False


def validate_model_config(model_config: ModelConfig, dimensions: Dimensions) -> None:
	"""
	Check that the model configuration is compatible with the dimensions.

	Parameters
	----------
	model_config
		Configuration to validate.
	dimensions
		Dimensions to validate the configuration against.

	Raises
	------
	ValueError
		If a hyperparameter is configured to vary along an axis of size 1.
	"""
	if not model_config.shared_task_hps and dimensions.T == 1:
		raise ValueError("Cannot have distinct task hyperparameters with only one task.")
	if not model_config.shared_cluster_hps and dimensions.K == 1:
		raise ValueError("Cannot have distinct cluster hyperparameters with only one cluster.")
	if not model_config.shared_channel_hps and dimensions.C == 1:
		raise ValueError("Cannot have distinct channel hyperparameters with only one channel.")
	if (not model_config.isotopic_output_in_grid or not model_config.isotopic_output_in_tasks) and dimensions.O == 1:
		raise ValueError("Cannot have multi-output grids/task inputs with only one output.")
	if model_config.isotopic_output_in_tasks and not model_config.isotopic_output_in_grid:
		raise ValueError("Cannot have isotopic output in tasks with inputs sampled from an heterotopic grid")


IdArg = int | Literal["all"]


def _resolve_ids(id_arg: IdArg, size: int) -> list[int]:
	"""
	Resolve a t_id/k_id/c_id/o_id argument into a list of indices: every index if "all",
	or a single-element list if an int.
	"""
	if id_arg == "all":
		return list(range(size))
	if isinstance(id_arg, int):
		if not (0 <= id_arg < size):
			raise ValueError(f"Index {id_arg} out of range for size {size}.")
		return [id_arg]
	raise TypeError(f"Expected 'all' or int, got {id_arg!r}.")


class Parameters(eqx.Module):
	"""
	Cluster mean, cluster kernel, task kernel and noise kernel of the model.

	Attributes
	----------
	cluster_mean
		Mean function of the mean-processes.
	cluster_kernel
		Covariance kernel of the mean-processes.
	task_kernel
		Covariance kernel modelling the deviation of each task from its mean-process.
	noise_kernel
		Covariance kernel modelling the observation noise.
	"""

	cluster_mean: MeanLike
	cluster_kernel: KernelLike
	task_kernel: KernelLike
	noise_kernel: KernelLike


class GPParameters(eqx.Module):
	"""
	Prior mean, kernel and noise kernel of a single Gaussian process, used by `mimosa.models.GPModel`.

	Every field is batched over the channel axis. Build them through `mimosa.synthetic.build_gp_parameters`.

	Attributes
	----------
	mean
		Prior mean function of the Gaussian process.
	kernel
		Prior covariance kernel of the Gaussian process.
	noise_kernel
		Covariance kernel modelling the observation noise.
	"""

	mean: MeanLike
	kernel: KernelLike
	noise_kernel: KernelLike


@dataclass(frozen=True)
class ParameterPriors:
	"""
	Priors for each parameter of the model, used to generate the hyperparameters or to perform maximum-a-posteriori
	inference.

	Attributes
	----------
	cluster_mean_priors
		Priors for the mean function's hyperparameters.
	cluster_kernel_priors
		Priors for the mean-process kernel's hyperparameters.
	task_kernel_priors
		Priors for the task kernel's hyperparameters.
	noise_kernel_priors
		Priors for the noise kernel's hyperparameters.
	"""

	cluster_mean_priors: dict
	cluster_kernel_priors: dict
	task_kernel_priors: dict
	noise_kernel_priors: dict


@jaxtyped(typechecker=typechecker)
class Dataset(eqx.Module):
	"""
	Dataset regrouping the inputs and outputs of all tasks.

	Attributes
	----------
	inputs
		Input points of each task. Shape is `(T, N, I)` when outputs share the same input locations
		(isotopic outputs), or `(T, O*N, I)` otherwise.
	outputs
		Output values of each task, at each output and input point.
	known_output_noise
		Known observation noise, when available.
	output_ids
		IDs of the output of each input point. None if isotopic outputs. Else, shape `(1, O*N)` if
		every task shares the same output ids (in particular, whenever `isotopic_tasks`), or
		`(T, O*N)` if they vary per task.
	"""

	inputs: Float[Array, "#T oN I"]  # "o" is 1 if isotopic_output_in_tasks and dims.O otherwise
	outputs: Float[Array, "T ON C"]
	known_output_noise: None | Float[Array, "T ON C"] = None
	output_ids: None | Int[Array, "#T oN"] = None

	def __getitem__(self, item):
		"""
		Index `inputs`, `outputs`, `known_output_noise` and `output_ids` jointly along the batch dimensions.
		They keep the same number of dimensions even if item is a specific id. Their first axis simply
		has length 1.
		"""
		inputs = self.inputs[0 if self.inputs.shape[0] == 1 else item]
		outputs = self.outputs[item]
		known_output_noise = None if self.known_output_noise is None else self.known_output_noise[item]
		output_ids = None if self.output_ids is None else self.output_ids[0 if self.output_ids.shape[0] == 1 else item]

		if isinstance(item, slice | list | tuple):
			return Dataset(
				inputs=inputs,
				outputs=outputs,
				known_output_noise=known_output_noise,
				output_ids=output_ids,
			)
		else:
			return Dataset(
				inputs=inputs[None, ...],
				outputs=outputs[None, ...],
				known_output_noise=None if known_output_noise is None else known_output_noise[None, ...],
				output_ids=None if output_ids is None else output_ids[None, ...],
			)

	@property
	def clean_inputs(self) -> Float[Array, "#T oN I"]:
		"""
		`inputs` with NaN padding replaced by 0. Use this, not `inputs`, to feed a kernel.

		Padding is masked downstream from `outputs`, so the value itself is never read -- but it must
		be finite: `jnp.where` does not stop NaN in the VJP (`0 * NaN = NaN`), so a single padded
		point turns every kernel hyperparameter's gradient into NaN and freezes the optimiser.

		Never pass this to a grid builder: 0 is a real input location and would add a spurious grid
		point.
		"""
		return jnp.nan_to_num(self.inputs)


@jaxtyped(typechecker=typechecker)
class GPDataset(eqx.Module):
	"""
	Observations of a single Gaussian process, over `O` correlated outputs and `C` channels, used by
	`mimosa.models.GPModel`.

	Contrary to Dataset, `inputs` can't have NaN padding -- a NaN there poisons every hyperparameter's gradient
	(see `Dataset.clean_inputs`); drop an unobserved row instead. A NaN in `outputs` is still read as
	a missing value, so an output or a channel may be missing at a kept input point.

	Attributes
	----------
	inputs
		Input points. Shape `(N, I)` when the outputs share their input locations -- a multi-output
		kernel expands them into `O` blocks itself -- and `(O*N, I)` otherwise, in which case
		`output_ids` labels each row. A single-output GP is `O = 1` with `output_ids=None`.
	outputs
		Observed values, at each output and input point.
	output_ids
		IDs of the output of each input point. None if the outputs share their input locations, else
		shape `(O*N,)`.
	"""

	inputs: Float[Array, "oN I"]  # "o" is 1 if the outputs share their input locations, and O otherwise
	outputs: Float[Array, "ON C"]
	output_ids: None | Int[Array, "oN"] = None


@jaxtyped(typechecker=typechecker)
class Grid(eqx.Module):
	"""
	Grid points and mappings of each task's inputs on the grid.

	Attributes
	----------
	points
		Input points of the grid.
	output_ids
		IDs of the output of each grid point. None if isotopic outputs. Else, shape `(O*G,)`. Defaults to `None`.
	mappings
		Index of each task's input points in `points`. Defaults to `None`.
	n_outputs
		Number of correlated outputs the grid spans. When `output_ids` is None they share `points`,
		so a distribution over the grid is `n_outputs * len(points)` long; otherwise `points`
		already holds one block per output. Defaults to `1`, a single-output grid.
	"""

	points: Float[Array, "FG I"]
	output_ids: None | Int[Array, "FG"] = None
	mappings: None | Int[Array, "#T N"] = None
	n_outputs: int = eqx.field(static=True, default=1)

	def remap(self, inputs: Float[Array, "#T N I"], input_mapper: InputMapper = ExactInputMapper()) -> "Grid":
		"""
		The same grid points, with `inputs` mapped onto them instead of this grid's own inputs.

		Parameters
		----------
		inputs
			Input points of every task.
		input_mapper
			Maps `inputs` onto `points`. Defaults to `ExactInputMapper`.

		Returns
		-------
		Grid with the same points, carrying `inputs`' mappings.
		"""
		return Grid(
			points=self.points,
			output_ids=self.output_ids,
			mappings=input_mapper(self.points, inputs),
			n_outputs=self.n_outputs,
		)


@jaxtyped(typechecker=typechecker)
class MultivariateNormal(eqx.Module):
	"""
	Multivariate normal distribution over a P-dimensional vector, possibly batched.

	Attributes
	----------
	mean
		Mean vector.
	covariance
		Covariance matrix.
	"""

	mean: Float[Array, "... P"]
	covariance: Float[Array, "... P P"]

	def __getitem__(self, item):
		"""
		Index `mean` and `covariance` jointly along the batch dimensions.
		"""
		return MultivariateNormal(mean=self.mean[item], covariance=self.covariance[item])

	def marginal(self, indices) -> "MultivariateNormal":
		"""
		Marginal distribution over a subset of the P dimensions, indexing `mean` and `covariance`
		jointly along the points (last) axis.

		Parameters
		----------
		indices
			Selector along the event axis: a `slice`, a 1-D integer array, or a 1-D boolean mask.

			An out-of-bounds entry -- `mimosa.PAD_INDEX`, marking an input point with no grid point
			-- is clamped by the gather.

		Returns
		-------
		Distribution over the selected dimensions. A subclass marginalises to a plain
		`MultivariateNormal`: the result is no longer over the grid points.
		"""
		return MultivariateNormal(
			mean=self.mean[..., indices],
			covariance=self.covariance[..., indices, :][..., :, indices],
		)

	def cross_covariance(self, indices) -> Float[Array, "... Q P"]:
		"""
		Covariance between the dimensions selected by `indices` (see `marginal`) and all P dimensions.
		"""
		return self.covariance[..., indices, :]


@jaxtyped(typechecker=typechecker)
class Hyperprior(MultivariateNormal):
	"""
	Prior distribution over the mean-process values at grid points, before observing data.
	"""

	mean: Float[Array, "*B FG"]
	covariance: Float[Array, "*B FG FG"]


@jaxtyped(typechecker=typechecker)
class Hyperposterior(MultivariateNormal):
	"""
	Posterior distribution over the mean-process values at grid points, after observing data.
	"""

	mean: Float[Array, "*B FG"]
	covariance: Float[Array, "*B FG FG"]


@jaxtyped(typechecker=typechecker)
class Mixture(eqx.Module):
	"""
	Soft-clustering of the tasks into mean-processes.

	Attributes
	----------
	responsibilities
		Probability of each task belonging to each mean-process.
	"""

	responsibilities: Float[Array, "T K"]

	@property
	def proportions(self) -> Float[Array, "K"]:
		"""
		Mixture weight of each mean-process, i.e. the mean responsibility towards it.

		This is the maximiser of the ELBO in the mixture weights, so it is derived from
		`responsibilities` rather than stored: the two can never disagree.
		"""
		return jnp.mean(self.responsibilities, axis=0)

	@property
	def assignments(self) -> Float[Array, "T"]:
		"""
		Hard cluster assignment of each task, i.e. its most likely mean-process.
		"""
		return jnp.argmax(self.responsibilities, axis=1)


@jaxtyped(typechecker=typechecker)
class PredictionMeanBlocks(eqx.Module):
	"""
	Predicted mean, split into blocks for observed points and grid points.

	Attributes
	----------
	mean_obs
		Predicted mean at the observed input points.
	mean_grid
		Predicted mean at the grid points.
	"""

	mean_obs: Float[Array, "*B FN"]
	mean_grid: Float[Array, "*B FG"]

	def __getitem__(self, item):
		"""
		Index `mean_obs` and `mean_grid` jointly along the batch dimensions.
		"""
		return PredictionMeanBlocks(mean_obs=self.mean_obs[item], mean_grid=self.mean_grid[item])


@jaxtyped(typechecker=typechecker)
class PredictionCovBlocks(eqx.Module):
	"""
	Predicted covariance, split into blocks for observed points, grid points, and their cross-covariance.

	Attributes
	----------
	cov_obs
		Covariance among the observed input points.
	cov_grid
		Covariance among the grid points.
	cov_crossed
		Cross-covariance between the observed input points and the grid points.

	Notes
	-----
	The batch dimensions are broadcastable (`#*B`) rather than identical: only `cov_obs` includes the
	noise kernel, so a noise carrying its own values per task or per channel -- e.g.
	`mimosa.synthetic.known_noise_kernel` -- batches `cov_obs` along an axis the two grid blocks are
	legitimately shared along. Mismatched sizes are still rejected.
	"""

	cov_obs: Float[Array, "#*B FN FN"]
	cov_grid: Float[Array, "#*B FG FG"]
	cov_crossed: Float[Array, "#*B FN FG"]

	def __getitem__(self, item):
		"""
		Index `cov_obs`, `cov_grid` and `cov_crossed` jointly along the batch dimensions.
		"""
		return PredictionCovBlocks(
			cov_obs=self.cov_obs[item], cov_grid=self.cov_grid[item], cov_crossed=self.cov_crossed[item]
		)

	@property
	def over_tasks(self) -> tuple["PredictionCovBlocks", "PredictionCovBlocks"]:
		"""
		This, and the `in_axes` mapping it over the leading (task) axis.

		A block whose leading axis is 1 is shared across tasks: it is squeezed and marked `None`, so
		the vmap reads the single copy instead of slicing it. Blocks are handled one by one, since
		they need not be shared along the same axes.

		Returns
		-------
		blocks, in_axes
			`blocks` with every shared block squeezed, and a matching pytree of `0`/`None` to pass as
			`jax.vmap`'s `in_axes`. Both are built by `tree_unflatten`, which skips the jaxtyped
			`__init__` -- `in_axes` holds ints and None, not Arrays.
		"""
		blocks, axes = [], []
		for block in jtu.tree_leaves(self):
			shared = block.shape[0] == 1
			blocks.append(block[0] if shared else block)
			axes.append(None if shared else 0)

		structure = jtu.tree_structure(self)
		return jtu.tree_unflatten(structure, blocks), jtu.tree_unflatten(structure, axes)
