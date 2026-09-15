"""
Pre- and post-processing helpers that sit around a model rather than inside it.

`Scaler` normalises a Dataset -- inputs onto the unit box, outputs to zero mean and unit variance --
and maps everything the model hands back (grids, hyperposteriors, predictions, whole datasets) into
the units the data came in with. The normalisation is what makes a *fixed* initial hyperparameter
guess defensible: on unit-box inputs and unit-variance outputs, a length scale of 0.2 and a variance
of 1 are informative for any dataset.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from mimosa.data_structures import Dataset, Grid, MultivariateNormal

__all__ = [
	"Scaler",
]


@dataclass(frozen=True)
class Scaler:
	"""
	Min-max scaling of the inputs onto [0, 1] and standardisation of the outputs, both fitted on the
	training Dataset and applied unchanged to the test one.

	A test point outside the training range simply lands outside [0, 1]; it is not clipped, since
	clipping would move an input the model is asked to predict at.

	Attributes
	----------
	lo, hi
		Per-input-dimension minimum and maximum of the training inputs. Shape `(I,)`.
	mu, sd
		Per-channel mean and standard deviation of the training outputs. Shape `(C,)`.

	Examples
	--------
	>>> scaler = Scaler.fit(train_data)  # doctest: +SKIP
	>>> train, test = scaler.scale(train_data), scaler.scale(test_data)  # doctest: +SKIP
	>>> scaler.unscale_dataset(train)  # back in the original units  # doctest: +SKIP
	"""

	lo: Array
	hi: Array
	mu: Array
	sd: Array

	@classmethod
	def fit(cls, dataset: Dataset) -> "Scaler":
		"""
		Read the scaling statistics off a training Dataset, ignoring NaN (padding and missing values).
		"""
		inputs, outputs = dataset.inputs, dataset.outputs
		lo, hi = jnp.nanmin(inputs, axis=(0, 1)), jnp.nanmax(inputs, axis=(0, 1))
		mu, sd = jnp.nanmean(outputs, axis=(0, 1)), jnp.nanstd(outputs, axis=(0, 1))
		# A constant dimension/channel has zero spread; keep it at its offset rather than dividing by 0.
		return cls(lo=lo, hi=jnp.where(hi > lo, hi, lo + 1.0), mu=mu, sd=jnp.where(sd > 0, sd, 1.0))

	def scale(self, dataset: Dataset) -> Dataset:
		"""
		Scale a Dataset's inputs and outputs. NaN padding and missing values stay NaN.
		"""
		return Dataset(
			inputs=(dataset.inputs - self.lo) / (self.hi - self.lo),
			outputs=(dataset.outputs - self.mu) / self.sd,
			known_output_noise=None if dataset.known_output_noise is None else dataset.known_output_noise / self.sd**2,
			output_ids=dataset.output_ids,
		)

	def unscale_dataset(self, dataset: Dataset) -> Dataset:
		"""
		The exact inverse of `scale`: a scaled Dataset back in the units it was fitted from.

		Noise is a variance, so it scales by `sd**2` where the outputs scale by `sd`. `output_ids`
		are output labels, not values, and pass through untouched.
		"""
		return Dataset(
			inputs=dataset.inputs * (self.hi - self.lo) + self.lo,
			outputs=dataset.outputs * self.sd + self.mu,
			known_output_noise=None if dataset.known_output_noise is None else dataset.known_output_noise * self.sd**2,
			output_ids=dataset.output_ids,
		)

	def unscale_points(self, points: Array) -> Array:
		"""
		Map grid points, shape `(G, I)`, back to the inputs' original units.
		"""
		return points * (self.hi - self.lo) + self.lo

	def unscale_grid(self, grid: Grid) -> Grid:
		"""
		The same Grid with its points back in the inputs' original units. Mappings are indices into
		`points`, so they survive the rescaling untouched.
		"""
		return Grid(
			points=self.unscale_points(grid.points),
			output_ids=grid.output_ids,
			mappings=grid.mappings,
			n_outputs=grid.n_outputs,
		)

	def unscale_distribution(self, distribution: MultivariateNormal) -> MultivariateNormal:
		"""
		Map a distribution over outputs -- a hyperposterior or a prediction -- back to the outputs'
		original units.

		Standardisation is affine, so the normal stays normal: the mean shifts and scales, the
		covariance scales by `sd**2`. The channel axis is second-to-last on the mean (`..., C, P`)
		and third-to-last on the covariance (`..., C, P, P`), so `sd` is broadcast accordingly. The
		subclass is preserved, keeping a `Hyperposterior` a `Hyperposterior`.
		"""
		return type(distribution)(
			mean=distribution.mean * self.sd[:, None] + self.mu[:, None],
			covariance=distribution.covariance * (self.sd**2)[:, None, None],
		)
