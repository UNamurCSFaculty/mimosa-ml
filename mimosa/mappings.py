"""
Map the input points of every task onto the points of a grid.

A mapping is the bridge between a `Dataset`'s inputs and a `Grid`: it gives, for each task input
point, the index of the grid point that represents it. `mimosa.grid`'s grids delegate to an
`InputMapper` to produce their own `mappings`.

Padding
-------
A task input point that does not resolve to a grid point -- a NaN point padding a variable-length
task, or a point simply absent from the grid -- maps to `PAD_INDEX`. As it is out of bounds, it is dropped
by a scatter (`mimosa.hyperpost`) and clamped by a gather (`mimosa.prediction`, `mimosa.nll`).

To avoid overflow errors, it is recommended not to perform arithmetics (mainly addition) on mappings.
"""

from abc import abstractmethod
from jax import Array, vmap
import equinox as eqx

from mimosa.linalg import find_exact_mappings, find_nearest_mappings

__all__ = ["InputMapper", "ExactInputMapper", "NearestInputMapper"]


class InputMapper(eqx.Module):
	"""
	Base class for mapping the input points of every task onto grid points.
	"""

	@abstractmethod
	def __call__(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
		"""
		Map each of `inputs`' points to its index in `points`.

		Parameters
		----------
		points
			Grid points to map `inputs` onto, of shape `(G, I)`.
		inputs
			Input points of every task, of shape `(#T, N, I)`. A padding point (NaN on every input
			dimension) maps to `PAD_INDEX`.

		Returns
		-------
		Index of each of `inputs`' points in `points`, of shape `(#T, N)`.
		"""
		...


class ExactInputMapper(InputMapper):
	"""
	Map every input point to the grid point it is exactly equal to, by binary search.

	Default mapper of `mimosa.grid`'s grids. Suited to a grid that contains the tasks' own input
	points, such as one built by `mimosa.grid.UnionGrid`: a point that is not bit-for-bit equal to a
	grid point maps to `PAD_INDEX`, and is thus ignored during training/prediction.

	jit- and vmap-compatible.
	"""

	def __call__(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
		"""
		See `InputMapper.__call__`.
		"""
		return vmap(lambda task_inputs: find_exact_mappings(points, task_inputs))(inputs)


class NearestInputMapper(InputMapper):
	"""
	Map every input point to the grid point nearest to it, in Euclidean distance.

	Unlike `ExactInputMapper`, fits a grid whose points do not coincide with the tasks' own input
	points, such as one built by `mimosa.grid.RegularGrid` or `mimosa.grid.KMeansGrid`: no point is
	dropped, at the cost of representing it by a merely nearby grid point.
	`mimosa.linalg.mapping_distances` measures that approximation error.

	jit- and vmap-compatible.

	Attributes
	----------
	chunk_size
		Input points handled per distance-computation step, each holding a `(chunk_size, G)`
		distance matrix. Defaults to every input point at once; lower it if that does not fit in
		memory.
	"""

	chunk_size: None | int = eqx.field(static=True, default=None)

	def __call__(self, points: Array, inputs: Array, *args, **kwargs) -> Array:
		"""
		See `InputMapper.__call__`.
		"""
		mappings = find_nearest_mappings(points, inputs.reshape(-1, inputs.shape[-1]), self.chunk_size)
		return mappings.reshape(inputs.shape[:-1])
