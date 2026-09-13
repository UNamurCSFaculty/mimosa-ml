"""
Linear algebra and grid-indexing primitives used throughout the package: batched Cholesky solves,
and lexicographic (exact) or nearest-point (approximate) search over grid points.
"""

import numpy as np
import jax.numpy as jnp
import jax.lax as jlx
from jax import Array, jit, vmap
from jax.lax import fori_loop

from mimosa.constants import DEFAULT_JITTER, PAD_INDEX

__all__ = [
	"cho_factor",
	"cho_solve",
	"searchsorted_2d",
	"searchsorted_2d_vectorised",
	"lexicographic_sort",
	"find_exact_mappings",
	"sq_dists",
	"find_nearest_mappings",
	"mapping_distances",
]


def cho_factor(cov: Array, jitter: Array = DEFAULT_JITTER) -> Array:
	"""
	Cholesky factor of a covariance matrix, with jitter added to the diagonal for numerical stability.

	Unlike `jax.scipy.linalg.cho_factor`, does not symmetrise the input.

	Parameters
	----------
	cov
		Covariance matrix to factor. Shape `(..., N, N)`.
	jitter
		Amount of jitter added to the diagonal before factoring.

	Returns
	-------
	Lower Cholesky factor of `cov`. Shape `(..., N, N)`.
	"""
	return jlx.linalg.cholesky(cov + jitter * jnp.eye(cov.shape[-1]), symmetrize_input=False)


def cho_solve(cov_l: Array, res: Array, left_side: bool = True, lower: bool = True) -> Array:
	"""
	Solve `cov @ x = res` for x (or `x @ cov = res` if `left_side` is False), given the lower Cholesky
	factor `cov_l` of `cov` (`cov = cov_l @ cov_l^T`), as returned by `cho_factor`.

	Equivalent to `jax.scipy.linalg.cho_solve`, but uses lower factorisation by default and avoids its
	deprecation warning on batched 1D right-hand sides.

	Parameters
	----------
	cov_l
		Lower Cholesky factor of the covariance matrix. Shape `(..., N, N)`.
	res
		Right-hand side to solve for. Shape `(..., N, M)`.
	left_side
		If True, solve `cov @ x = res`; if False, solve `x @ cov = res`.
	lower
		Whether `cov_l` is lower- or upper-triangular.

	Returns
	-------
	x. Shape `(..., N, M)`.
	"""
	if left_side:
		y = jlx.linalg.triangular_solve(cov_l, res, left_side=True, lower=lower, transpose_a=False)
		return jlx.linalg.triangular_solve(cov_l, y, left_side=True, lower=lower, transpose_a=True)
	y = jlx.linalg.triangular_solve(cov_l, res, left_side=False, lower=lower, transpose_a=True)
	return jlx.linalg.triangular_solve(cov_l, y, left_side=False, lower=lower, transpose_a=False)


def searchsorted_2d(vector: Array, matrix: Array) -> Array:
	"""
	Find the index of `vector` in `matrix`, along axis 0.

	`matrix`'s rows must be sorted lexicographically (see `lexicographic_sort`), e.g.:

		[[1, 1, 0],
		 [1, 2, 1],
		 [1, 2, 2],
		 [2, 1, 3],
		 [2, 2, 1]]

	Uses a fixed number of bisection steps (`fori_loop`, from `matrix`'s static shape) so this is jit- and vmap-compatible.

	Parameters
	----------
	vector
		Vector to search for. If it contains NaN in any component, `PAD_INDEX` is returned
		unconditionally.
	matrix
		Matrix to search in.

	Returns
	-------
	Index of `vector` in `matrix`, or `PAD_INDEX` if not found.
	"""
	n = matrix.shape[0]
	steps = int(np.ceil(np.log2(n))) + 1  # static (n is a shape, always a Python int) -> fori_loop-safe

	def lex_lt(a, b):
		"""a < b, lexicographically; False if a == b."""
		differs = a != b
		i = jnp.argmax(differs)
		return jnp.where(jnp.any(differs), a[i] < b[i], False)

	def body(_, lo_hi):
		lo, hi = lo_hi
		mid = (lo + hi) // 2
		mid_lt_vector = lex_lt(matrix[mid], vector)
		return jnp.where(mid_lt_vector, mid + 1, lo), jnp.where(mid_lt_vector, hi, mid)

	lo, _ = fori_loop(0, steps, body, (0, n))

	safe_idx = jnp.minimum(lo, n - 1)
	found = (lo < n) & jnp.all(matrix[safe_idx] == vector)
	return jnp.where(found, lo, PAD_INDEX)


searchsorted_2d_vectorised = jit(vmap(searchsorted_2d, in_axes=(0, None)))


def lexicographic_sort(arr: Array) -> Array:
	"""
	Sort a 2D array lexicographically along its first dimension.

	Parameters
	----------
	arr
		2D array to sort.

	Returns
	-------
	`arr`, with rows sorted lexicographically.
	"""
	return arr[jnp.lexsort(arr.T[::-1])]


def find_exact_mappings(grid: Array, points: Array) -> Array:
	"""
	Find the indices of `points` in `grid`.

	Parameters
	----------
	grid
		Sorted grid points, of shape `(FG, I)`. If 2D, rows must be sorted lexicographically
		(see `lexicographic_sort`).
	points
		points to search for, of shape `(FN, I)`.

	Returns
	-------
	Index of each of `points` in `grid`, or `PAD_INDEX` for a point that is not a grid point (NaN
	included).
	"""
	if grid.shape[-1] == 1:
		# We only have 1 input dimension, and we can use the fast jnp.searchsorted function
		grid, points = grid.squeeze(axis=-1), points.squeeze(axis=-1)
		idx = jnp.searchsorted(grid, points)  # insertion point, so the hit must still be verified
		return jnp.where(grid[jnp.minimum(idx, len(grid) - 1)] == points, idx, PAD_INDEX)
	# Multiple input dimensions requires our custom lexicographic search
	return searchsorted_2d_vectorised(points, grid)


def sq_dists(points: Array, centers: Array) -> Array:
	"""
	Pairwise squared Euclidean distances between `points` and `centers`.

	Parameters
	----------
	points
		Points, of shape `(N, I)`.
	centers
		Centers, of shape `(FG, I)`.

	Returns
	-------
	Squared distances, of shape `(N, FG)`.
	"""
	return jnp.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=-1)


def find_nearest_mappings(grid: Array, points: Array, chunk_size: None | int = None) -> Array:
	"""
	Find the indices in `grid` of the grid points nearest to `points`, in Euclidean distance.

	Approximate counterpart of `find_exact_mappings`: every non-NaN point gets a grid point, however
	far it is. `grid` need not be sorted, ties go to the lowest index.

	Chunked over `points`, so at most `(chunk_size, FG)` distances exist at once and peak memory is
	set by `chunk_size` rather than by `FN`.

	Parameters
	----------
	grid
		Grid points, of shape `(FG, I)`.
	points
		Points to search for, of shape `(FN, I)`.
	chunk_size
		Rows of `points` handled per `jax.lax.map` step, each holding a `(chunk_size, FG)` distance
		matrix. Defaults to all of `points` at once; lower it if that does not fit in memory. Must be
		a static Python int, never a traced value.

	Returns
	-------
	Index in `grid` of the point nearest each of `points`, or `PAD_INDEX` for a NaN point.
	"""
	chunk_size = max(len(points), 1) if chunk_size is None else chunk_size
	n_chunks = -(-len(points) // chunk_size)  # ceil division
	padded = jnp.pad(points, ((0, n_chunks * chunk_size - len(points)), (0, 0)))
	chunks = padded.reshape(n_chunks, chunk_size, points.shape[-1])
	nearest = jlx.map(lambda chunk: jnp.argmin(sq_dists(chunk, grid), axis=-1), chunks)
	# `argmin` is per row, so a NaN point only corrupts its own index: overwriting it here is enough.
	return jnp.where(jnp.any(jnp.isnan(points), axis=-1), PAD_INDEX, nearest.reshape(-1)[: len(points)])


def mapping_distances(grid: Array, points: Array, mappings: Array) -> Array:
	"""
	Euclidean distance from each of `points` to the grid point its mapping selects, i.e. the error
	an approximate mapping such as `mimosa.mappings.NearestInputMapper` introduces.

	Parameters
	----------
	grid
		Grid points, of shape `(FG, I)`.
	points
		Points measured against `grid`, of shape `(..., I)`.
	mappings
		Index of each of `points` in `grid`, of shape `(...)`, as returned by an
		`mimosa.mappings.InputMapper`.

	Returns
	-------
	Distances, of shape `(...)`, NaN where `mappings` is `PAD_INDEX` and there is thus no grid point
	to measure against.
	"""
	# PAD_INDEX is out of bounds, so the gather clamps it onto the NaN sentinel appended here.
	grid = jnp.concatenate([grid, jnp.full((1, grid.shape[-1]), jnp.nan, grid.dtype)])
	return jnp.sqrt(jnp.sum((points - grid[mappings]) ** 2, axis=-1))
