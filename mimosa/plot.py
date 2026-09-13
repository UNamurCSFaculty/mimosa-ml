"""
Plotting utilities for mimosa-ml, using matplotlib with a seaborn-style theme.

Composable plotting functions for Dataset points (`plot_channel`/`plot_task`/`plot_dataset`),
mean-processes (`plot_single_cluster_single_channel`/`plot_single_cluster`/`plot_clusters`), and
per-task predictions (`plot_single_task_prediction`).

Convention
----------
Subplots: columns = channels (`C`), rows = outputs (`O`). Every plotting function takes the
dataset's `Dimensions` explicitly (`dims`), used to resolve "all" selectors and to slice the
`O*N`/`O*G` block-major arrays (`Dataset.outputs`, `Grid.points`, `Hyperprior`/`Hyperposterior`
mean/covariance, `prediction`/`samples`) down to a single output's block -- see
`mimosa.synthetic.sample_inputs`/`generate_grid` and kernax's multi-output Mean/Kernel classes for
the block-major convention itself. Every plotting function also accepts optional `fig`/`ax` so
plots can be composed into larger figures (see `_get_fig_ax`). Selector arguments (`t_id`, `k_id`,
`c_id`, `o_id`) pick a single task/cluster/channel/output index (int), or "all" (default) to plot
every one.

Limitations
-----------
Only 1D inputs (`I == 1`) are supported.
"""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from mimosa.data_structures import Dataset, Dimensions, Grid, Hyperposterior, Hyperprior, Mixture, MultivariateNormal

__all__ = [
	"IdArg",
	"plot_channel",
	"plot_task",
	"plot_dataset",
	"plot_single_cluster_single_channel",
	"plot_single_cluster",
	"plot_clusters",
	"plot_single_task_prediction",
]

_STYLE = "seaborn-v0_8-whitegrid"


IdArg = int | Literal["all"]

_DEFAULT_SCATTER_KWARGS = {"s": 15, "alpha": 0.7}
_DEFAULT_LINE_KWARGS = {"linewidth": 1.0, "alpha": 0.7}


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


def _get_fig_ax(fig, ax, nrows: int, ncols: int, figsize: tuple[float, float] | None = None):
	"""
	Get or create a (fig, ax) pair with an `nrows` x `ncols` grid of axes, returning `ax` as a 2D
	array regardless of grid size. Reuses `fig`/`ax` if given, so plots can be composed together.

	A figure created here is styled with `_STYLE`, applied through `plt.style.context` so the
	caller's global rcParams are left untouched. One passed in through `fig`/`ax` keeps whatever
	style it was created with.

	Either way, the figure's layout engine is set to "constrained": unlike a one-shot
	`fig.tight_layout()` call, it keeps titles, axis labels and figure-level legends from
	overlapping (or from being clipped at the figure's edge) even as more elements are added later
	by composing further plots onto the same `fig`/`ax`, or by the caller (e.g. `fig.suptitle`).
	"""
	if ax is not None:
		ax_arr = np.atleast_2d(ax)
		if fig is None:
			fig = ax_arr.flat[0].figure
		fig.set_layout_engine("constrained")
		return fig, ax_arr

	if figsize is None:
		figsize = (4 * ncols, 3 * nrows)
	with plt.style.context(_STYLE):
		fig, ax = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, layout="constrained")
	return fig, ax


def _output_rows(output_ids, o_id: int, block_size: int) -> np.ndarray | slice:
	"""
	Row indices belonging to output `o_id` in an `output_ids`-labelled array (if given), or the
	`o_id`-th of `dims.O` contiguous blocks of `block_size` rows (output-major), if `output_ids`
	is `None` -- the convention used throughout `mimosa.synthetic` (`sample_inputs`/`generate_grid`)
	and by kernax's multi-output Mean/Kernel classes when called without their own `output_ids`.
	"""
	if output_ids is not None:
		return np.flatnonzero(np.asarray(output_ids) == o_id)
	return slice(o_id * block_size, (o_id + 1) * block_size)


def _task_xy(dataset: Dataset, dims: Dimensions, t_id: int, c_id: int, o_id: int):
	"""
	Extract a single task's observed `(x, y)` points for a single channel and output, as numpy
	arrays with missing (NaN) points already dropped.
	"""
	t = 0 if dataset.inputs.shape[0] == 1 else t_id
	output_ids = dataset.output_ids[t] if dataset.output_ids is not None else None

	# `inputs` has dims.N rows if every output shares this task's input locations
	# (isotopic_output_in_tasks), dims.O * dims.N rows otherwise.
	in_rows = slice(None) if dataset.inputs.shape[1] == dims.N else _output_rows(output_ids, o_id, dims.N)
	x = np.asarray(dataset.inputs[t, in_rows, 0])

	out_rows = _output_rows(output_ids, o_id, dims.N)
	y = np.asarray(dataset.outputs[t_id, out_rows, c_id])

	mask = ~np.isnan(y)
	return x[mask], y[mask]


def _grid_x(grid: Grid, dims: Dimensions, o_id: int):
	"""
	Extract a single output's grid input locations as a 1D numpy array (only 1D inputs, `I == 1`,
	supported). Every output shares the whole of `grid.points` if `grid.output_ids` is None
	(isotopic_output_in_grid); otherwise `output_ids` says which rows belong to which output.

	Note that `grid` need not be the one the data was generated on: a `mimosa.grid.GridBuilder`
	builds its points from the data, so their count is `len(grid.points)`, not `dims.G`.
	"""
	rows = slice(None) if grid.output_ids is None else np.flatnonzero(np.asarray(grid.output_ids) == o_id)
	return np.asarray(grid.points[rows, 0])


def _grid_block(grid: Grid, dims: Dimensions, o_id: int, length: int):
	"""
	Row selector for output `o_id` in a grid-indexed vector of `length` rows -- a
	Hyperprior/Hyperposterior mean, a prediction, or a sample. Rows are `dims.O` equal blocks over a
	shared pool of grid points (block-major convention, see `mimosa.synthetic.generate_data` and
	kernax's multi-output Mean/Kernel classes), or one block per output when `grid.output_ids`
	labels them.

	The block size is read off `length` rather than `dims.G`, so it also holds for a grid built from
	the data by a `mimosa.grid.GridBuilder`.
	"""
	if grid.output_ids is None:
		block = length // dims.O
		return slice(o_id * block, (o_id + 1) * block)
	return np.flatnonzero(np.asarray(grid.output_ids) == o_id)


def _mvn_cell(obj, grid: Grid, dims: Dimensions, k_id: int, c_id: int, o_id: int):
	"""
	Index a Hyperprior/Hyperposterior's `(K, C, O*G)`/`(K, C, O*G, O*G)` mean/covariance at
	`(k_id, c_id)`, broadcasting any axis of size 1 (shared hyperparameters) to index 0 instead,
	then slice out the `o_id`-th output block (see `_grid_block`).
	"""
	k = k_id if obj.mean.shape[0] > 1 else 0
	c = c_id if obj.mean.shape[1] > 1 else 0
	rows = _grid_block(grid, dims, o_id, obj.mean.shape[-1])
	cell = obj[k, c].marginal(rows)
	return cell.mean, cell.covariance


def _palette(n: int) -> list:
	"""
	Build one color per index in `[0, n)`. Shared by `plot_dataset` (colored by mixture assignment,
	or by task index) and `plot_clusters` (colored by cluster index), so the two can be composed on
	the same axes with matching cluster colors.

	Colors cycle once `n` exceeds the colormap's size (10 or 20), so distinct indices can share a
	color for large `n`.
	"""
	cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
	return [cmap(i % cmap.N) for i in range(n)]


def plot_channel(
	dataset: Dataset,
	dims: Dimensions,
	t_id: int,
	c_id: int,
	o_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	kind: Literal["scatter", "line"] = "scatter",
	**scatter_kwargs,
):
	"""
	Scatter-plot a single task's observed points for a single channel, one subplot per output.

	x-axis is the input value (only 1D inputs, `I == 1`, are supported for now), y-axis is the
	channel value.

	Parameters
	----------
	dataset
		Dataset to plot, as returned by `generate_data`.
	dims
		Dimensions of `dataset`, used to resolve `o_id="all"` and to slice its `O*N`-row arrays.
	t_id
		Index of the task to plot.
	c_id
		Index of the channel to plot.
	o_id
		"all" (default) or int, restrict the plot to a single output.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(o_id), 1)`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this task's points.
	kind
		"scatter" (default) draws each task's points, "line" joins them into a curve.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter` (`ax.plot` when `kind="line"`), overriding
		the defaults (s=15, alpha=0.7; linewidth=1.0, alpha=0.7 for a line).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), 1)`.
	"""
	if dataset.inputs.shape[-1] != 1:
		raise NotImplementedError("plot_channel only supports 1D inputs (I=1) for now.")

	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), 1, figsize=figsize)

	if kind not in ("scatter", "line"):
		raise ValueError(f"Expected kind 'scatter' or 'line', got {kind!r}.")

	kwargs = (_DEFAULT_LINE_KWARGS if kind == "line" else _DEFAULT_SCATTER_KWARGS) | scatter_kwargs

	for row, o in enumerate(o_ids):
		x, y = _task_xy(dataset, dims, t_id, c_id, o)
		a = ax[row, 0]
		(a.plot if kind == "line" else a.scatter)(x, y, color=color, **kwargs)
		a.set_title(f"channel {c_id}" + (f", output {o}" if len(o_ids) > 1 else ""))
		a.set_xlabel("input")
		a.set_ylabel("channel value")

	return fig, ax


def plot_task(
	dataset: Dataset,
	dims: Dimensions,
	t_id: int,
	c_id: IdArg = "all",
	o_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	kind: Literal["scatter", "line"] = "scatter",
	**scatter_kwargs,
):
	"""
	Scatter-plot a single task's observed points, looping `plot_channel` over channels.

	Parameters
	----------
	dataset
		Dataset to plot, as returned by `generate_data`.
	dims
		Dimensions of `dataset`, used to resolve `c_id`/`o_id="all"` and to slice its block-major arrays.
	t_id
		Index of the task to plot.
	c_id, o_id
		"all" (default) or int, restrict the plot to a single channel/output.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(o_id), len(c_id))`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this task's points.
	kind
		"scatter" (default) draws each task's points, "line" joins them into a curve.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter` (`ax.plot` when `kind="line"`), overriding
		the defaults (s=15, alpha=0.7; linewidth=1.0, alpha=0.7 for a line).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), len(c_id))`.
	"""
	c_ids = _resolve_ids(c_id, dims.C)
	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), len(c_ids), figsize=figsize)

	for col, c in enumerate(c_ids):
		plot_channel(
			dataset, dims, t_id, c, o_id=o_id, fig=fig, ax=ax[:, col : col + 1], color=color, kind=kind, **scatter_kwargs
		)

	return fig, ax


def plot_dataset(
	dataset: Dataset,
	dims: Dimensions,
	mixture: Mixture | None = None,
	t_id: IdArg = "all",
	c_id: IdArg = "all",
	o_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color_by_task: bool = False,
	legend: bool = True,
	kind: Literal["scatter", "line"] = "scatter",
	**scatter_kwargs,
):
	"""
	Scatter-plot a Dataset's observed points, looping `plot_task` over tasks. Points are colored
	by each task's hard cluster assignment if `mixture` is given, or by task index if
	`color_by_task` is True.

	Parameters
	----------
	dataset
		Dataset to plot, as returned by `generate_data`.
	dims
		Dimensions of `dataset`, used to resolve `t_id`/`c_id`/`o_id="all"` and to slice its
		block-major arrays.
	mixture
		Cluster assignments used to color tasks. If None, every task shares one color.
	t_id, c_id, o_id
		"all" (default) or int, restrict the plot to a single task/channel/output.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(o_id), len(c_id))`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color_by_task
		If True, give every task its own color (from the same palette as the cluster colors),
		instead of coloring by cluster assignment. Takes precedence over `mixture`.
	legend
		If True, add a legend mapping colors to cluster indices (if `mixture` is given) or to task
		indices (if `color_by_task` is True). Skipped when more tasks are plotted than the palette
		has distinct colors, since colors then repeat across tasks.
	kind
		"scatter" (default) draws each task's points, "line" joins them into a curve.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter` (`ax.plot` when `kind="line"`), overriding
		the defaults (s=15, alpha=0.7; linewidth=1.0, alpha=0.7 for a line).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), len(c_id))`.
	"""
	t_ids = _resolve_ids(t_id, dims.T)
	c_ids = _resolve_ids(c_id, dims.C)
	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), len(c_ids), figsize=figsize)

	if color_by_task:
		palette = _palette(dims.T)
		colors = {t: palette[t] for t in t_ids}
		# Colors repeat past the palette's size, which would make a legend misleading (and unwieldy).
		handles = (
			[plt.Line2D([0], [0], marker="o", linestyle="", color=palette[t], label=f"task {t}") for t in t_ids]
			if len(t_ids) <= len(set(palette))
			else []
		)
	elif mixture is None:
		colors = {t: "C0" for t in t_ids}
		handles = []
	else:
		K = mixture.responsibilities.shape[1]
		palette = _palette(K)
		assignments = np.asarray(mixture.assignments)
		colors = {t: palette[assignments[t]] for t in t_ids}
		handles = [
			plt.Line2D([0], [0], marker="o", linestyle="", color=palette[k], label=f"cluster {k}") for k in range(K)
		]

	for t in t_ids:
		plot_task(dataset, dims, t, c_id=c_id, o_id=o_id, fig=fig, ax=ax, color=colors[t], kind=kind, **scatter_kwargs)

	if legend and handles:
		fig.legend(handles=handles, loc="outside right center")

	return fig, ax


def plot_single_cluster_single_channel(
	grid: Grid,
	dims: Dimensions,
	k_id: int,
	c_id: int,
	hyperprior: Hyperprior | None = None,
	hyperposterior: Hyperposterior | None = None,
	o_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	**line_kwargs,
):
	"""
	Plot a single mean-process's values at the grid points, for a single channel, one subplot per
	output: `hyperprior.mean` as a dashed line, `hyperposterior.mean` as a solid line, and a
	confidence interval shaded from the diagonal of `hyperposterior.covariance`.

	Parameters
	----------
	grid
		Grid of points the mean-process is evaluated at (only 1D inputs, `I == 1`, supported for
		now). x-axis of the plot.
	dims
		Dimensions of the dataset `grid`/`hyperprior`/`hyperposterior` were generated/fitted from,
		used to resolve `o_id="all"` and to slice their block-major arrays.
	k_id
		Index of the mean-process (cluster) to plot.
	c_id
		Index of the channel to plot.
	hyperprior
		Prior distribution over the mean-process's grid values, shape `(K, C, O*G)`/`(K, C, O*G, O*G)`.
		Plotted as a dashed line if given; skipped otherwise.
	hyperposterior
		Posterior distribution over the mean-process's grid values, same shape as `hyperprior`.
		Plotted as a solid line with a shaded confidence interval if given; skipped otherwise.
	o_id
		"all" (default) or int, restrict the plot to a single output.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(o_id), 1)`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this cluster's prior/posterior lines and confidence interval.
	ci_scale
		Number of standard deviations spanned by the shaded confidence interval (default 1.96, ~95%).
	ci_alpha
		Opacity of the shaded confidence interval.
	**line_kwargs
		Extra keyword arguments forwarded to `ax.plot` for both the prior and posterior lines.

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), 1)`.
	"""
	if hyperprior is None and hyperposterior is None:
		raise ValueError("At least one of hyperprior/hyperposterior must be given.")
	if grid.points.shape[-1] != 1:
		raise NotImplementedError("plot_single_cluster_single_channel only supports 1D inputs (I=1) for now.")

	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), 1, figsize=figsize)

	for row, o in enumerate(o_ids):
		a = ax[row, 0]
		x = _grid_x(grid, dims, o)
		if hyperprior is not None:
			prior_mean, _ = _mvn_cell(hyperprior, grid, dims, k_id, c_id, o)
			a.plot(x, np.asarray(prior_mean), linestyle="--", color=color, **line_kwargs)
		if hyperposterior is not None:
			post_mean, post_cov = _mvn_cell(hyperposterior, grid, dims, k_id, c_id, o)
			post_mean = np.asarray(post_mean)
			post_std = np.sqrt(np.diagonal(np.asarray(post_cov)))
			a.plot(x, post_mean, linestyle="-", color=color, **line_kwargs)
			a.fill_between(
				x,
				post_mean - ci_scale * post_std,
				post_mean + ci_scale * post_std,
				color=color,
				alpha=ci_alpha,
				linewidth=0,
			)
		a.set_title(f"channel {c_id}" + (f", output {o}" if len(o_ids) > 1 else ""))
		a.set_xlabel("input")
		a.set_ylabel("channel value")

	return fig, ax


def plot_single_cluster(
	grid: Grid,
	dims: Dimensions,
	k_id: int,
	c_id: IdArg = "all",
	hyperprior: Hyperprior | None = None,
	hyperposterior: Hyperposterior | None = None,
	o_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	color="C0",
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	**line_kwargs,
):
	"""
	Plot a single mean-process's values at the grid points, looping `plot_single_cluster_single_channel`
	over channels.

	Parameters
	----------
	grid
		Grid of points the mean-process is evaluated at.
	dims
		Dimensions of the dataset `grid`/`hyperprior`/`hyperposterior` were generated/fitted from,
		used to resolve `c_id`/`o_id="all"` and to slice their block-major arrays.
	k_id
		Index of the mean-process (cluster) to plot.
	c_id, o_id
		"all" (default) or int, restrict the plot to a single channel/output.
	hyperprior, hyperposterior
		See `plot_single_cluster_single_channel`. At least one must be given.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(o_id), len(c_id))`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	color
		Color of this cluster's prior/posterior lines and confidence interval.
	ci_scale, ci_alpha
		See `plot_single_cluster_single_channel`.
	**line_kwargs
		Extra keyword arguments forwarded to `ax.plot` for both the prior and posterior lines.

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), len(c_id))`.
	"""
	if hyperprior is None and hyperposterior is None:
		raise ValueError("At least one of hyperprior/hyperposterior must be given.")

	c_ids = _resolve_ids(c_id, dims.C)
	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), len(c_ids), figsize=figsize)

	for col, c in enumerate(c_ids):
		plot_single_cluster_single_channel(
			grid,
			dims,
			k_id,
			c,
			hyperprior=hyperprior,
			hyperposterior=hyperposterior,
			o_id=o_id,
			fig=fig,
			ax=ax[:, col : col + 1],
			color=color,
			ci_scale=ci_scale,
			ci_alpha=ci_alpha,
			**line_kwargs,
		)

	return fig, ax


def plot_clusters(
	grid: Grid,
	dims: Dimensions,
	k_id: IdArg = "all",
	c_id: IdArg = "all",
	o_id: IdArg = "all",
	hyperprior: Hyperprior | None = None,
	hyperposterior: Hyperposterior | None = None,
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	legend: bool = True,
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	**line_kwargs,
):
	"""
	Plot every mean-process's values at the grid points, looping `plot_single_cluster` over
	clusters. Each cluster gets its own color, matching `plot_dataset`'s mixture-based coloring so
	the two can be composed on the same axes.

	Parameters
	----------
	grid
		Grid of points the mean-processes are evaluated at.
	dims
		Dimensions of the dataset `grid`/`hyperprior`/`hyperposterior` were generated/fitted from,
		used to resolve `k_id`/`c_id`/`o_id="all"` and to slice their block-major arrays.
	k_id, c_id, o_id
		"all" (default) or int, restrict the plot to a single cluster/channel/output.
	hyperprior, hyperposterior
		See `plot_single_cluster_single_channel`. At least one must be given.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots (e.g. `plot_dataset`). If
		given, `ax` must already have shape `(len(o_id), len(c_id))`. A new figure/axes grid is
		created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	legend
		If True, add a legend mapping colors to cluster indices. Set to False on one of the calls
		when composing with `plot_dataset` on the same figure, to avoid a duplicate legend.
	ci_scale, ci_alpha
		See `plot_single_cluster_single_channel`.
	**line_kwargs
		Extra keyword arguments forwarded to `ax.plot` for both the prior and posterior lines.

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), len(c_id))`.
	"""
	if hyperprior is None and hyperposterior is None:
		raise ValueError("At least one of hyperprior/hyperposterior must be given.")

	k_ids = _resolve_ids(k_id, dims.K)
	c_ids = _resolve_ids(c_id, dims.C)
	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), len(c_ids), figsize=figsize)

	palette = _palette(dims.K)

	for k in k_ids:
		plot_single_cluster(
			grid,
			dims,
			k,
			c_id=c_id,
			hyperprior=hyperprior,
			hyperposterior=hyperposterior,
			o_id=o_id,
			fig=fig,
			ax=ax,
			color=palette[k],
			ci_scale=ci_scale,
			ci_alpha=ci_alpha,
			**line_kwargs,
		)

	if legend:
		handles = [plt.Line2D([0], [0], color=palette[k], label=f"cluster {k}") for k in k_ids]
		fig.legend(handles=handles, loc="outside right center")

	return fig, ax


def plot_single_task_prediction(
	dataset: Dataset,
	grid: Grid,
	dims: Dimensions,
	hyperposterior: Hyperposterior,
	mixture: Mixture,
	t_id: int,
	c_id: int,
	prediction: MultivariateNormal | None = None,
	samples: Array | None = None,
	o_id: IdArg = "all",
	fig=None,
	ax=None,
	figsize: tuple[float, float] | None = None,
	legend: bool = True,
	point_color="black",
	prediction_color="black",
	sample_color="gray",
	ci_scale: float = 1.96,
	ci_alpha: float = 0.2,
	sample_alpha: float = 0.15,
	**scatter_kwargs,
):
	"""
	Plot a single task's prediction for a single channel, one subplot per output: the task's
	observed points (scatter), every mean-process's hyperposterior mean (dashed lines, one per
	cluster, colored and made transparent by that cluster's mixture coefficient for this task), and
	optionally the task's predictive distribution (solid mean line + shaded confidence interval)
	and samples drawn from it (thin lines, all the same color/alpha).

	Parameters
	----------
	dataset
		Dataset the task's observed points are read from.
	grid
		Grid of points the hyperposterior/prediction are evaluated at (only 1D inputs, `I == 1`,
		supported for now). x-axis of the plot.
	dims
		Dimensions of the dataset/grid/hyperposterior/prediction, used to resolve `o_id="all"` and
		to slice their block-major arrays. `mimosa.prediction` doesn't handle multi-output
		explicitly yet, but follows the same `O*G` block-major convention as everything else.
	hyperposterior
		Posterior distribution over every mean-process's values at the grid points.
	mixture
		Soft-clustering of the dataset's tasks into mean-processes, used to set each hyperposterior
		line's alpha to this task's mixture coefficient towards that cluster.
	t_id
		Index of the task to plot.
	c_id
		Index of the channel to plot.
	prediction
		This task's predictive distribution at the grid points (e.g. from
		`mimosa.prediction.predict_task_channel`), for a single channel and mean-process, shape
		`(O*G,)`/`(O*G, O*G)`. Plotted as a solid mean line with a shaded confidence interval if
		given; skipped otherwise.
	samples
		Samples drawn from `prediction`, shape `(S, O*G)`. Plotted as thin lines, all the same
		color/alpha, if given; skipped otherwise.
	o_id
		"all" (default) or int, restrict the plot to a single output.
	fig, ax
		Existing figure/axes to draw on, to combine with other plots. If given, `ax` must already
		have shape `(len(o_id), 1)`. A new figure/axes grid is created if None.
	figsize
		Passed to `plt.subplots` when a new figure is created.
	legend
		If True, add a legend mapping colors to cluster indices, with this task's mixture
		coefficient (%) towards each.
	point_color
		Color of the observed points.
	prediction_color
		Color of the predictive mean line and confidence interval.
	sample_color
		Color of the prediction samples.
	ci_scale
		Number of standard deviations spanned by the shaded confidence interval (default 1.96, ~95%).
	ci_alpha
		Opacity of the shaded confidence interval.
	sample_alpha
		Opacity of each sample line.
	**scatter_kwargs
		Extra keyword arguments forwarded to `ax.scatter` for the observed points, overriding the
		defaults (s=15, alpha=0.7).

	Returns
	-------
	fig, ax
		The (possibly newly created) figure and 2D array of axes, shape `(len(o_id), 1)`.
	"""
	if dataset.inputs.shape[-1] != 1 or grid.points.shape[-1] != 1:
		raise NotImplementedError("plot_single_task_prediction only supports 1D inputs (I=1) for now.")

	o_ids = _resolve_ids(o_id, dims.O)

	fig, ax = _get_fig_ax(fig, ax, len(o_ids), 1, figsize=figsize)

	K = hyperposterior.mean.shape[0]
	palette = _palette(K)
	weights = np.asarray(mixture.responsibilities[t_id])

	scatter_kwargs = _DEFAULT_SCATTER_KWARGS | scatter_kwargs

	for row, o in enumerate(o_ids):
		a = ax[row, 0]
		x_grid = _grid_x(grid, dims, o)
		block = _grid_block(grid, dims, o, hyperposterior.mean.shape[-1])

		if samples is not None:
			for s in np.asarray(samples)[:, block]:
				a.plot(x_grid, s, color=sample_color, alpha=sample_alpha, linewidth=1)

		for k in range(K):
			cluster_mean, _ = _mvn_cell(hyperposterior, grid, dims, k, c_id, o)
			a.plot(x_grid, np.asarray(cluster_mean), linestyle="--", color=palette[k], alpha=float(weights[k]))

		if prediction is not None:
			pred_block = prediction.marginal(block)
			pred_mean = np.asarray(pred_block.mean)
			pred_std = np.sqrt(np.diagonal(np.asarray(pred_block.covariance)))
			a.plot(x_grid, pred_mean, linestyle="-", color=prediction_color)
			a.fill_between(
				x_grid,
				pred_mean - ci_scale * pred_std,
				pred_mean + ci_scale * pred_std,
				color=prediction_color,
				alpha=ci_alpha,
				linewidth=0,
			)

		x_obs, y_obs = _task_xy(dataset, dims, t_id, c_id, o)
		a.scatter(x_obs, y_obs, color=point_color, **scatter_kwargs)

		a.set_title(f"task {t_id}, channel {c_id}" + (f", output {o}" if len(o_ids) > 1 else ""))
		a.set_xlabel("input")
		a.set_ylabel("channel value")

	if legend:
		handles = [
			plt.Line2D(
				[0], [0], linestyle="--", color=palette[k], label=f"cluster {k} ({100 * float(weights[k]):.0f}%)"
			)
			for k in range(K)
		]
		fig.legend(handles=handles, loc="outside right center")

	return fig, ax
