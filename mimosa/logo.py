"""
Functions to generate the logo of MIMOSA, saving them in `docs/images`.

Every element is a Gaussian process realisation drawn with `mimosa.sample_gp`: the logo is a
picture of what the library does. Run `python -m mimosa.logo` to regenerate the images.

Sampling needs float64 -- the covariance of a few hundred densely-spaced points under a squared
exponential kernel is far too ill-conditioned for a float32 Cholesky, which silently returns NaN.
This module therefore enables jax's x64 mode on import, and is deliberately left out of
`mimosa/__init__.py` so importing the library never has that side effect.
"""

from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, vmap
from kernax import SEKernel, VarianceKernel
from matplotlib import font_manager
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from mimosa.sampling import sample_gp

jax.config.update("jax_enable_x64", True)

__all__ = ["generate_samples", "generate_flower", "generate_logo", "generate_square_logo"]

TRANSPARENT_BG = False  # False: white background

_ROOT = Path(__file__).resolve().parent.parent
_IMAGES_DIR = _ROOT / "docs" / "images"
_FONT_PATH = _ROOT / "docs" / "assets" / "fonts" / "LibertinusSans-Regular.ttf"

# Pink, following the RdPu palette from the original MagmaClustR. The lightest fifth of RdPu is
# near-white, so sampling starts at 0.3 to keep every curve legible on a white background.
_PALETTE = plt.get_cmap("RdPu")
_PALETTE_RANGE = (0.3, 0.95)
_NAME_COLOUR = _PALETTE(0.95)
_MEAN_COLOUR = "0.65"

_GRID = 400  # points per curve; enough that the curves read as smooth at any output size

# Conditioning a process on a handful of nearby inputs makes its covariance very close to
# singular, which a float64 Cholesky still cannot factor at the library's default jitter.
_JITTER = jnp.asarray(1e-8)

# Space left between the drawing and the edge of the image, in data units, and between the
# elements of a logo. Insetting every element by the same margin is what makes the padding match
# from side to side. The samples run edge to edge, so only the bloom and the name are held off the
# left and right edges, and they are held off twice as far as the top and bottom.
_MARGIN = 0.35
_SIDE_MARGIN = 2 * _MARGIN
_GAP = 0.25

# The wide logos share a 12.8 x 6.4 frame, so their images come out at 2:1.
_SPAN, _TOP, _BOTTOM = 6.4, 4.3, -2.1


# --- Gaussian processes ------------------------------------------------------------------------


def _condition(kernel, x: Array, anchors: Array, values: Array) -> tuple[Array, Array]:
	"""
	Mean and covariance of `kernel` over `x`, conditioned on the process taking `values` at the
	inputs `anchors`. Realisations drawn from it all pass through those points and fan back out to
	the prior in between, which is the shape both the flower and the task samples are built from.
	"""
	kxa = kernel(x, anchors)
	kaa = kernel(anchors) + _JITTER * jnp.eye(anchors.shape[0])
	gain = kxa @ jnp.linalg.inv(kaa)
	return gain @ values, kernel(x) - gain @ kxa.T


def _draw(key: Array, mean: Array, cov: Array, n: int) -> np.ndarray:
	"""`n` realisations of the process `(mean, cov)`. Shape `(n, len(mean))`."""
	return np.asarray(vmap(lambda k: sample_gp(k, mean, cov, _JITTER))(jr.split(key, n)))


def _petals(
	key: Array,
	n: int = 44,
	radius: float = 1.8,
	length_scale: float = 1.8,
	width: float = 1.8,
	stem_gap: float = 0.45,
	stem_spread: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	The flower's petals, as `(xs, ys)` arrays of shape `(n, _GRID)` with the stem at the origin and
	points outside the bloom set to NaN.

	Petals are realisations conditioned on four inputs that share the same output value and sit
	below the drawn range, so they are never themselves drawn: pinning the process over a short
	stretch leaves every petal emerging from the stem in the same direction, and only then fanning
	out to span the whole uncertainty estimate. `stem_gap` is how far the drawn range starts beyond
	the last of those inputs, so raising it cuts the base of the bloom a little higher, where the
	petals have already begun to separate. They are clipped to a disc of `radius` around the stem
	for a round boundary, and rotated 90° so the bloom opens upwards.

	`length_scale` controls how much a petal bends; because four nearby anchors under a long length
	scale also flatten the fan, `width` rescales the deviations to a fixed spread at the tip and
	keeps the two choices independent.
	"""
	x = jnp.linspace(0.0, radius, _GRID)[:, None]
	kernel = VarianceKernel(1.0) * SEKernel(length_scale=length_scale)
	anchors = jnp.linspace(-stem_gap - stem_spread, -stem_gap, 4)[:, None]

	mean, cov = _condition(kernel, x, anchors, jnp.zeros(anchors.shape[0]))
	petals = _draw(key, mean, cov, n) * (width / float(jnp.sqrt(jnp.diag(cov)[-1])))

	xs = np.asarray(x[:, 0])
	inside = xs**2 + petals**2 <= radius**2
	# Rotate 90°: a petal's input runs up the y axis, its output across the x axis.
	return np.where(inside, -petals, np.nan), np.where(inside, np.broadcast_to(xs, petals.shape), np.nan)


def _task_curves(
	key: Array,
	n: int = 24,
	span: float = 6.0,
	cluster_variance: float = 0.4,
	cluster_length_scale: float = 1.8,
	task_length_scale: float = 1.6,
	task_spread: float = 0.35,
	crossings: tuple[float, ...] = (-4.0, 0.6, 4.4),
	offsets: tuple[float, ...] = (0.55, -0.5, 0.45),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	A cluster's mean process and `n` task realisations over `[-span, span]`, as `(xs, mean, tasks)`.

	The mean process is a realisation of the cluster kernel. Every task is that mean plus a
	realisation of the task kernel conditioned to take `offsets` at `crossings` -- offsets from the
	mean process, not points on it, so the tasks visibly agree where they are pinned, away from the
	mean, and relax back onto it where they are not.
	"""
	x = jnp.linspace(-span, span, _GRID)[:, None]
	cluster_key, task_key = jr.split(key)

	cluster_kernel = VarianceKernel(cluster_variance) * SEKernel(length_scale=cluster_length_scale)
	mean = _draw(cluster_key, jnp.zeros(_GRID), cluster_kernel(x), 1)[0]

	task_kernel = VarianceKernel(task_spread**2) * SEKernel(length_scale=task_length_scale)
	anchors = jnp.asarray(crossings)[:, None]
	task_mean, task_cov = _condition(task_kernel, x, anchors, jnp.asarray(offsets))
	return np.asarray(x[:, 0]), mean, mean + _draw(task_key, task_mean, task_cov, n)


def _band_extent(curves: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[float, float]:
	"""Lowest and highest point of `curves`, mean process included."""
	_, mean, tasks = curves
	return min(mean.min(), tasks.min()), max(mean.max(), tasks.max())


# --- Drawing -----------------------------------------------------------------------------------


def _colours(n: int) -> list:
	"""`n` colours spanning the palette, light to dark."""
	return [_PALETTE(v) for v in np.linspace(*_PALETTE_RANGE, n)]


def _plot_petals(
	ax,
	petals: tuple[np.ndarray, np.ndarray],
	dx: float = 0.0,
	dy: float = 0.0,
	scale: float = 1.0,
) -> None:
	"""
	Draw `petals` onto `ax`, resized by `scale` and with the stem moved to `(dx, dy)`.

	Resizing here rather than sampling at a different radius is what keeps the bloom identical
	across the logos: the same realisations are drawn, only larger or smaller.
	"""
	xs, ys = petals
	for x, y, colour in zip(xs, ys, _colours(len(xs))):
		ax.plot(x * scale + dx, y * scale + dy, color=colour, lw=1.6, alpha=0.75)


def _plot_samples(
	ax,
	curves: tuple[np.ndarray, np.ndarray, np.ndarray],
	base: float | None = None,
	scale: float = 1.0,
) -> None:
	"""
	Draw `curves` onto `ax` with the mean process as a dashed grey line, squashed vertically by
	`scale` and resting on `base` (their own lowest point, by default).

	Squashing the band is what frees height in a fixed frame: the samples keep their full width and
	their crossings, and everything the band gives up goes to the bloom and the name.
	"""
	xs, mean, tasks = curves
	low, _ = _band_extent(curves)
	base = low if base is None else base

	def place(y: np.ndarray) -> np.ndarray:
		return base + (y - low) * scale

	ax.plot(xs, place(mean), ls="--", lw=1.4, color=_MEAN_COLOUR, zorder=0)
	for task, colour in zip(tasks, _colours(len(tasks))):
		ax.plot(xs, place(task), color=colour, lw=1.1, alpha=0.45)


@cache
def _wordmark() -> TextPath:
	"""
	The name as glyph outlines, at an arbitrary size that `_name` scales from.

	Drawing it as a path rather than as text is what lets the logos line the name up with the rest
	of the drawing exactly: a text box is measured from the font's advance widths, so it extends
	past the last glyph by that glyph's side bearing, and the margin it leaves is visibly wider
	than the one on the other side of the image.
	"""
	return TextPath((0.0, 0.0), "MIMOSA", prop=font_manager.FontProperties(fname=_FONT_PATH, size=100))


def _name_aspect() -> float:
	"""How many times wider than tall the name is."""
	ink = _wordmark().get_extents()
	return ink.width / ink.height


def _name(ax, x: float, y: float, height: float, ha: str, va: str = "center") -> None:
	"""
	Draw the name on `ax`, `height` data units tall, with the edge or centre of its ink named by
	`ha` at `x` and the one named by `va` at `y`.
	"""
	path = _wordmark()
	ink = path.get_extents()
	scale = height / ink.height
	horizontal = {"left": ink.x0, "center": ink.x0 + 0.5 * ink.width, "right": ink.x1}
	vertical = {"bottom": ink.y0, "center": ink.y0 + 0.5 * ink.height, "top": ink.y1}
	placed = Affine2D().scale(scale).translate(x - scale * horizontal[ha], y - scale * vertical[va])
	ax.add_patch(PathPatch(placed.transform_path(path), facecolor=_NAME_COLOUR, edgecolor="none"))


def _wide_canvas():
	"""The frame both wide logos are laid out in."""
	fig, ax = _canvas((8.0, 4.0))
	ax.set_xlim(-_SPAN, _SPAN)
	ax.set_ylim(_BOTTOM, _TOP)
	return fig, ax


def _canvas(figsize: tuple[float, float], aspect: str = "equal"):
	"""A single full-bleed axes with no decorations."""
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
	ax.set_aspect(aspect)
	ax.set_axis_off()
	return fig, ax


def _save(fig, name: str, dpi: float = 300, crop: bool = True, transparent: bool = TRANSPARENT_BG) -> Path:
	"""
	Save `fig` to `docs/images/<name>.png`, or to `<name>_no_bg.png` if `transparent`, and close it.

	`crop` trims the image to its content, which suits the two component images. The logos instead
	set their own limits and margins, and are saved uncropped so that the file comes out at exactly
	`figsize * dpi` pixels.
	"""
	_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
	path = _IMAGES_DIR / f"{name}{'_no_bg' if transparent else ''}.png"
	crop_kwargs = {"bbox_inches": "tight", "pad_inches": 0.1} if crop else {}
	fig.savefig(path, dpi=dpi, transparent=transparent, **crop_kwargs)
	plt.close(fig)
	return path


# --- Images ------------------------------------------------------------------------------------


def generate_samples() -> Path:
	"""
	Samples are transparent and pink (following RdPu palette from original MagmaClustR), varying
	around a dashed-grey-line background mean process.
	"""
	fig, ax = _canvas((8.0, 2.5), aspect="auto")
	_plot_samples(ax, _task_curves(jr.PRNGKey(5)))
	return _save(fig, "logo_samples")


def generate_flower() -> Path:
	"""
	Flower is samples conditioned on four points of equal value sitting below the drawn range, over
	the range where they leave those points to span the whole uncertainty estimate, cut to make a
	nice round boundary, and then rotated 90° to look like a pink mimosa flower.
	"""
	fig, ax = _canvas((4.0, 2.5))
	_plot_petals(ax, _petals(jr.PRNGKey(3)))
	return _save(fig, "logo_flower")


def generate_logo(samples_scale: float = 0.75, section_gap: float = 0.75, transparent: bool = TRANSPARENT_BG) -> Path:
	"""
	Logo is flower and name (using LibertinusSans font) above samples around their mean process,
	spanning all the way from left to right in a nice continuous fashion.

	`samples_scale` squashes the samples band vertically and `section_gap` holds the two sections
	apart. The frame is fixed, so the bloom takes up whatever height the two of them leave.
	`transparent` drops the white background and saves to `logo_no_bg.png` instead.
	"""
	fig, ax = _wide_canvas()

	# The samples run the full width, edge to edge, resting on the bottom margin, and the bloom is
	# resized to fill the height left above them.
	curves = _task_curves(jr.PRNGKey(5), span=_SPAN)
	low, high = _band_extent(curves)
	base = _BOTTOM + _MARGIN
	height = (_TOP - _MARGIN) - (base + samples_scale * (high - low) + section_gap)

	petals = _petals(jr.PRNGKey(3))
	scale = height / (np.nanmax(petals[1]) - np.nanmin(petals[1]))
	dx = -_SPAN + _SIDE_MARGIN - scale * np.nanmin(petals[0])
	stem_height = _TOP - _MARGIN - scale * np.nanmax(petals[1])
	_plot_petals(ax, petals, dx=dx, dy=stem_height, scale=scale)

	# The name is inset from the right edge by the margin the bloom is inset from the left, and
	# rests on the bloom's bottom edge. A taller bloom is also a wider one, so the name is capped to
	# the width actually left beside it rather than growing until the two overlap.
	free = (_SPAN - _SIDE_MARGIN) - (dx + scale * np.nanmax(petals[0]) + _GAP)
	name_height = min(0.5 * height, free / _name_aspect())
	_name(ax, _SPAN - _SIDE_MARGIN, stem_height, height=name_height, ha="right", va="bottom")

	_plot_samples(ax, curves, base=base, scale=samples_scale)
	return _save(fig, "logo", dpi=160, crop=False, transparent=transparent)


def generate_square_logo(transparent: bool = TRANSPARENT_BG) -> Path:
	"""
	Square logo is the flower with the name underneath it, for use as an icon or avatar where the
	wide logo would have to be shrunk to fit. `transparent` drops the white background and saves to
	`logo_square_no_bg.png` instead.
	"""
	half, gap, name_height = 2.4, 0.5, 0.55  # a 4.8 x 4.8 frame, so the image comes out square
	fig, ax = _canvas((4.0, 4.0))
	ax.set_xlim(-half, half)
	ax.set_ylim(-half, half)

	# Bloom and name are centred as one block, so the frame keeps matching margins above and below.
	petals = _petals(jr.PRNGKey(3))
	bloom_height = np.nanmax(petals[1]) - np.nanmin(petals[1])
	stem_height = 0.5 * (bloom_height + gap + name_height) - bloom_height

	_plot_petals(ax, petals, dx=-0.5 * (np.nanmin(petals[0]) + np.nanmax(petals[0])), dy=stem_height)
	_name(ax, 0.0, stem_height - gap - 0.5 * name_height, height=name_height, ha="center")
	return _save(fig, "logo_square", dpi=160, crop=False, transparent=transparent)


if __name__ == "__main__":
	images = (
		generate_samples(),
		generate_flower(),
		generate_logo(),
		generate_logo(transparent=True),
		generate_square_logo(),
		generate_square_logo(transparent=True),
	)
	for path in images:
		print(f"wrote {path}")
