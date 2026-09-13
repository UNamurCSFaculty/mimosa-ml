"""
Read and write Dataset objects to/from external file formats.

Multi-output datasets (`dataset.output_ids is not None`, or several equal-size output blocks when
it's `None`) are stored as extra columns of one wide CSV, `Output<o>_<c>`/`Noise<o>_<c>` (`<o>` =
output index, `<c>` = channel index, both 1-based): `save_csv`/`load_csv` dispatch to
`save_single_csv`/`load_single_csv` for a single path, or to
`split_into_single_output_datasets`/`merge_multioutput_datasets` for a sequence of paths (one per
output, in output-index order) -- worth it for heterotopic data, where a single wide file would pay
an O-times empty-column cost.
"""

import re
from collections.abc import Sequence
from pathlib import Path
import numpy as np
import polars as pl
import jax.numpy as jnp

from mimosa.data_structures import Dataset

__all__ = [
	"save_single_csv",
	"split_into_single_output_datasets",
	"save_csv",
	"load_single_csv",
	"merge_multioutput_datasets",
	"load_csv",
]


def _exact_equal(a, b) -> bool:
	"""
	True iff `a` and `b` (broadcastable float arrays) match at every position, NaN included as a
	match. Used to decide whether input coordinates read from a CSV can be collapsed -- see
	`load_single_csv` and `merge_multioutput_datasets`. Deliberately exact (no tolerance): within one
	file, two coordinates that format to the same decimal text read back to the same float, and two
	that format differently are genuinely different grid points, not neighbours to be fused.
	"""
	return bool(jnp.all((jnp.isnan(a) & jnp.isnan(b)) | (a == b)))


def save_single_csv(csv_path: str | Path, dataset: Dataset) -> None:
	"""
	Write a Dataset to a single CSV file, in wide multi-output form: one row per (task, input
	coordinate), one column pair `Output<o>_<c>`/`Noise<o>_<c>` per output/channel.

	A coordinate shared by several outputs (structurally, whenever outputs share input locations;
	otherwise only by coincidence) is written once, with every sharing output's cells filled on that
	row. A coordinate that belongs to only some outputs' input axes is still written once, with an
	*empty* field (not "nan") for the outputs it doesn't belong to -- this is what lets `load_csv`
	recover, rather than guess, which output a point belongs to (see `load_single_csv`). `nan` is
	written where a coordinate is in an output's axis but its value is unobserved. `Noise<o>_<c>` is
	only written when `dataset.known_output_noise is not None`, and mirrors its `Output<o>_<c>` cell's
	sentinel.

	A point missing from every channel of every output (NaN padding a variable-length task) is
	dropped entirely, so tasks can end up with a different number of rows. A point real on its input
	but missing every channel of an output it belongs to keeps its row, with "nan" written for that
	output's cells.

	Parameters
	----------
	csv_path
		Path to write the CSV file to.
	dataset
		Dataset to write (see `save_csv` for a sequence of paths, one file per output).
	"""
	single_output_datasets = split_into_single_output_datasets(dataset)

	frames = []
	for o, single_output_dataset in enumerate(single_output_datasets, start=1):
		T, N, C = single_output_dataset.outputs.shape
		I = single_output_dataset.inputs.shape[-1]

		inputs = np.broadcast_to(np.asarray(single_output_dataset.inputs), (T, N, I)).reshape(T * N, I)
		outputs = np.asarray(single_output_dataset.outputs).reshape(T * N, C)
		task_ids = np.repeat(np.arange(T), N)
		noise = (
			None
			if single_output_dataset.known_output_noise is None
			else np.asarray(single_output_dataset.known_output_noise).reshape(T * N, C)
		)

		# Padding rows (NaN on every input) are always dropped; a real point with all-NaN outputs is
		# always kept -- it's what lets an isotopic_tasks dataset round-trip.
		keep = ~np.isnan(inputs).any(axis=-1)
		task_ids, inputs, outputs = task_ids[keep], inputs[keep], outputs[keep]
		if noise is not None:
			noise = noise[keep]

		columns = ["TaskID"] + [f"Input{i + 1}" for i in range(I)] + [f"Output{o}_{c + 1}" for c in range(C)]
		data = [task_ids] + [inputs[:, i] for i in range(I)] + [outputs[:, c] for c in range(C)]
		if noise is not None:
			columns += [f"Noise{o}_{c + 1}" for c in range(C)]
			data += [noise[:, c] for c in range(C)]
		frames.append(pl.DataFrame(dict(zip(columns, data))))

	# Merge output frames on (TaskID, Input*): a coordinate present in several frames lands on one
	# row with every present output's cells filled; a coordinate present in only some frames gets a
	# genuine polars null (not NaN) on the others -- the "not in this output's axis" sentinel.
	df = frames[0]
	key_cols = [c for c in df.columns if c == "TaskID" or c.startswith("Input")]
	for frame in frames[1:]:
		df = df.join(frame, on=key_cols, how="full", coalesce=True)

	# A full join's row order is an implementation detail, not a function of the row's own content --
	# two tasks with the exact same input points could come out with the points in a different
	# relative order. Sorting by the join key itself makes row order a pure function of content again,
	# so identical tasks read back identical: load_single_csv's task-uniformity collapse depends on
	# this.
	df = df.sort(key_cols)

	df.write_csv(csv_path)


def split_into_single_output_datasets(dataset: Dataset) -> list[Dataset]:
	"""
	Split a multi-output Dataset into one single-output Dataset per output, in output-index order.
	Inverse of `merge_multioutput_datasets`.

	Parameters
	----------
	dataset
		Dataset to split.

	Returns
	-------
	One Dataset per output, each re-padded to that output's own longest task. A single-output
	`dataset` (`output_ids is None` and only one implied output) is returned as `[dataset]`.
	"""
	T, oN, C = dataset.outputs.shape
	outputs = np.asarray(dataset.outputs)
	known_noise = None if dataset.known_output_noise is None else np.asarray(dataset.known_output_noise)

	if dataset.output_ids is None:
		# isotopic_output_in_tasks: every output has the same N points and shares the same inputs
		# -- N is `inputs`' own row count (never O*N-collapsed like `outputs`), no dims/config needed.
		N = dataset.inputs.shape[1]
		O = oN // N
		if O == 1:
			return [dataset]
		return [
			Dataset(
				inputs=dataset.inputs,
				outputs=jnp.asarray(outputs[:, o * N : (o + 1) * N, :]),
				known_output_noise=None if known_noise is None else jnp.asarray(known_noise[:, o * N : (o + 1) * N, :]),
			)
			for o in range(O)
		]

	# Heterotopic outputs: each may have a different number of points, varying per task.
	inputs = np.broadcast_to(np.asarray(dataset.inputs), (T, oN, dataset.inputs.shape[-1]))
	output_ids = np.broadcast_to(np.asarray(dataset.output_ids), (T, oN))
	O = int(output_ids.max()) + 1

	datasets = []
	for o in range(O):
		mask = output_ids == o  # (T, oN)
		rank = np.cumsum(mask, axis=1) - 1  # each True row's position within output o's compacted block
		N_o = int(mask.sum(axis=1).max())

		out_inputs = np.full((T, N_o, inputs.shape[-1]), np.nan)
		out_outputs = np.full((T, N_o, C), np.nan)
		out_noise = None if known_noise is None else np.full((T, N_o, C), np.nan)

		task_idx, row_idx = np.nonzero(mask)
		dest = rank[task_idx, row_idx]
		out_inputs[task_idx, dest] = inputs[task_idx, row_idx]
		out_outputs[task_idx, dest] = outputs[task_idx, row_idx]
		if known_noise is not None:
			out_noise[task_idx, dest] = known_noise[task_idx, row_idx]

		datasets.append(
			Dataset(
				inputs=jnp.asarray(out_inputs),
				outputs=jnp.asarray(out_outputs),
				known_output_noise=None if out_noise is None else jnp.asarray(out_noise),
			)
		)
	return datasets


def save_csv(csv_path: str | Path | Sequence[str | Path], dataset: Dataset) -> None:
	"""
	Write a Dataset to CSV. A single path writes one wide file (see `save_single_csv`); a sequence of
	paths splits `dataset` into one file per output (see `split_into_single_output_datasets`), in
	output-index order.

	Parameters
	----------
	csv_path
		Path (single-output dataset) or one path per output, in output-index order.
	dataset
		Dataset to write.

	Raises
	------
	ValueError
		If `csv_path` is a sequence whose length doesn't match `dataset`'s number of outputs.
	"""
	if isinstance(csv_path, (str, Path)):
		save_single_csv(csv_path, dataset)
		return

	csv_paths = list(csv_path)
	datasets = split_into_single_output_datasets(dataset)
	if len(datasets) != len(csv_paths):
		raise ValueError(f"Dataset has {len(datasets)} output(s), but got {len(csv_paths)} csv path(s).")
	for path, single_output_dataset in zip(csv_paths, datasets):
		save_single_csv(path, single_output_dataset)


_OUTPUT_COLUMN_RE = re.compile(r"^Output(\d+)_(\d+)$")


def _infer_output_groups(output_cols: list[str], output_groups: Sequence | None) -> tuple[np.ndarray, int, int]:
	"""
	Assign each of `output_cols` to one of `O` output groups, either from an explicit `output_groups`
	override or inferred from self-describing `Output<o>_<c>` names -- see `load_single_csv`.

	Parameters
	----------
	output_cols
		Names of the CSV's "Output*" columns, in file order.
	output_groups
		Explicit group label per column (see `load_single_csv`), or None to infer from column names.

	Returns
	-------
	group_index
		Group index (0-based) of each entry of `output_cols`, in output-index order -- i.e. the order
		in which each group's label first appears in `output_cols` (or in `output_groups`, when given).
	O, C
		Number of output groups, and number of columns (channels) per group.

	Raises
	------
	ValueError
		If `output_groups` is given and doesn't have exactly one entry per column of `output_cols`, or
		if the resulting groups don't all have the same number of columns (`Dataset` cannot represent
		a ragged number of channels per output).
	"""
	if output_groups is not None:
		if len(output_groups) != len(output_cols):
			raise ValueError(
				f"output_groups has {len(output_groups)} entries, but the CSV has {len(output_cols)} 'Output*' columns."
			)
		labels = list(output_groups)
	else:
		matches = [_OUTPUT_COLUMN_RE.match(c) for c in output_cols]
		if all(matches):
			# Self-describing Output<o>_<c> names: <o> is the group label.
			labels = [int(m.group(1)) for m in matches]
		else:
			# Legacy flat "Output*" names (arbitrary suffixes): one output, one channel per column.
			labels = [0] * len(output_cols)

	first_seen = {}
	group_index = np.empty(len(labels), dtype=int)
	for i, label in enumerate(labels):
		group_index[i] = first_seen.setdefault(label, len(first_seen))
	O = len(first_seen)

	counts = np.bincount(group_index)
	if not np.all(counts == counts[0]):
		raise ValueError(f"All {O} output groups must have the same number of channels, got sizes {counts.tolist()}.")

	return group_index, O, int(counts[0])


def load_single_csv(csv_path: str | Path, output_groups: Sequence | None = None) -> Dataset:
	"""
	Read a Dataset from a single CSV file written by `save_single_csv`, or a compatible foreign one.

	Columns: "TaskID", "Input<suffix>" (one per input dim), "Output<o>_<c>"/"Noise<o>_<c>" (one pair
	per output/channel; "Noise*" optional). `<o>`/`<c>` are inferred from column names unless
	`output_groups` overrides them (see below); any other column is ignored.

	A row is read once for a coordinate shared by several outputs. An empty "Output<o>_<c>" field
	means that coordinate isn't in output `o`'s input axis for that task (the row is absent from
	output `o`'s block entirely); "nan" means it is, but unobserved. This distinction -- not a vote
	across tasks -- is what lets heterotopic and isotopic outputs be told apart from one file with no
	extra information: a coordinate belongs to output `o`'s axis, for a given task, iff at least one
	of output `o`'s `C` cells is non-empty on that row.

	Tasks may have a different number of rows, per output. Each output is padded to its own longest
	task to build a uniform-shape block; blocks are concatenated in output order. If every output
	turns out to have selected the exact same rows, `output_ids` is redundant and dropped. If every
	task then turns out to hold the exact same input points, `inputs`' leading axis collapses to 1 --
	the shared-input-locations representation `isotopic_tasks` datasets use. Both comparisons are
	exact (no tolerance): see `_exact_equal`.

	Within one output's block, points come back in the file's row order, not necessarily the order
	they were generated in: `save_single_csv` sorts by `(TaskID, Input*)` before writing (see there).
	Harmless -- `inputs`, `outputs`, `known_output_noise` and `output_ids` are all built from the same
	row selection, so they permute together, and a GP is exchangeable over its points.

	A foreign CSV with no empty "Output*" cells (only "nan") reads back as isotopic outputs, with
	every "nan" treated as an unobserved point rather than a structurally absent one -- a superset of
	whatever the writer intended (more NaN-padded points, no lost values), safe because NaN outputs
	are masked downstream (`hyperpost`, `nll`, prediction).

	Parameters
	----------
	csv_path
		Path to read the CSV file from.
	output_groups
		Group label of each "Output*" column, in file order, e.g. `[1, 1, 2, 2, 3, 3]` for 3 outputs
		of 2 channels each. Overrides the names of "Output*" columns for foreign CSVs that don't
		follow the self-describing "Output<o>_<c>" convention. None (default) infers grouping from
		column names: legacy flat "Output1".."OutputC" (no "_<c>" suffix) is read as a single output
		of C channels. Output index order is the order each label first appears (in `output_groups`,
		or in the file's column order when inferring). When given, it always wins: a grouping that
		contradicts self-describing "Output<o>_<c>" names is accepted as-is and reshapes the dataset
		accordingly, silently -- explicit input is never checked against the names it overrides.

	Returns
	-------
	Dataset built from the CSV's "Input*"/"Output*"/"Noise*" columns.

	Raises
	------
	ValueError
		If the CSV has no "TaskID" column, no "Input*" column, or no "Output*" column; if it has some
		"Noise*" columns but not one per "Output*" column; if `output_groups` doesn't have exactly one
		entry per "Output*" column; or if the output groups (explicit or inferred) don't all have the
		same number of channels.
	"""
	df = pl.read_csv(csv_path)

	if "TaskID" not in df.columns:
		raise ValueError("CSV must contain a 'TaskID' column.")
	input_cols = [c for c in df.columns if c.startswith("Input")]
	output_cols = [c for c in df.columns if c.startswith("Output")]
	if not input_cols:
		raise ValueError("CSV must contain at least one 'Input*' column.")
	if not output_cols:
		raise ValueError("CSV must contain at least one 'Output*' column.")

	group_index, O, C = _infer_output_groups(output_cols, output_groups)
	# Reorder so each group's columns are contiguous, in output-index order, so the later columns-axis
	# reshape to (O, C) is valid regardless of how groups were interleaved in the file.
	order = np.argsort(group_index, kind="stable")
	output_cols = [output_cols[i] for i in order]

	noise_cols = [c.replace("Output", "Noise", 1) for c in output_cols]
	present_noise = [c for c in noise_cols if c in df.columns]
	# All or nothing: a file carrying noise for only some channels would otherwise have the rest
	# silently read as "no known noise at all", losing the values it does carry.
	if present_noise and len(present_noise) != len(noise_cols):
		missing = [c for c in noise_cols if c not in df.columns]
		raise ValueError(f"CSV has some 'Noise*' columns but not all; missing {missing}.")
	has_noise = bool(present_noise)

	task_ids = df["TaskID"].to_numpy()
	inputs_all = df.select(input_cols).to_numpy()
	# `to_numpy()` collapses null and NaN to the same float NaN (verified against polars 1.42.1) --
	# the null mask, needed to tell "not this output's axis" from "unobserved", is read separately.
	output_null = df.select([pl.col(c).is_null() for c in output_cols]).to_numpy().reshape(-1, O, C)
	output_vals = df.select(output_cols).to_numpy().reshape(-1, O, C)
	noise_vals = df.select(noise_cols).to_numpy().reshape(-1, O, C) if has_noise else None

	# Stable sort by TaskID groups each task's rows together while preserving their file order within
	# the group -- both are relied on below ("in file order" within a task).
	sort_idx = np.argsort(task_ids, kind="stable")
	task_ids = task_ids[sort_idx]
	inputs_all = inputs_all[sort_idx]
	output_null = output_null[sort_idx]
	output_vals = output_vals[sort_idx]
	if has_noise:
		noise_vals = noise_vals[sort_idx]

	_, counts = np.unique(task_ids, return_counts=True)  # ascending TaskID order, matching the sort above
	T = counts.size
	task_idx = np.repeat(np.arange(T), counts)

	# A row belongs to output o's axis iff at least one of its C cells is non-null there.
	belongs = ~output_null.all(axis=-1)  # (R, O)

	I = len(input_cols)
	single_output_datasets = []
	for o in range(O):
		mask = belongs[:, o]
		t_o = task_idx[mask]
		counts_o = np.bincount(t_o, minlength=T)
		starts_o = np.concatenate([[0], np.cumsum(counts_o)[:-1]])
		pos_o = np.arange(t_o.size) - np.repeat(starts_o, counts_o)
		N_o = int(counts_o.max(initial=0))

		out_inputs = np.full((T, N_o, I), np.nan)
		out_outputs = np.full((T, N_o, C), np.nan)
		out_inputs[t_o, pos_o] = inputs_all[mask]
		out_outputs[t_o, pos_o] = output_vals[mask, o, :]
		out_noise = None
		if has_noise:
			out_noise = np.full((T, N_o, C), np.nan)
			out_noise[t_o, pos_o] = noise_vals[mask, o, :]

		single_output_datasets.append(
			Dataset(
				inputs=jnp.asarray(out_inputs),
				outputs=jnp.asarray(out_outputs),
				known_output_noise=None if out_noise is None else jnp.asarray(out_noise),
			)
		)

	dataset = merge_multioutput_datasets(single_output_datasets)

	# Collapse the leading task axis to 1 if every task ended up holding the exact same input
	# points. Done once, on the merged result, not per output before merging: outputs whose own points
	# happen to be task-uniform could otherwise collapse to different leading-axis sizes and break the
	# concatenation merge_multioutput_datasets does across outputs.
	if dataset.inputs.shape[0] > 1 and _exact_equal(dataset.inputs, dataset.inputs[:1]):
		# `output_ids`' leading axis mirrors `inputs`' own (see `Dataset`), so it collapses with it --
		# `merge_multioutput_datasets` broadcast it from a single row, so it is always task-uniform here.
		output_ids = dataset.output_ids
		if output_ids is not None and output_ids.shape[0] > 1:
			output_ids = output_ids[:1]
		dataset = Dataset(
			inputs=dataset.inputs[:1],
			outputs=dataset.outputs,
			known_output_noise=dataset.known_output_noise,
			output_ids=output_ids,
		)
	return dataset


def merge_multioutput_datasets(datasets: Sequence[Dataset]) -> Dataset:
	"""
	Merge single-output Datasets (e.g. from `load_single_csv`), in output-index order, into one
	multi-output Dataset. Inverse of `split_into_single_output_datasets`.

	Every dataset must describe the same tasks, in the same order -- `load_csv` guarantees this
	(by validating every CSV shares the same set of `TaskID` values) before calling
	`load_single_csv` on each file and this function on the results; `Dataset` itself doesn't carry
	`TaskID` values, so this function can't check or realign them itself.

	Parameters
	----------
	datasets
		One Dataset per output, in output-index order, all describing the same tasks in the same order.

	Returns
	-------
	Merged multi-output Dataset, with `output_ids` marking each row's source dataset.
	"""
	if len(datasets) == 1:
		return datasets[0]

	inputs = jnp.concatenate([d.inputs for d in datasets], axis=1)
	outputs = jnp.concatenate([d.outputs for d in datasets], axis=1)

	if all(d.known_output_noise is None for d in datasets):
		known_output_noise = None
	else:
		known_output_noise = jnp.concatenate(
			[
				d.known_output_noise if d.known_output_noise is not None else jnp.full(d.outputs.shape, jnp.nan)
				for d in datasets
			],
			axis=1,
		)

	# If every output shares the exact same input locations, output_ids is redundant -- collapse
	# back to the compact `None` (isotopic_output_in_tasks) representation, so a dataset that was
	# isotopic before going through `split_into_single_output_datasets`/CSV round-trips back to
	# being isotopic, instead of gaining a spurious per-row output_ids.
	same_N = all(d.inputs.shape[1] == datasets[0].inputs.shape[1] for d in datasets)
	if same_N and all(_exact_equal(d.inputs, datasets[0].inputs) for d in datasets[1:]):
		return Dataset(inputs=datasets[0].inputs, outputs=outputs, known_output_noise=known_output_noise)

	output_ids = jnp.concatenate([jnp.full((d.outputs.shape[1],), o, dtype=int) for o, d in enumerate(datasets)])
	output_ids = jnp.broadcast_to(output_ids, (inputs.shape[0],) + output_ids.shape)  # mirror inputs' own leading axis

	return Dataset(inputs=inputs, outputs=outputs, known_output_noise=known_output_noise, output_ids=output_ids)


def load_csv(csv_path: str | Path | Sequence[str | Path], output_groups: Sequence | None = None) -> Dataset:
	"""
	Read a Dataset from CSV. A single path reads one wide file (see `load_single_csv`); a sequence of
	paths reads one file per output (in output-index order) and merges them (see
	`merge_multioutput_datasets`).

	Parameters
	----------
	csv_path
		Path (single-output dataset) or one path per output, in output-index order.
	output_groups
		Forwarded to `load_single_csv`. Only valid with a single `csv_path` -- a sequence of paths is
		already one output per file.

	Returns
	-------
	Dataset built from the CSV file(s).

	Raises
	------
	ValueError
		If `output_groups` is given together with a sequence `csv_path`, or if `csv_path` is a
		sequence and the files don't all share the same set of `TaskID` values.
	"""
	if isinstance(csv_path, (str, Path)):
		return load_single_csv(csv_path, output_groups)

	if output_groups is not None:
		raise ValueError(
			"output_groups is only valid for a single csv_path, not a sequence of paths (already one output per file)."
		)

	csv_paths = list(csv_path)
	if len(csv_paths) == 1:
		return load_single_csv(csv_paths[0])

	task_id_sets = [set(pl.read_csv(p, columns=["TaskID"])["TaskID"].to_list()) for p in csv_paths]
	if any(s != task_id_sets[0] for s in task_id_sets[1:]):
		raise ValueError(
			"All csv_path files must share the exact same set of TaskIDs to be merged into a multi-output Dataset."
		)

	return merge_multioutput_datasets([load_single_csv(p) for p in csv_paths])
