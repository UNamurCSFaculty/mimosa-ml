# Installation

Mimosa is distributed through PyPI:

```bash
pip install mimosa-ml
```

Mimosa requires Python 3.10+ although a more recent version is encouraged.

```{note}
The PyPI distribution is `mimosa-ml`; the package you import is `mimosa`.
```

## What gets installed

Mimosa pulls in its own stack: `jax`, `equinox` and `optimistix` for the computation and
optimisation, [`kernax-ml`](https://github.com/UNamurCSFaculty/kernax-ml) for every kernel and mean
function, plus `polars`, `numpy` and `matplotlib` for I/O and plotting.

## Importing

The supported API can be imported from the package root:

```python
from mimosa import Dataset, UnionGrid, BasicModel, plot_dataset
```

Extension points — abstract bases, free numerical functions — stay in their submodule and are
imported from there:

```python
from mimosa.hyperpost import Hyperpost
from mimosa.linalg import cho_factor
```

## GPU and TPU

Mimosa runs wherever jax runs. To train on an accelerator, install the matching jax build
(see [jax's installation guide](https://docs.jax.dev/en/latest/installation.html)); mimosa itself
needs no change.

```{warning}
Mimosa is in early development. Features work in most cases, but the API is still
subject to change — version-lock your dependency for research or production use, and read
`CHANGELOG.md` when upgrading.
```
