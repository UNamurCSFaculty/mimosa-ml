# API reference

Generated from the package's docstrings.

Mimosa's root module is a façade: it re-exports the supported flat API and defines nothing itself.
Everything below is grouped by the submodule it actually lives in. Names absent from a submodule's
`__all__` are private and are not documented here.

## Data structures

```{eval-rst}
.. automodule:: mimosa.data_structures
   :members:
```

## Models

```{eval-rst}
.. automodule:: mimosa.models
   :members:
```

## Hyperposterior

```{eval-rst}
.. automodule:: mimosa.hyperpost
   :members:
```

## Mixture

```{eval-rst}
.. automodule:: mimosa.mixture
   :members:
```

## Prediction

```{eval-rst}
.. automodule:: mimosa.prediction
   :members:
```

## Grids and input mappings

```{eval-rst}
.. automodule:: mimosa.grid
   :members:

.. automodule:: mimosa.mappings
   :members:
```

## Likelihood approximations

```{eval-rst}
.. automodule:: mimosa.laplace
   :members:
```

## Simulation

```{eval-rst}
.. automodule:: mimosa.synthetic
   :members:

.. automodule:: mimosa.sampling
   :members:
```

## I/O

```{eval-rst}
.. automodule:: mimosa.io
   :members:
```

## Plotting

```{eval-rst}
.. automodule:: mimosa.plot
   :members:
```

## Numerical internals

Building blocks for custom training loops and new components.

```{eval-rst}
.. automodule:: mimosa.linalg
   :members:

.. automodule:: mimosa.nll
   :members:

.. automodule:: mimosa.optimisers
   :members:

.. automodule:: mimosa.kmeans
   :members:

.. automodule:: mimosa.constants
   :members:
```
