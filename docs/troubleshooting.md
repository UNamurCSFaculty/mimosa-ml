# Troubleshooting

Symptoms and what usually causes them. Most of these are covered in more depth in
{doc}`sharp_bits`.

## My fit produces NaNs, or does not converge

Almost always numerical rather than logical. In order:

1. Enable 64-bit precision, *before* any jax/kernax/mimosa import:
   `jax.config.update("jax_enable_x64", True)`.
2. Raise the jitter added to the diagonal: `BasicModel(..., jitter=jnp.asarray(1e-5))`. The default
   is `1e-8` in every component.
3. Check your initial hyperparameters are on a sensible scale for your data — a length-scale far
   smaller than the spacing of your inputs makes the covariance matrix effectively diagonal and
   badly conditioned.

## `Invalid input dimensions: x1 has shape (T, N, I)`

Raised from the kernel when `fit` receives parameters that have not been batched to the shapes
`Dimensions` and `ModelConfig` imply. Build them first:

```python
init_params = build_parameters(base_params, dims, config)
```

`generate_data` already returns batched parameters as its fifth value, so that path needs no extra
call.

## A shape error deep inside a `vmap`

Usually `Dimensions` disagreeing with the dataset — a `T`, `C` or `O` that does not match what was
loaded or generated. The error surfaces wherever the mismatch first reaches an operation, often in
a kernel, rather than at the point where the dimensions were declared. Check `dims` against the
dataset's own shapes before looking at the kernels.

## Why did nothing change after I called `model.fit`?

`fit` returns the fitted hyperposterior, mixture and parameters; it does not mutate the model.
Capture the return values:

```python
hyperposterior, mixture, fitted_params = model.fit(...)
```

This is standard JAX/Equinox semantics — nothing in Mimosa mutates in place.

## Why is the first call so slow?

`jit` compilation, paid once per distinct set of shapes. Changing `Dimensions`, a `ModelConfig`
flag, or the structure of your kernels forces a recompile, so a configuration sweep pays it
repeatedly. Time the *second* call, not the first.

## My samples or predictions look wrong, but nothing errored

Check the fit before the prediction: plot the fitted mean-processes over the data with
`plot_dataset` and `plot_clusters`. Mean-processes that sit flat or far from the observations point
at an optimisation that did not converge (see above) rather than at the prediction step.

If the clusters look wrong specifically, try a different `K` — too many clusters typically leaves
some empty or holding a single outlier.

## A cluster came out empty

Usually too many clusters for the data. Refit with a smaller `K`. This is expected behaviour rather
than a failure: the mixture simply assigns no task to that mean-process.

## The progress bar prints warnings about `ipywidgets`

Cosmetic. `jax-tqdm` writes its progress bar to stderr and warns when `ipywidgets` is missing in a
notebook. Install `ipywidgets`, or ignore it — the documentation build drops stderr for this
reason.
