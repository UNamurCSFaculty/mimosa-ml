# 🔪 The sharp bits 🔪

Things that bite newcomers. Most of them are inherited from Gaussian processes or from JAX rather
than being specific to Mimosa.

## Enable 64-bit precision

GPs invert covariance matrices, and float32 is rarely enough to do that stably. Turn on x64
*before* importing jax, kernax or mimosa:

```python
import jax
jax.config.update("jax_enable_x64", True)
# ...all other imports
```

## NaNs usually mean an ill-conditioned matrix

NaN in the hyperposterior or the likelihood, an optimisation that does not converge, odd-looking
mean-processes or samples — these are nearly always numerical, not logical. Raise the jitter added
to the diagonal:

```python
import jax.numpy as jnp

model = BasicModel(prng_key=key, n_clusters=K, jitter=jnp.asarray(1e-5))
```

The default is `1e-8` in every component. Individual components accept their own `jitter` if you
only need to loosen one.

You can also increase the noise value in initial parameters.

## Everything is functional

Mimosa is JAX all the way down, which means:

* **Arrays are immutable.** There is no in-place update; every operation returns a new array.
* **Nothing mutates the model.** `model.fit` *returns* the fitted hyperposterior, mixture and
  parameters, it doesn't store results.
* **Randomness is explicit.** You pass a `jax.random.PRNGKey` and split it yourself. Reusing a key
  reproduces the same draw, which is a feature for reproducibility.

```python
key, subkey = jr.split(key)  # split before each consuming call
```

## The first call is slow

`jit` compiles on first execution for a given set of shapes. Changing `Dimensions`, a `ModelConfig`
flag, or the structure of your kernels triggers a fresh compilation, so an experiment that sweeps
configurations pays that cost repeatedly.

## Dimensions and ModelConfig must match the data

These two objects are what the entire pipeline shapes itself around. A `T`, `C` or `O` that
disagrees with the dataset does not fail cleanly at the entry point — it surfaces as a shape error
deep inside a `vmap`, often pointing at a kernel rather than at the mismatch.

Equally, `fit` expects parameters already batched to those shapes. Build them with
`build_parameters(base_params, dims, config)`; passing the unbatched "base" parameters straight in
raises `Invalid input dimensions` from the kernel.

## New to JAX?

Read [JAX 101](https://docs.jax.dev/en/latest/101/index.html) first. Immutability, PRNG keys and the
`jit`/`vmap`/`grad` transformations are assumed everywhere in this documentation and in Mimosa's
own API.
