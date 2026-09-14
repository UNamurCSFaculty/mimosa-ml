"""
Sample from multivariate normal distributions and from mixtures of them (used to sample Gaussian
process realisations).
"""

import jax.random as jr
import jax.numpy as jnp
from jax import Array, vmap
from jaxtyping import Float, Int

from mimosa.constants import DEFAULT_JITTER
from mimosa.data_structures import MultivariateNormal

__all__ = ["sample_gp", "mixture_sampler", "exact_mixture_sampler"]


def sample_gp(key: Array, mvn: MultivariateNormal, jitter: Array = DEFAULT_JITTER) -> Float[Array, "... N"]:
	"""
	Sample from a multivariate normal distribution, with jitter added to the covariance's
	diagonal for numerical stability.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	mvn
		Distribution to sample from: mean of shape `(..., N)` and covariance of shape `(..., N, N)`.
	jitter
		Diagonal jitter added to the covariance before sampling.

	Returns
	-------
	Sampled values. Shape `(..., N)`.
	"""
	cov = mvn.covariance + jitter * jnp.eye(mvn.covariance.shape[-1])
	return jr.multivariate_normal(key, mvn.mean, cov)


def mixture_sampler(
	key: Array,
	mvn: MultivariateNormal,
	coefficients: Float[Array, "K"],
	jitter: Array = DEFAULT_JITTER,
) -> tuple[Float[Array, "N"], Int[Array, ""]]:
	"""
	Draw one sample from a mixture of multivariate normals.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	mvn
		The `K` mixture components, batched over the leading axis: mean of shape `(K, ..., N)` and
		covariance of shape `(K, ..., N, N)`. A `Hyperprior` or `Hyperposterior` is such a batch.
	coefficients
		Mixture coefficient of each component, summing to 1. `Mixture.coefficients` is such a vector.
	jitter
		Diagonal jitter added to the chosen component's covariance before sampling.

	Returns
	-------
	Sampled values of shape `(N,)`, and the index of the component they were drawn from.

	Notes
	-----
	`vmap` this over a batch of keys to obtain several samples of the multi-modal distribution. The
	draws are i.i.d., so their cluster proportions match `coefficients` in expectation only; see
	`exact_mixture_sampler` for guaranteed proportions.
	"""
	key_component, key_sample = jr.split(key)
	component_id = jr.categorical(key_component, jnp.log(coefficients))
	return sample_gp(key_sample, mvn[component_id], jitter), component_id


def exact_mixture_sampler(
	key: Array,
	mvn: MultivariateNormal,
	coefficients: Float[Array, "K"],
	n_samples: int,
	jitter: Array = DEFAULT_JITTER,
) -> tuple[Float[Array, "S N"], Int[Array, "S"]]:
	"""
	Draw `n_samples` samples from a mixture of multivariate normals, with the cluster coefficients
	guaranteed rather than sampled.

	Parameters
	----------
	key
		`jax.random` PRNG key.
	mvn
		The `K` mixture components, batched over the leading axis: mean of shape `(K, ..., N)` and
		covariance of shape `(K, ..., N, N)`. A `Hyperprior` or `Hyperposterior` is such a batch.
	coefficients
		Mixture coefficient of each component, summing to 1. `Mixture.coefficients` is such a vector.
	n_samples
		Number of samples to draw. Static: it is the leading shape of the result.
	jitter
		Diagonal jitter added to each covariance before sampling.

	Returns
	-------
	Sampled values of shape `(n_samples, N)`, and the component each was drawn from, shape
	`(n_samples,)`.
	"""
	probes = (jnp.arange(n_samples) + 0.5) / n_samples
	# Clipped because `cumsum` ends a float epsilon below 1.0 for some coefficients, which would
	# send the last probe to the out-of-range index `K`.
	components = jnp.clip(jnp.searchsorted(jnp.cumsum(coefficients), probes), 0, coefficients.shape[0] - 1)
	samples = vmap(lambda k, c: sample_gp(k, mvn[c], jitter))(jr.split(key, n_samples), components)
	return samples, components
