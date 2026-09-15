# FAQ

Questions about what Mimosa is and how to model with it. For things that go *wrong*, see
{doc}`troubleshooting`.

## What is Mimosa for?

Fitting many related functional samples *jointly* rather than one at a time. If you have dozens or
hundreds of curves that share structure — repeated measurements, individuals in a cohort, sensors,
time series per unit — a multi-task GP borrows strength across them, so a sparsely observed sample
is informed by the others. Mimosa builds on the
[MagmaClust framework](https://jmlr.org/papers/v24/20-1321.html), adding clustering,
multi-dimensional inputs and outputs, and a modular JAX implementation.

## Why is it called `mimosa-ml` on PyPI but `mimosa` when I import it?

The name `mimosa` was already taken on PyPI. Install `mimosa-ml`, import `mimosa`.

## Do my tasks have to be observed at the same input locations?

No — that is a large part of the point. Set `isotopic_tasks=False` in `ModelConfig` and each task
carries its own inputs, with its own number of missing points. The grid the mean-processes live on
is built separately, usually with `UnionGrid(dataset.inputs)`.

## How do I choose the number of clusters `K`?

Methods to determine number of clusters in a dataset automatically all make various kinds of
asumptions. There is no "one-fit-all" method for this. Rather than implement a lot of methods,
we leave it to the user to apply its own model selection method.

Usually, fitting for a few values of `K` and comparing the results is enough. Too many clusters
often lead to some clusters getting empty or containing only one outlier.

You can also perform model selection using metrics specific to the task you are trying to solve.
An example of this is given in {doc}`examples/level2/turnip_example`.

## Which `ModelConfig` flags should I use?

The defaults share hyperparameters across tasks, clusters and channels, which is the most
constrained and most stable setting. Relax a flag when you believe that dimension genuinely differs
— per-task noise, say, or per-cluster kernels. The most reliable way to build intuition is to
generate synthetic data with `generate_data` under different configurations until it resembles your
real dataset. Each flag is described in {doc}`examples/level1/configurations`.

It is usually ill-advised to have distinct hyperparameters in all dimensions, as a single
unstable task can take down the whole training by itself.

## Can I run on a GPU or TPU?

Yes — install the matching jax build and mimosa follows. Nothing in the API changes. For splitting
a fit across devices, see {doc}`examples/level3/distributed_training`.

## Can I use my own kernels?

Yes. Kernels and mean functions come from
[Kernax](https://github.com/UNamurCSFaculty/kernax-ml) and compose with `*` and `+`, so any kernax kernel
(Matérn, periodic, linear, …) can be dropped into `Parameters`. Beyond that, Mimosa is layered: the
`Model` API covers the general case, and you can assemble your own training loop from the same
components — see {doc}`examples/level3/custom_training_loop`.

## Can I fit non-Gaussian data?

Experimentally. Binary, count and duration observations can be Laplace-matched into Gaussian
pseudo-observations with the `*LaplaceApproximator` classes, leaving the rest of the pipeline
unchanged. See {doc}`examples/level2/binary_classif_example`.

## How large a dataset can Mimosa handle?

A GP costs cubically in the number of points it factorises, so the grid size `G` is what usually
binds, not the number of tasks. Tasks are batched with `vmap` and scale well; grids do not. When a
full grid becomes too large, use a coarser one (`RegularGrid`, `KMeansGrid`) or move to
{doc}`examples/level3/sparse_approximations`. Training on minibatches of tasks is covered in
{doc}`examples/level3/stochastic_learning`.

## Is the API stable?

Not yet — Mimosa is `v0.5.0-alpha`. Version-lock your dependency and read `CHANGELOG.md` when
upgrading.

## How do I cite Mimosa, and can I contribute?

See {doc}`contributing`.
