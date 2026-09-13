# Contributing and citation

## Contributing

At this stage of development, we are not yet accepting contributions to the codebase. The API is
still moving quickly, and merging external changes against it would cost more than it saves.

That said, we are interested in hearing from you:

* **Bug reports** are welcome — open an issue on
  [GitHub](https://github.com/SimLej18/mimosa-ml/issues). A minimal script, the `Dimensions` and
  `ModelConfig` you used, and the versions of `mimosa-ml` and `jax` are usually enough to reproduce.
* **Examples from your field.** If your research provides an interesting application of Mimosa,
  consider publishing a toy example in this documentation. Tutorials are plain scripts in
  `docs/examples` written in jupytext's `py:percent` format, so a working script is most of the
  work.
* **Anything else** — reach out by email or through the GitHub repository.

## Authors

Mimosa is primarily developed by the *Magma Task Force*:

* [Arthur Leroy](https://arthur-leroy.netlify.app/), researcher at Paris Saclay and INRAe (FR),
  main author of the original [MagmaClustR](https://arthurleroy.github.io/MagmaClustR/) and
  coordinator of the Task Force.
* [Simon Lejoly](https://researchportal.unamur.be/fr/persons/slejoly/), PhD student at UNamur (BE),
  main developer of the package and author of [Kernax](https://github.com/SimLej18/kernax-ml).
* Alexia Grenouillat, PhD student at INSA Toulouse (FR), working on multi-output correlation
  discovery.
* Térence Viellard, PhD student at INRAe (FR), working on sparse approximations and scalability.

## Citation

If you use Mimosa in your research, please consider citing:

```{note}
Citation to be added once the paper is published.
```

The theoretical foundation is the MagmaClust framework, which can be cited in the meantime:

> Leroy, A., Latouche, P., Guedj, B., & Gey, S. (2023). *Cluster-Specific Predictions with
> Multi-Task Gaussian Processes.* Journal of Machine Learning Research, 24(5), 1–49.
> <https://jmlr.org/papers/v24/20-1321.html>

## License

Mimosa is distributed under the [MIT License](https://opensource.org/licenses/MIT). See `LICENSE`
in the repository for the full text.
