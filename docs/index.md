# mimosa-ml

**MIMOSA: Multi-Input Multi-Output Sample Analysis** — a fully-featured *multi-task Gaussian
process* framework for analysing functional data, built on
[MagmaClust](https://jmlr.org/papers/v24/20-1321.html) and powered by JAX.

Mimosa fits many related samples jointly instead of one at a time, so sparsely observed samples
borrow strength from the rest. It handles unaligned sampling grids, clusters samples, learns
correlations between outputs, and returns probabilistic predictions with uncertainty — on CPU, GPU
or TPU, with full `jit`/`vmap`/`grad` compatibility.

```bash
pip install mimosa-ml
```

New here? {doc}`installation` → {doc}`getting_started` → {doc}`sharp_bits`, then pick a tutorial.

```{toctree}
:maxdepth: 1
:caption: 🚀 Start here

installation
getting_started
🔪 The sharp bits 🔪 <sharp_bits>
```

```{toctree}
:maxdepth: 2
:caption: 📖 Tutorials

How to use these tutorials <tutorials>
Level 0 — predictions in 3 lines <examples/level0/predictions_in_3_lines>
tutorials_level1
tutorials_level2
tutorials_level3
```

```{toctree}
:maxdepth: 1
:caption: 🔍 Reference

API reference <api>
faq
troubleshooting
contributing
```
