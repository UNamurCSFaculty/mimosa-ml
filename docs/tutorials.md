# How to use these tutorials

The tutorials go from the surface down: **level 1** is what every user needs, **level 2** is what
the model can do, **level 3** is how to tailor and scale it.

## Three ways to read them

Every tutorial is a single script in `docs/examples`, written in jupytext's `py:percent` format.
That means the same file can be:

* **read as a page**, here in the documentation, with all outputs and figures rendered;
* **run cell-by-cell**, as a notebook — use the 🚀 launch button at the top of any tutorial page to
  open it in Colab;
* **executed as a plain Python script**, top to bottom, with `python docs/examples/<name>.py`.

Running them locally needs the docs dependencies for plotting; a
`pip install mimosa-ml matplotlib` is enough for most of them.

## Where to start

If you have never used Mimosa, read {doc}`examples/basic_example` first — it 
walks the whole pipeline end to end, and every later one assumes it. From there:

* unsure *why* the model is built this way → {doc}`examples/concepts_behind_mimosa`
* unsure which flags to set → {doc}`examples/configurations`
* ready to load your own CSV → {doc}`examples/run_your_data`

Levels 2 and 3 are independent of each other: pick the page that matches the problem you have,
rather than reading them in order.

```{tip}
The tutorials assume you are comfortable with [JAX's basics](https://docs.jax.dev/en/latest/101/index.html#) — immutable arrays, PRNG keys, and
`jit`/`vmap`/`grad`. 
```
