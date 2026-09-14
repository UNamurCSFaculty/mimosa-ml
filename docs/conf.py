from pathlib import Path

from jupyter_cache import get_cache

project = "mimosa-ml"
author = "Magma Task Force: A. Leroy, S. Lejoly, A. Grenouillat, T. Viellard"
copyright = "2026, Magma Task Force"

extensions = [
    "myst_nb",
    "sphinx_togglebutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]

# jaxtyping annotations ("Float[Array, '*B FN']") are informative but turn every rendered signature
# into an unreadable wall. Moving them into the parameter descriptions keeps the signature readable
# while still showing each shape next to the prose that explains it. `documented_params` restricts
# this to parameters the docstring already lists, and keeps it away from the return: the numpy-style
# "Returns" sections here hold prose rather than a type.
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"

# Render defaults as they are written in the source. Without this autodoc evaluates them, so
# `jitter=DEFAULT_JITTER` prints as `jitter=Array(1.e-08, dtype=float32, weak_type=True)`.
autodoc_preserve_defaults = True

autodoc_member_order = "bysource"

# Napoleon routes a typeless numpy-style "Returns" section to `:rtype:`, which Sphinx then labels
# "Return type" — wrong for the prose descriptions used here.
napoleon_use_rtype = False

# jupyter-cache creates its directory and SQLite database lazily, on first access. Under
# `sphinx-build -j auto` (Read the Docs' default command) several workers do that at the same time
# and the build breaks (FileExistsError, then "table settings already exists"). Fixing the path and
# initialising the cache here, in the parent process, gets it done before the fork.
nb_execution_cache_path = str(Path(__file__).parent / "_build" / ".jupyter_cache")
get_cache(nb_execution_cache_path).db

nb_execution_mode = "cache"
nb_execution_timeout = 600          # generous: JIT compilation plus Cholesky factorisations
nb_execution_raise_on_error = True  # a broken example fails the build

# jax-tqdm's progress bar (and its TqdmWarning when ipywidgets isn't installed) goes to stderr,
# which is of no use in a static page.
nb_output_stderr = "remove"

html_theme = "sphinx_book_theme"

# Without this, Sphinx titles every page "mimosa-ml <version> documentation".
html_title = "mimosa-ml"

# The wide logo replaces the title text in the sidebar header; the square one is the tab icon.
html_logo = "images/logo_no_bg.png"
html_favicon = "images/logo_square.png"

html_static_path = ["_static"]
html_css_files = ["custom.css"]  # only to keep the logo legible in dark mode; see the file

html_theme_options = {
    "repository_url": "https://github.com/UNamurCSFaculty/mimosa-ml",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "launch_buttons": {"colab_url": "https://colab.research.google.com"},
}

# `basic_mo_example` is kept as a runnable script but is not part of the documented tour.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "examples/unpublished/*"]

nb_execution_excludepatterns = [
]