"""
mimosa-ml: multi-task, multi-cluster Gaussian process regression with heterogeneous sampling.

Import layout
-------------
This module is a pure façade: it holds no definitions, only re-exports, so no submodule ever needs
to import from `mimosa` itself. Submodules import from each other (and from `mimosa.constants`),
never from the package root.

`__all__` below is the supported flat API -- everything needed to build, fit, predict, plot,
save/load and simulate. Names used only to *extend* the library (abstract bases, free numerical
functions, internal block containers) stay in their submodule and are reached through it, e.g.
`mimosa.linalg.cho_factor`, `mimosa.prediction.predict`, `mimosa.grid.GridBuilder`. Each submodule
declares its own `__all__`; anything absent from it is private.
"""

import importlib.metadata

from mimosa.constants import DEFAULT_JITTER, PAD_INDEX
from mimosa.data_structures import (
	Dataset,
	Dimensions,
	ModelConfig,
	DataRemovalConfig,
	Parameters,
	GPParameters,
	ParameterPriors,
	GPDataset,
	Grid,
	Mixture,
	Hyperprior,
	Hyperposterior,
	MultivariateNormal,
)
from mimosa.hyperpost import one_shot_hyperpost
from mimosa.grid import UnionGrid, RegularGrid, KMeansGrid, MultiOutputUnionGrid, MergedGrid
from mimosa.mappings import ExactInputMapper, NearestInputMapper
from mimosa.io import save_csv, load_csv
from mimosa.laplace import (
	IdentityLaplaceApproximator,
	BinomialLaplaceApproximator,
	PoissonLaplaceApproximator,
	ExponentialLaplaceApproximator,
)
from mimosa.mixture import KMeansMixtureInitialiser
from mimosa.models import BasicModel, GPModel
from mimosa.plot import (
	plot_channel,
	plot_task,
	plot_dataset,
	plot_single_cluster_single_channel,
	plot_single_cluster,
	plot_clusters,
	plot_single_task_prediction,
)
from mimosa.prediction import FunctionPredictor, ObservationPredictor, GPPredictor
from mimosa.sampling import sample_gp, mixture_sampler, exact_mixture_sampler
from mimosa.synthetic import (
	generate_data,
	known_noise_kernel,
	build_parameters,
	build_gp_parameters,
	sample_parameters_from_priors,
	RandomDataRemover,
)

__all__ = [
	# constants
	"DEFAULT_JITTER",
	"PAD_INDEX",
	# data structures
	"Dataset",
	"GPDataset",
	"Dimensions",
	"ModelConfig",
	"DataRemovalConfig",
	"Parameters",
	"GPParameters",
	"ParameterPriors",
	"Grid",
	"Mixture",
	"Hyperprior",
	"Hyperposterior",
	"MultivariateNormal",
	# grids and input mappers
	"UnionGrid",
	"RegularGrid",
	"KMeansGrid",
	"MultiOutputUnionGrid",
	"MergedGrid",
	"ExactInputMapper",
	"NearestInputMapper",
	# io
	"save_csv",
	"load_csv",
	# likelihood approximations
	"IdentityLaplaceApproximator",
	"BinomialLaplaceApproximator",
	"PoissonLaplaceApproximator",
	"ExponentialLaplaceApproximator",
	# models and prediction
	"BasicModel",
	"GPModel",
	"KMeansMixtureInitialiser",
	"one_shot_hyperpost",
	"FunctionPredictor",
	"ObservationPredictor",
	"GPPredictor",
	# plotting
	"plot_channel",
	"plot_task",
	"plot_dataset",
	"plot_single_cluster_single_channel",
	"plot_single_cluster",
	"plot_clusters",
	"plot_single_task_prediction",
	# sampling and simulation
	"sample_gp",
	"mixture_sampler",
	"exact_mixture_sampler",
	"generate_data",
	"known_noise_kernel",
	"build_parameters",
	"build_gp_parameters",
	"sample_parameters_from_priors",
	"RandomDataRemover",
]

__version__ = importlib.metadata.version("mimosa-ml")
