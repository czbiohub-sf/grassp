# Configuration file for the Sphinx documentation builder

import os
import sys

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

project = "grassp"
copyright = "2025, Max Frank"
author = "Max Frank"
try:
    release = version("grassp")
except PackageNotFoundError:  # building from a checkout that was never installed
    release = "unknown"
version = ".".join(release.split(".")[:2])

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "scanpydoc",
    "myst_nb",  # enable notebook execution and rendering (includes myst_parser)
]

modindex_common_prefix = ["grassp."]
autodoc_typehints = "description"  # Show type hints in the description
autodoc_typehints_format = "short"  # Use short form (e.g., List instead of typing.List)
python_use_unqualified_type_names = True  # Remove module names from type hints
api_dir = HERE / "api"  # function_images
add_module_names = False

# Add any paths that contain templates here
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# The theme is sphinx-book-theme, with patches for readthedocs-sphinx-search
html_theme = "scanpydoc"
html_theme_options = dict(
    use_repository_button=True,
    repository_url="https://github.com/czbiohub-sf/grassp",
    repository_branch="main",
    navigation_with_keys=False,  # https://github.com/pydata/pydata-sphinx-theme/issues/1492
)
html_logo = "_static/img/logo.svg"
# Canonical URL; .github/workflows/docs.yml publishes the site there.
html_baseurl = "https://czbiohub-sf.github.io/grassp/"
issues_github_path = "czbiohub-sf/grassp"
html_show_sphinx = False
intersphinx_mapping = dict(
    python=("https://docs.python.org/3", None),
    numpy=("https://numpy.org/doc/stable", None),
    pandas=("https://pandas.pydata.org/pandas-docs/stable", None),
    anndata=("https://anndata.readthedocs.io/en/stable/", None),
    scanpy=("https://scanpy.readthedocs.io/en/stable/", None),
    scipy=("https://docs.scipy.org/doc/scipy", None),
    protdata=("https://protdata.sf.czbiohub.org/", None),
)


# Execute notebooks only when out-of-date and set timeout
nb_execution_mode = "cache"  # or "force" to always run, "off" to skip
nb_execution_timeout = 600  # seconds; the R tutorial downloads a dataset before it can start
# The C-COMPASS tutorial needs the optional tensorflow-backed extra and takes minutes to
# train, so it is shipped pre-rendered and excluded from build-time execution.
nb_execution_excludepatterns = ["**/ccompass_tutorial.ipynb"]
# A notebook that fails to execute fails the build. myst-nb's default is to log a warning, embed
# the traceback in the page and exit 0 -- which is how the diffusion tutorial shipped with five
# failing cells.
nb_execution_raise_on_error = True
# Put the failing cell's traceback in the log, not just the notebook's name.
nb_execution_show_tb = True

# R Markdown is a notebook format too. jupytext converts .Rmd on the fly, so myst-nb executes
# it with the `ir` kernel exactly as it executes the .ipynb tutorials -- and because the .Rmd
# *is* the page source, the theme's download button hands readers the runnable original rather
# than a rendered copy.
#
# Building the R tutorial therefore needs R with IRkernel registered as the `ir` kernelspec,
# plus pRoloc/pRolocdata/grasspio -- see `make setup-r`. Without them that page fails to
# execute, which fails the build (nb_execution_raise_on_error above).
nb_custom_formats = {".Rmd": ["jupytext.reads", {"fmt": "Rmd"}]}

# Without an R kernel, set GRASSP_DOCS_SKIP_R=1: the R tutorial is then rendered from its source,
# without outputs, instead of failing the build.
if os.environ.get("GRASSP_DOCS_SKIP_R"):
    nb_execution_excludepatterns.append("**/*.Rmd")
