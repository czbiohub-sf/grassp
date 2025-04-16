import os
import sys
from pathlib import Path

project = "grassp"
copyright = "2025, Max Frank"
author = "Max Frank"
release = "0.1"

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))


# Configuration file for the Sphinx documentation builder

project = "grassp"
copyright = "2025, Max Frank"
author = "Max Frank"

# Add any Sphinx extension module names here
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "scanpydoc",
    "myst_parser",
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
html_logo = "_static/img/grassp_logo.png"
issues_github_path = "czbiohub-sf/grassp"
html_show_sphinx = False
