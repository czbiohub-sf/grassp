# Contributing

We welcome contributions to **grassp**! This guide will help you get started with contributing to the project.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/yourusername/grassp.git
   cd grassp
   ```

3. **Create a development environment:**

This installs the package with the `-e` option and development and documentation dependencies as well as pre-commit hooks.

   ```bash
   make setup-develop
   ```

4. **If you are touching the pRoloc bridge or building the docs, also create the R environment:**

   The R half of the bridge (`r/grasspio`) needs a Bioconductor stack that is not
   pip-installable, so it lives in a conda environment declared by
   [r-environment.yml](https://github.com/czbiohub-sf/grassp/blob/main/r-environment.yml):

   ```bash
   make setup-r
   ```

   That creates the `grassp-r` environment, installs `grasspio` into it, and registers the `ir`
   Jupyter kernel. The kernel is the part worth knowing about: **the docs build executes
   `docs/source/tutorials/proloc_r_tutorial.Rmd` with it**. Without it, build with
   `GRASSP_DOCS_SKIP_R=1`, which renders that page without outputs instead of failing. Run
   `make test-r` for the grasspio test suite.

## Testing

1. Tests can be added to [grassp/tests](https://github.com/czbiohub-sf/grassp/blob/main/grassp/tests)
2. Run the tests:

   ```bash
   pytest grassp/tests
   ```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use type hints where appropriate.
- Add docstrings to all public functions.
- Use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

## Documentation

- Update docstrings for any changed functions.
- Update tutorials if adding new features.
- Build docs locally to check formatting:

   ```bash
   make docs                       # from the repo root; writes docs/build/html
   GRASSP_DOCS_SKIP_R=1 make docs  # without the R environment
   ```

**Every tutorial is executed as part of the build**, except `ccompass_tutorial`, which ships
pre-rendered because it needs TensorFlow. So building the docs needs the `.[docs]` extra, network
access (the tutorials download datasets from the grassp portal), and — for the R tutorial — the
conda environment from `make setup-r`.

A notebook that fails to execute **fails the build**, with the failing cell's traceback in the
log.

The docs are built on every pull request and published to
<https://czbiohub-sf.github.io/grassp/> on every push to `main`, by
[docs.yml](https://github.com/czbiohub-sf/grassp/blob/main/.github/workflows/docs.yml). There is
nothing to upload by hand.

## Adding New public functions

When adding support for new functionality:

1. Add the code for the function in the most appropriate submodule
   1. `preprocessing` is for data filtering, enrichment, qc
   2. `tools` is for dimensionality reduction, classification, differential testing
   3. `plotting` is for visualization
   4. `io` is for reading from diffferent sources and writing outputs
2. Make sure you follow the recommended **Code Style** section below and describe the function and parameters accurately with a docstring
3. Expose the function as public by importing it in the `__init__.py` file of the submodule
4. Add the function to the documentation by adding it to the [api documentation](https://github.com/czbiohub-sf/grassp/blob/main/docs/source/api).
5. Build the documentation (**Documentation** below) to check proper formatting.
6. Add tests and run them (**Testing** above)
7. Create a PR against `main`
