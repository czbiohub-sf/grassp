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

## Testing

1. Tests can be added to [grassp/tests](grassp/tests)
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
   cd docs
   # Optional (if running into errors): make clean
   make html
   ```

## Adding New public functions

When adding support for new functionality:

1. Add the code for the function in the most appropriate submodule
   1. `preprocessing` is for data filtering, enrichment, qc
   2. `tools` is for dimensionality reduction, classification, differential testing
   3. `plotting` is for visualization
   4. `io` is for reading from diffferent sources and writing outputs
2. Make sure you follow the recommended [code style](#code-style) and describe the function and parameters accurately with a docstring
3. Expose the function as public by importing it in the `__init__.py` file of the submodule
4. Add the function to the documentation by adding it to the [api documentation](docs/source/api).
5. [Build the documentation](#documentation) to check proper formatting.
6. Add tests and [run them](#testing)
7. Create a PR against `main`
