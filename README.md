# Biohub Python project template
This repo contains a template for starting new Python projects at the Biohub. It adheres to various best practices for organizing and configuring Python packages and also includes the developer tooling and test infrastructure that have become de facto standards within the open-source Python community. See the [Features](#Features) section below for details.

## Organization
The structure of this repo is illustrated below.
```
├── some_package                  # top-level python package directory
│   ├── __init__.py
│   ├── conftest.py               # pytest configuration and test fixtures
│   ├── cli
│   │   ├── __init__.py
│   │   └── some_cli.py           # example CLI module (defined as an entrypoint in pyproject.toml)
│   ├── some_subpackage           # an example subpackage
│   │   ├── __init__.py
│   │   └── some_module.py        # an example submodule
│   └── tests                     # tests as a subpackage of the main package
│       ├── __init__.py
│       └── test_placeholder.py
├── LICENSE
├── MANIFEST.in
├── Makefile
├── README.md
├── pyproject.toml   # all project metadata and configs for dev and build tools
├── setup.cfg        # flake8 config (flake8 does not support pyproject.toml)
└── setup.py         # shim to enable editable pip installs
```

## Usage
To use this template as the basis of a new Python project, follow the steps below.

1. Choose a name for your project and create a repo for it on GitHub under the `czbiohub` organization. By convention, project names and repo names should be the same, and they should be dash-separated (for example, `my-new-project`).

2. Clone this repo into a directory named after your new project:
```sh
git clone git@github.com/czbiohub-sf:python-package-template.git my-new-project
```

3. Update the URL of the remote repo to point to your new repository:
```sh
git remote set-url origin git@github.com:czbiohub-sf/my-new-project
```

4. Change the package name by changing the name of the `some_package` directory *and* the project name in `pyproject.toml`. By convention, package names are identical to project/repo names but with dashes replaced by underscores; for example, `my-new-project` would contain a Python package called `my_new_project`. (Of course, for complex projects with multiple packages, this guideline breaks down).

5. Update all project-specific metadata in `pyproject.toml` (including the package description, authors, classifiers, dependencies, URLs, CLI entrypoints, etc).

6. Install pre-commit hooks (see [below](#pre-commit-hooks) for details):
```sh
pre-commit install
```

7. Install the package in editable mode:
```sh
pip install -e .'[dev]'
```

8. Remove or rename the example CLI and example subpackage.

9. Update the `AUTHORS` file as appropriate (this is especially important for projects that will become public, as this file is referred to in the `LICENSE`).

10. Once you are ready to build a release you can install build dependencies, make a build, and publish on PyPI with:
```sh
make setup-build
make build
make publish
```
Note that the final publishing step requires authentication with an account on [pypi.org](https://pypi.org/).


## Features and design decisions
### Defining package and project metadata
We follow [PEP 621](https://peps.python.org/pep-0621/) and use `pyproject.toml` (instead of `setup.py` or `setup.cfg`) to define the core metadata and dependencies of the Python package *and* to specify the configuration settings for most of the developer tooling included with this template. There are several important advantages of using `pyproject.toml`:

- it is declarative and both human- and machine-readable
- it is forward-looking and build-tool-agnostic (`setup.py` is deprecated and both it and `setup.cfg` are `setuptools`-specific)
- it centralizes as much project configuration as possible in one place, limiting the proliferation of tool-specific config files

Note that `flake8` does not yet support `pyproject.toml`, so we are forced to retain the tool-specific `.flake8` config file.

### Package structure and CLI
This template documents the recommended Python package organization by including a minimal example Python package called `some_package` that includes a mock subpackage (in the directory `some_subpackage`) as well as an example CLI written using `argparse`. This CLI is made accessible at a command-line entrypoint defined in the `projects.scripts` section of `pyproject.toml`. This entrypoint is created whenever the package is installed.

### Code formatting and linting
This template uses [`black`](https://black.readthedocs.io/en/stable/) for code formatting, [`isort`](https://pycqa.github.io/isort/) for organizing imports, and [`flake8`](https://flake8.pycqa.org/en/latest/) for linting. We consider `flake8` to be essential, as it is easy and fast to run and it captures many common typos, bugs, and mistakes, as well as enforces various consensus style guidelines.  `black` and `isort` are opinionated; not all projects use them, and other formatting conventions exist. However, all three tools have become increasingly widely adopted in the Python community and __we strongly recommend using `black`, `isort`, and `flake8` together to ensure that code formatting and style are consistent both across Biohub projects and with the broader open-source community.__

These tools can be run manually via `Makefile` commands (`make lint`) and are also defined as pre-commit hooks that run automatically before each commit (see below). Recommended configuration settings for `black` and `isort` are defined in `pyproject.toml` and for `flake8` in `.flake8`.

### Pre-commit hooks
This template uses `pre-commit` to define and run 'commit hooks'; these are commands that are automatically triggered to run locally (on the developer's machine) whenever a commit is made. The purpose of these hooks is to serve as a first line of defense by ensuring that any code committed to *any* clone of the repo meets certain quality and formatting standards.

This template includes common hooks that both check for minor formatting errors (trailing whitespace, missing end-of-file line returns) and run the formatting and linting tools discussed above (`black`, `isort`, and `flake8`). All of the hooks are defined in `.pre-commit-config.yaml`.

If any of the hooks fail, the commit itself is rejected (but the changes remain staged). Most of the hooks automatically modify the code to correct whatever issues were encountered, but `flake8` violations must be manually fixed by the developer. In either case, the new changes must be staged, then the commit can be attempted a second time (and it should, this time, pass the pre-commit hooks and be accepted).

Note that pre-commit hooks are only a first line of defense; more elaborate checks and testing should be run automatically when a PR is opened or modified (see the 'Testing and CI' section below) and it is a good practice to require that a PR must pass CI before it can be merged.

### Testing and CI
This template is configured to use `pytest` for running tests. Please refer to the [pytest docs](https://docs.pytest.org/en/7.2.x/) for details about how to define tests and test fixtures. A basic CI workflow that is run using GitHub actions is defined in `.github/workflows/CI.yaml` and includes both the pre-commit hooks discussed above, static type checking, and testing with pytest. Tests can also be run locally using `make test`.

### Makefile
This template includes a makefile with a few basic development-related commands to install the package in editable mode, run the linters, and run tests. These commands may be useful as-is but can of course be modified to accomodate per-project constraints. They also serve to document how the developer tools (black, flake8, pylint, pre-commit) are intended to be used.

### Software license
Biohub software projects should be licensed under the standard 3-clause BSD license. A copy of this license is included in this repo. Please be sure to update the `AUTHORS` file as appropriate, as this file is referred to by the license.

## Features *not* included in this template
This template omits, or is agnostic to, several common aspects of software development
that are important for some Python packages, depending on their purpose and scope. Most
prominently, it does not specify a virtual-environment manager (e.g., `conda`, `mamba`,
`venv`, etc). It also does not use static type checking (e.g., `mypy`) and it does not
include tooling to calculate/track test coverage (e.g., `pytest-cov`).
