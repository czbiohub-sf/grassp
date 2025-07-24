# Installation

**grassp** can be installed via [pip](https://pypi.org/project/pip/) from [PyPI](https://pypi.org/project/grassp/) with:
```bash
pip install grassp
```

We recommend to install **grassp** in a conda environment (see how to [install conda](https://www.anaconda.com/docs/getting-started/miniconda/install)):

```bash
conda create -n grassp python=3.12
conda activate grassp
pip install grassp
```

## Install curent development version

To install the most up-to-date version of grassp, clone the repository and install from source

```{code-block} bash
git clone https://github.com/czbiohub-sf/grassp.git
cd grassp
conda create -n grassp python=3.12
conda activate grassp
pip install -e .
```
