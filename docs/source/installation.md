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

grassp needs Python 3.11 or newer, and is tested on 3.11 through 3.14.

## Optional extras

Some functionality needs dependencies that are too heavy — or too slow to support a new Python
release — to install by default:

```bash
pip install "grassp[proloc]"    # gr.io.read_prolocdata, for pRoloc's .rda/.rds files
pip install "grassp[ccompass]"  # gr.tl.ccompass, the C-COMPASS neural-network classifier
```

```{note}
`grassp[ccompass]` requires **Python <= 3.13**. It pulls in tensorflow, which publishes no wheels
for 3.14, so on 3.14 the install fails to resolve with a message about `cp314`. Everything else in
grassp works on 3.14 — use 3.13 if you need C-COMPASS.
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
