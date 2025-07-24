[![PyPI - Version](https://img.shields.io/pypi/v/grassp)](https://img.shields.io/pypi/v/grassp)
[![CI](https://github.com/czbiohub-sf/grassp/actions/workflows/test.yml/badge.svg)](https://github.com/czbiohub-sf/grassp/actions/workflows/test.yml)
[![docs online](https://img.shields.io/badge/docs-online-blue)](https://public.czbiohub.org/comp.bio/grassp/)


[anndata]: https://anndata.readthedocs.io
[scanpy]: https://scanpy.readthedocs.io
[protdata]: https://protdata.sf.czbiohub.org
[documentation]: https://public.czbiohub.org/comp.bio/grassp


<p align="center">
  <img src="docs/source/_static/img/logo.svg" alt="grassp logo" width="300"/>
</p>

The **grassp** (**GR**aph-based **A**nalysis of **S**ubcellular/**S**patial **P**roteomics) python module enables fast, flexible and scalable analysis of subcellular proteomics datasets.

It uses the [anndata][] format to store mass-spec data and analysis results and [scanpy][] for many of the dimensionality reduction and visualization functions.

**grassp** enables

- Reading the ouput format of most mass-spectrometry search engines (using [protdata][])
- Calculating subcellular enrichment profiles of proteins for different experimental protocols
- Annotating the subcellular location of proteins in an unsupervised and semi-supervised manner
- Detecting proteins at the interface of organelles
- Detecting multi-localizing proteins (work in progress)
- Detecting re-localizing proteins between conditions (work in progress)
- Combining multiple subcellular proteomics datasets
- Assessing subcellular resolution
- Finding the optimal experimental design for future experiments based on simulations
- Integration of multiple modalities (e.g. Lipidomics) (work in progress)

Please refer to the [documentation] for reference to individual functions and [tutorials](https://public.czbiohub.org/comp.bio/grassp/tutorials.html).

## Installation

grassp can be installed via [pip](https://pypi.org/project/pip/) from [PyPI](https://pypi.org/project/grassp/) with:
```
pip install grassp
```
For details on installation, please see the [install section of the documentation](https://public.czbiohub.org/comp.bio/grassp/installation.html)

## Authors

**grassp** is created and maintained by the [Computational Biology Platform](https://www.czbiohub.org/comp-biology/) at the [Chan Zuckerberg Biohub San Francisco](https://www.czbiohub.org/sf/). For details, see the [Contributors page][documentation/contributors.html].

To get in touch please use the [GihHub issues](https://github.com/czbiohub-sf/grassp/issues) page.
