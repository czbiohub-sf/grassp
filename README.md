[![PyPI - Version](https://img.shields.io/pypi/v/grassp)](https://pypi.org/project/grassp/)
[![CI](https://github.com/czbiohub-sf/grassp/actions/workflows/CI.yaml/badge.svg)](https://github.com/czbiohub-sf/grassp/actions/workflows/CI.yaml)
[![docs online](https://img.shields.io/badge/docs-online-blue)](https://public.czbiohub.org/comp.bio/grassp/)


[anndata]: https://anndata.readthedocs.io
[scanpy]: https://scanpy.readthedocs.io
[protdata]: https://protdata.sf.czbiohub.org
[alphapepttools]: https://mannlabs.github.io/alphapepttools/
[documentation]: https://public.czbiohub.org/comp.bio/grassp/


<p align="center">
  <img src="https://raw.githubusercontent.com/czbiohub-sf/grassp/refs/heads/main/docs/source/_static/img/logo.svg" alt="grassp logo" width="300"/>
</p>

The **grassp** (**GR**aph-based **A**nalysis of **S**ubcellular/**S**patial **P**roteomics) python module enables fast, flexible and scalable analysis of subcellular proteomics datasets.

It uses the [anndata][] format to store mass-spec data and analysis results and [scanpy][] for many of the dimensionality reduction and visualization functions.

**grassp** enables

- Reading the ouput format of most mass-spectrometry search engines (using [protdata][] or [alphapepttools][])
- Calculating subcellular enrichment profiles of proteins for different experimental protocols
- Annotating the subcellular location of proteins in an unsupervised and semi-supervised manner
- Combining multiple subcellular proteomics datasets
- Finding the optimal experimental design for future experiments based on simulations
- Detecting proteins at the interface of organelles
- Assessing subcellular resolution
- Detecting re-localizing proteins between conditions (work in progress)
- Detecting multi-localizing proteins (work in progress)
- Integration of multiple modalities (e.g. Lipidomics) (work in progress)

Please refer to the [documentation] for reference to individual functions and [tutorials](https://public.czbiohub.org/comp.bio/grassp/tutorials/index.html).

## Installation

grassp can be installed via [pip](https://pypi.org/project/pip/) from [PyPI](https://pypi.org/project/grassp/) with:
```
pip install grassp
```
For details on installation, please see the [install section of the documentation](https://public.czbiohub.org/comp.bio/grassp/installation.html).

## Changelog

See the [CHANGELOG.md](https://github.com/czbiohub-sf/grassp/blob/main/CHANGELOG.md) for a detailed history of changes, updates, and version releases.

## Contributing

If you'd like to contribute to **grassp** please feel free to look at our [contribution guide](https://public.czbiohub.org/comp.bio/grassp/contributing.html) and open a [Pull request](https://github.com/czbiohub-sf/grassp/pulls).

## Authors

**grassp** is created and maintained by the [Leonetti Group](https://biohub.org/leonetti/) at the [Biohub San Francisco](https://biohub.org/). For details, see the [Contributors page](https://public.czbiohub.org/comp.bio/grassp/contributors.html).

To get in touch please use the [GihHub issues](https://github.com/czbiohub-sf/grassp/issues) page.
