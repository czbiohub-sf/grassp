[![PyPI - Version](https://img.shields.io/pypi/v/grassp)](https://pypi.org/project/grassp/)
[![CI](https://github.com/czbiohub-sf/grassp/actions/workflows/CI.yaml/badge.svg)](https://github.com/czbiohub-sf/grassp/actions/workflows/CI.yaml)
[![docs online](https://img.shields.io/badge/docs-online-blue)](https://czbiohub-sf.github.io/grassp/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.11.23.768756-blue)](https://doi.org/10.64898/2026.11.23.768756)

[anndata]: https://anndata.scverse.org
[scanpy]: https://scanpy.scverse.org
[protdata]: https://protdata.sf.czbiohub.org
[alphapepttools]: https://mannlabs.github.io/alphapepttools/
[documentation]: https://czbiohub-sf.github.io/grassp/


<p align="center">
  <img src="https://raw.githubusercontent.com/czbiohub-sf/grassp/refs/heads/main/docs/source/_static/img/logo.svg" alt="grassp logo" width="300"/>
</p>

The **grassp** (**GR**aph-based **A**nalysis of **S**ubcellular/**S**patial **P**roteomics) python module enables fast, flexible and scalable analysis of subcellular proteomics datasets.

> **Read the preprint:** Frank *et al.*, [grassp: A graph-based framework unifies subcellular proteomics across assays and studies](https://doi.org/10.64898/2026.11.23.768756)

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

Please refer to the [documentation] for reference to individual functions and [tutorials](https://czbiohub-sf.github.io/grassp/tutorials/index.html).

## Installation

grassp can be installed via [pip](https://pypi.org/project/pip/) from [PyPI](https://pypi.org/project/grassp/) with:
```
pip install grassp
```
For details on installation, please see the [install section of the documentation](https://czbiohub-sf.github.io/grassp/installation.html).

## Citation

If you use **grassp** in your research, please cite our preprint:

> Frank M, Vaid S, Bhave M, Awayan K, Clark M, McCarthy F, Truong J, DeFelice BC, Lima L, Januel C, Baltazar-Nunez R, Yu Z, Vangipuram M, Njoya L, Savy L, Itzhak DN, Elias JE, Leonetti MD, Peng D. **grassp: A graph-based framework unifies subcellular proteomics across assays and studies.** *bioRxiv* (2026). doi: [10.64898/2026.11.23.768756](https://doi.org/10.64898/2026.11.23.768756)

```bibtex
@article{Frank2026grassp,
  title = {{grassp}: A graph-based framework unifies subcellular proteomics across assays and studies},
  author = {Frank, Max and Vaid, Shivanshi and Bhave, Madhura and Awayan, Kyle and Clark, Marika and McCarthy, Frank and Truong, Johnson and DeFelice, Brian C. and Lima, Leandro and Januel, Camille and Baltazar-Nunez, Rodrigo and Yu, Zachary and Vangipuram, Madhuri and Njoya, Leila and Savy, Laura and Itzhak, Daniel N. and Elias, Joshua E. and Leonetti, Manuel D. and Peng, Duo},
  journal = {bioRxiv},
  year = {2026},
  month = nov,
  doi = {10.64898/2026.11.23.768756},
  url = {https://doi.org/10.64898/2026.11.23.768756}
}
```

## Changelog

See the [CHANGELOG.md](https://github.com/czbiohub-sf/grassp/blob/main/CHANGELOG.md) for a detailed history of changes, updates, and version releases.

## Contributing

If you'd like to contribute to **grassp** please feel free to look at our [contribution guide](https://czbiohub-sf.github.io/grassp/contributing.html) and open a [Pull request](https://github.com/czbiohub-sf/grassp/pulls).

## Authors

**grassp** is created and maintained by the [Leonetti Group](https://biohub.org/leonetti/) at the [Biohub San Francisco](https://biohub.org/). For details, see the [Contributors page](https://czbiohub-sf.github.io/grassp/contributors.html).

To get in touch please use the [GihHub issues](https://github.com/czbiohub-sf/grassp/issues) page.
