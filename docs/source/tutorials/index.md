# Tutorials

Below you will find hands-on notebooks demonstrating typical `grassp` workflows.

## Basic processing tutorials

```{toctree}
:maxdepth: 1

notebooks/DC_tutorial
notebooks/OrgIP_tutorial
notebooks/integration_tutorial
```


## Advanced annotation tutorials

```{toctree}
:maxdepth: 1
notebooks/ccompass_tutorial
notebooks/diffusion_tutorial
```

## pRoloc interoperability

grassp and the R/Bioconductor [pRoloc](https://bioconductor.org/packages/pRoloc/) framework
exchange objects as h5ad in both directions, so there are two tutorials — pick the one that
matches where your work lives.

```{toctree}
:maxdepth: 1

notebooks/proloc_tutorial
proloc_r_tutorial
```

- **[for grassp users](notebooks/proloc_tutorial)** — you preprocess and plot in Python, hand the
  object to R for a classifier pRoloc has and grassp does not, and read it back.
- **[for pRoloc users](proloc_r_tutorial)** — you work in R, and want to pull a dataset off the
  [grassp portal](https://grassp.apps.czbiohub.org/datasets) and analyse it with pRoloc. Needs no
  Python at all.
