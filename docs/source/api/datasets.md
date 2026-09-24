# Datasets


```{eval-rst}
.. module:: grassp.ds
```

```{eval-rst}
.. currentmodule:: grassp
```

Download public datasets in a preprocessed format.

## Public Datasets


```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   ds.hein_2024
   ds.itzhak_2016
   ds.schessner_2023
```

## Unpublished Datasets


```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   ds.hek_dc_2025
   ds.hek_atps_2025
```

## Data Portal

Browse and download curated datasets from the grassp data portal, and fetch raw
datasets from the [pRoloc](https://bioconductor.org/packages/pRolocdata/) collection.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   ds.load_dataset
   ds.list_prolocdata_files
   ds.download_prolocdata
```

## Compartment gene sets

Gene to subcellular compartment term mappings bundled with grassp. These can be used to
annotate clusters or diffused through the graph with {func}`~grassp.tl.independent_diffusion`.
`gene_sets_curated` is the coarsest set of terms with only compartments a subcellular proteomics
experiment can usually resolve. Functions return `{term: [gene, ...]}` dicts, ready to pass as `gene_sets`
to e.g. {func}`~grassp.tl.mgsa`, {func}`~grassp.tl.calculate_cluster_enrichment` or
{func}`~grassp.tl.merge_clusters_go` {func}`~grassp.tl.independent_diffusion`.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   ds.gene_sets_curated
   ds.gene_sets_uniprot_sl
```
