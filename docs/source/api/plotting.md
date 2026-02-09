# Plotting: `pl`

```{eval-rst}
.. module:: grassp.pl
```

```{eval-rst}
.. currentmodule:: grassp
```

This module provides visualization functions for proteomics data.

```{note}
Many of [Scanpy's plotting](https://scanpy.readthedocs.io/en/stable/api/plotting.html) functions can be used directly with grassp AnnData objects. If you are looking for a specific plot, check whether it is already implemented in Scanpy.
```

## Preprocessing

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.highly_variable_proteins
   pl.bait_volcano_plots
   pl.marker_profiles
   pl.marker_profiles_split
   pl.protein_clustermap
   pl.sample_heatmap
```

## Integration
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.aligned_umap
   pl.remodeling_score
   pl.remodeling_sankey
   pl.mr_plot
```

## Clustering
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.qsep_heatmap
   pl.qsep_boxplot
   pl.tagm_map_contours
   pl.tagm_map_pca_ellipses
   pl.knn_violin
```

## Ternary
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.ternary
```
