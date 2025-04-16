## Plotting: `pl`

```{eval-rst}
.. module:: grassp.pl
```

```{eval-rst}
.. currentmodule:: grassp
```

This module provides visualization functions for proteomics data.

```{note}
Many of scanpy's plotting functions can be directly used with grassp AnnData objects. So if you are looking for a specific plot, it is worth checking if it is already implemented in scanpy.
```

### Preprocessing

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.highly_variable_proteins
   pl.bait_volcano_plots
   pl.protein_clustermap
```

### Integration
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.aligned_umap
   pl.remodeling_score
   pl.remodeling_sankey
``` 

### Clustering
```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pl.tagm_map_contours
   pl.tagm_map_pca_ellipses