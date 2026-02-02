# Tools: `tl`

```{eval-rst}
.. module:: grassp.tl
```

```{eval-rst}
.. currentmodule:: grassp
```

This module provides various tools for analyzing proteomics data.

## Clustering

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.leiden_mito_sweep
   tl.calculate_interfacialness_score
   tl.silhouette_score
   tl.calinski_habarasz_score
   tl.tagm_map_train
   tl.tagm_map_predict
   tl.knn_f1_score
```

## Semi-supervised Localization

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.knn_annotation
   tl.svm_train
   tl.svm_annotation
```

## Ontology Enrichment

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.calculate_cluster_enrichment
```


## Integration

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.align_adatas
   tl.aligned_umap
   tl.remodeling_score
```

## Graph analysis

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.to_knn_graph
   tl.get_n_nearest_neighbors
```
