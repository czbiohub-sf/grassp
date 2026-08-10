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
   tl.markov_clustering
   tl.calculate_interfacialness_score
   tl.silhouette_score
   tl.calinski_habarasz_score
   tl.qsep_score
   tl.tagm_map_train
   tl.tagm_map_predict
   tl.knn_f1_score
   tl.class_balance
```

## Cluster Merging

Consolidate overclustered Leiden solutions using PAGA connectivity and ontology
enrichment.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.merge_clusters_go
   tl.merge_small_clusters
   tl.paga_dendrogram
   tl.dendrogram_cherry_pairs
```

## Semi-supervised Localization

Propagate mutually exclusive labels (markers) along the *k*-NN graph, or train a
classifier on them.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.competitive_propagation
   tl.knn_annotation
   tl.soft_cluster_annotation
   tl.resolve_soft_labels
   tl.svm_train
   tl.svm_annotation
```

## Ontology-aware Annotation

Propagate overlapping / hierarchical labels (GO-CC, UniProt-SL) one-vs-rest, which
yields per-term membership probabilities rather than a simplex.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.independent_diffusion
   tl.resolve_diffusion
```

## Ontology Enrichment

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.calculate_cluster_enrichment
   tl.enrichment_to_cluster_distribution
```

## Model-based Gene Set Analysis

MGSA explains an observed protein set with a sparse set of ontology terms, rather
than testing each term independently.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.mgsa
   tl.calculate_mgsa
   tl.mgsa_to_cluster_distribution
   tl.MgsaResult
   tl.load_gmt
```

## C-COMPASS

Neural-network compartment prediction, provided by the optional ``ccompass`` extra
(``pip install grassp[ccompass]``).

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.ccompass
   tl.ccompass_default_params
```


## Integration

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.align_adatas
   tl.aligned_umap
   tl.remodeling_score
   tl.mr_score
```

## Graph analysis

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.to_knn_graph
   tl.get_n_nearest_neighbors
```
