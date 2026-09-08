# Tools: `tl`

```{eval-rst}
.. module:: grassp.tl
```

```{eval-rst}
.. currentmodule:: grassp
```

This module provides various tools for analyzing proteomics data.

## Annotation

Every annotator writes the same slots, so a result can be read, scored and plotted
without knowing which one produced it:

| slot | contents |
|---|---|
| `obsm[f"{key_added}_probabilities"]` | one column per compartment, named |
| `obs[key_added]` | the call, `NaN` where it abstained |
| `obs[f"{key_added}_probability"]` | the probability of that call |
| `uns[f"{key_added}_params"]` | which annotator, and what it was told |
| `uns[f"{key_added}_colors"]` | in the category order of `obs[key_added]` |

`uns[f"{key_added}_params"]["kind"]` says how to read the matrix. `"simplex"` means the
compartments competed for one unit of mass, so the rows sum to 1 and an argmax is a
call. `"per_term"` means each column is an independent membership probability; the rows
do not sum to anything in particular, and resolving them into a call is a separate step.

**Which annotator?** If you have marker proteins and want one compartment each, use
{func}`~grassp.tl.competitive_diffusion` (graph) or {func}`~grassp.tl.svm_annotation` /
{func}`~grassp.tl.tagm_map_predict` (feature-space classifiers). If your labels overlap
or nest — GO-CC, UniProt-SL — use {func}`~grassp.tl.independent_diffusion`, which does
not force them to compete. If you have clusters and an enrichment table rather than
markers, use {func}`~grassp.tl.soft_cluster_annotation`.

### Graph label propagation

Mutually exclusive labels spread over the *k*-NN graph and compete, giving a simplex.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.competitive_diffusion
```

### Marker-supervised classifiers

Fitted in feature space on marker proteins rather than on the graph.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.svm_tune_hyperparameters
   tl.svm_annotation
   tl.tagm_map_train
   tl.tagm_map_predict
   tl.tagm_model
   tl.ccompass
   tl.ccompass_default_params
```

### Ontology-aware annotation

Overlapping or hierarchical labels, diffused one-vs-rest so they do not compete. The
result is a per-term membership probability, and resolving it into a call is explicit.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.independent_diffusion
   tl.resolve_diffusion
   tl.load_gmt
```

### Soft cluster annotation

Propagate the *uncertainty* of a per-cluster enrichment rather than one hard top term.
Listed in pipeline order; `soft_cluster_annotation` ties all three together.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.enrichment_to_cluster_distribution
   tl.mgsa_to_cluster_distribution
   tl.soft_cluster_annotation
   tl.resolve_soft_labels
```

### Marker QC and evaluation

Clean a marker set before annotating, and score any annotation against ground truth.
These take `(gt_col, pred_col)` and work for every annotator above.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.prune_markers
   tl.annotation_f1_score
   tl.annotation_confusion_matrix
   tl.class_balance
```

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

## Ontology Enrichment

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.calculate_cluster_enrichment
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
   tl.MgsaResult
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

## Deprecated

Kept for one release each so existing code keeps running; both emit a
``DeprecationWarning`` and forward to :func:`~grassp.tl.competitive_diffusion`.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   tl.competitive_propagation
   tl.knn_annotation
```
