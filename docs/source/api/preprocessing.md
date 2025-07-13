# Preprocessing: `pp`

```{eval-rst}
.. module:: grassp.pp
```

```{eval-rst}
.. currentmodule:: grassp
```

Any transformation of the data matrix that is not a *tool*. Other than *tools*, preprocessing steps usually don't return an easily interpretable annotation, but perform a basic transformation on the data matrix.

## Basic Preprocessing


```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pp.calculate_qc_metrics
   pp.filter_samples
   pp.filter_proteins
   pp.highly_variable_proteins
   pp.filter_proteins_per_replicate
   pp.aggregate_proteins
   pp.aggregate_samples
   pp.normalize_total
   pp.drop_excess_MQ_metadata
   pp.remove_contaminants
   pp.filter_min_consecutive_fractions
```

## Enrichment

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pp.calculate_enrichment_vs_untagged
   pp.calculate_enrichment_vs_all
```

## Imputation

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   pp.impute_gaussian
```
