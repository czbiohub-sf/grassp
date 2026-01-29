# IO: `io`

```{eval-rst}
.. module:: grassp.io
```

```{eval-rst}
.. currentmodule:: grassp
```

Read proteomics data from various search engines and file formats into AnnData objects. These functions leverage the [protdata](https://protdata.sf.czbiohub.org/) package to parse search engine outputs and standardize proteomics data for analysis.

## Search Engine Outputs

Read proteomics data from common search engine outputs (MaxQuant, DIA-NN, FragPipe).

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   io.read_maxquant
   io.read_diann
   io.read_fragpipe
```

## Other Formats

Read data from other subcellular proteomics analysis tools.

```{eval-rst}
.. autosummary::
   :nosignatures:
   :toctree: ../generated/

   io.read_prolocdata
```
