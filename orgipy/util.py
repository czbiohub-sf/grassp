from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData
import warnings


def confirm_proteins_as_obs(data: AnnData) -> None:
    if "proteins_as_obs" in data.uns.keys():
        if data.uns["proteins_as_obs"]:
            return
        else:
            raise ValueError(
                "data.uns['proteins_as_obs'] is set to False."
                "This function assumes that the .obs dimension are proteins."
            )
    else:
        dims = data.shape
        if dims[0] < dims[1]:
            # Print a warning that this is likely a mistake
            warnings.warn(
                f"It seems like you have less proteins ({dims[0]}) than Samples ({dims[1]})."
                "This function assumes that the .obs dimension are proteins."
                "Please check if you have set the correct dimensions."
            )
        else:
            warnings.warn(
                "Assuming .obs is proteins. To get rid of this warning, please set adata.uns['proteins_as_obs'] = True"
            )
