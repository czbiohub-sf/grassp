## 0.1.0 (2026-09-09)


### ⚠ BREAKING CHANGES

* gr.tl.knn_f1_score is now gr.tl.annotation_f1_score, gr.tl.knn_confusion_matrix is now gr.tl.annotation_confusion_matrix, gr.pl.knn_violin is now gr.pl.annotation_violin and gr.pl.knn_marker_df is now gr.pl.annotation_marker_df. pred_col is required on the two tl functions; pass the annotation you want scored, e.g. pred_col="competitive_diffusion".
* read_prolocdata now returns NaN where it previously returned the string "unknown". Any code that filtered on == "unknown" needs .isna(), and marker counts will change. Pass unknown_to_nan=False to restore the old output.

### Features

* pRoloc interoperability — exchange objects as plain h5ad ([#31](https://github.com/czbiohub-sf/grassp/issues/31)) ([63d1463](https://github.com/czbiohub-sf/grassp/commit/63d14633a2bf1846dd3e4b9342f9c59612ffc4c2))


### Code Refactoring

* Annotation functions ([#33](https://github.com/czbiohub-sf/grassp/issues/33)) ([0f8b141](https://github.com/czbiohub-sf/grassp/commit/0f8b1417e7d5867f9f640084aab27ec6709b04d6))
