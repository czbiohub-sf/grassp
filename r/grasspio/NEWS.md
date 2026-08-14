## 0.1.0 (2026-08-14)


### ⚠ BREAKING CHANGES

* read_prolocdata now returns NaN where it previously returned the string "unknown". Any code that filtered on == "unknown" needs .isna(), and marker counts will change. Pass unknown_to_nan=False to restore the old output.

### Features

* pRoloc interoperability — exchange objects as plain h5ad ([#31](https://github.com/czbiohub-sf/grassp/issues/31)) ([63d1463](https://github.com/czbiohub-sf/grassp/commit/63d14633a2bf1846dd3e4b9342f9c59612ffc4c2))
