# Improvement ideas

Things that were investigated, judged not worth doing *now*, and would otherwise have to be
re-investigated from scratch. Each entry records what was measured, so the next person can
decide from the evidence rather than repeating the work.

---

## Fitting `alpha` for `competitive_diffusion`

**Status:** deferred, 2026-09-07. Worth doing if `method="spreading"` becomes the default,
or if a user reports poor calls on a map with heavily overlapping compartments.

### The idea

`independent_diffusion` fits a diffusion depth `a*` **per term** by sweeping `alphas` and
maximising leave-one-out average precision. `competitive_diffusion` takes `alpha=0.8` as a
fixed default. Fitting it there too would make the two consistent and remove a magic number.

### Blocking fact: `alpha` only exists on one path

`alpha` is used **only** in the `method="spreading"` branch of `_propagate_soft`. The default
`method="propagation"` is a single-step `T @ seed`, and the `iterative` variant is a clamped
loop — neither has an `alpha`. The code already knows this: `write_annotation` records
`"alpha": float(alpha) if method == "spreading" else None`.

So this feature only applies to a **non-default** mode, and `alpha="auto"` under
`method="propagation"` must raise rather than silently do nothing.

### Is it worth fitting? Modestly

Held-out macro-F1 against `alpha`, on synthetic maps: 5 compartments × 60 proteins, 10
dimensions, 50% of proteins used as markers, 3-fold stratified CV over the markers,
`method="spreading"`.

| regime (separation, spread) | a=0.1 | 0.3 | 0.5 | 0.7 | 0.9 | best | range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| well separated (4.0, 1.0) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — | **0.000** |
| overlapping (1.6, 1.4) | 0.868 | 0.889 | 0.888 | 0.901 | 0.927 | 0.9 | 0.059 |
| heavy overlap (1.0, 1.6) | 0.573 | 0.593 | 0.599 | 0.620 | 0.591 | 0.7 | 0.046 |

Reading: **irrelevant when compartments separate cleanly, worth roughly 1–6 macro-F1 points
when they do not**, and the optimum genuinely moves (0.9 vs 0.7). The current default of 0.8
is already near-optimal in both non-trivial regimes — it costs about 1.4 points in the
heavy-overlap case. Real, but small. The stronger argument for building it is not hard-coding
a magic number.

### Negative result: do **not** port `independent_diffusion`'s shortcut

The obvious implementation is to reuse the closed-form leave-one-out score
`s = (F - (1-a)*y) / (den - (1-a))`, which removes a protein's own seed contribution
analytically and needs one pass instead of k folds. **It was prototyped and it does not
work here.**

| regime | CV picks | closed-form LOO picks | agree |
| --- | --- | --- | --- |
| well separated | 0.1 | 0.1 | yes |
| overlapping | 0.9 | 0.9 | yes |
| **heavy overlap** | **0.7** | **0.9** | **no** |

It disagrees exactly where `alpha` matters most, and is systematically optimistic (0.693 vs
0.591 at a=0.9) — its curve rises monotonically where the CV curve turns over. Rank
correlation with CV was 0.73–0.86.

**Why it does not transfer.** In `independent_diffusion` the honest score is judged by
average precision on a *single column* against a binary target: the correction is exact and
nothing crosses columns. In `competitive_diffusion` the decision is an argmax **across**
columns after class-mass normalisation, and CMN is a global column rescale computed from
every protein — which a per-protein correction cannot undo. Residual leakage through
neighbours compounds it.

The speedup would not have justified it anyway: CV costs `alphas x folds` propagations and
the shortcut costs `alphas`, so the speedup is bounded by the fold count (measured 3–5x).

### Recommended design

Marker-holdout cross-validation, which is also the idiom `svm_tune_hyperparameters` already
establishes in this package.

```python
competitive_diffusion(
    data, gt_col,
    method="spreading",
    alpha: float | Literal["auto"] = 0.8,
    alphas=None,           # grid used when alpha="auto"; default linspace(0.1, 0.9, 19)
    tune_folds: int = 3,   # StratifiedKFold over the marker proteins
    ...
)
```

`alphas` deliberately reuses `independent_diffusion`'s parameter name so the two read alike.

- **Mechanics.** Stratified-split the markers. Per fold, seed with the training markers only,
  propagate at each `alpha`, score macro-F1 on the held-out markers. Average over folds, take
  the best `alpha` (ties toward the smaller value, matching `independent_diffusion`). Then one
  final propagation on all markers at `a*`.
- **Recording.** `a*` into `uns[f"{key}_params"]["alpha"]`, which the annotation contract
  already carries — so `resolve_soft_labels` picks it up automatically and its entropy null
  tests the tuned depth rather than the default. Store the scan beside it
  (`{"alphas": [...], "macro_f1": [...], "folds": 3}`), mirroring `svm_params["cv_results"]`.
- **Cost.** `len(alphas) * folds + 1` propagations; 58 with the defaults. Each is a sparse
  matmul loop of 10–60 iterations, so seconds on a real map. A coarser default grid than
  `independent_diffusion`'s 19 is justified — the measured curves are smooth.
- **Guards.** `alpha="auto"` with `method != "spreading"` raises. `fix_markers` needs no
  special handling: held-out markers are not in the seed, so nothing clamps them.

### Decide first

- **Is `spreading` becoming the default?** If not, this tunes a path most users never take.
- **Leave `independent_diffusion` alone?** Its per-term average-precision objective is right
  for one-vs-rest, so probably yes — but then the two functions fit `alpha` by *different
  criteria*, which is a subtler inconsistency than the one this fixes. Document it either way.

### Reproducing the numbers

Synthetic maps from `np.random.default_rng(0)`, `k=5` compartments of `n=60` proteins in 10
dimensions, centres at `arange(k) * separation`, `sc.pp.neighbors(n_neighbors=15, use_rep="X")`.
Markers are a random 50% of proteins; scoring is `sklearn.metrics.f1_score(average="macro")`
on the held-out markers of each `StratifiedKFold(3, shuffle=True, random_state=0)` split.

---

## Also deferred

Smaller items from the same review, listed so they are not rediscovered.

- **C-COMPASS is outside the annotation contract.** It is the one annotator that does not write
  the five slots (`_probabilities`, the label, `_probability`, `_params`, `_colors`). It has
  **zero test coverage** and its `_contributions` are neural-network contribution scores whose
  normalisation was not verified — declaring a `kind` without measuring it would be worse than
  declaring none, and `require_kind` lets unlabelled annotations through. Needs someone who can
  run the model.

- **`independent_diffusion` fit/apply split.** It does the most fitting of anything in the
  package — 19 depths x n_terms diffusions, Hutchinson probes, cross-fitted isotonic
  regressors — and keeps only `a*`, in `uns[f"{key}_alpha"]`, which nothing reads.
  The serialisation obstacle is **not** real: an `IsotonicRegression` is two float arrays
  (`X_thresholds_`, `y_thresholds_`), they round-trip through h5ad, and `np.interp` on the
  knots reproduces `predict` exactly (verified, max difference 0.0). The real obstacle is
  statistical: the calibration maps *this map's* z-scores to probabilities, and the z-score is
  built from the term's prevalence and the effective neighbourhood size **in this map**.
  Whether the curve transfers across maps is an empirical question about the data.
  A cheap interim that commits to nothing: an `alphas_fixed: dict[str, float] | None`
  parameter, making the one already-persisted piece of fitted state reusable and removing the
  depth sweep from repeat runs.

- **`resolve_soft_labels`' non-inplace payload** is still the nested
  `{"obs": ..., "null_summary": ...}` rather than the flat shape the other annotators return.

- **`plot_optimization` and the commented-out F1 optimiser** in `competitive_diffusion` are
  still there: the parameter is accepted and ignored (its own docstring says so), and ~40 lines
  of commented-out code sit in the body referencing `f1_score` and `plt`, neither of which is
  imported any more. Removing the parameter and the three notebook cells that still pass it
  must be **one commit** — a failing notebook is only a Sphinx warning, so a split would
  publish three broken cells.

- **`renormalize_class_mass` is Class Mass Normalization** (Zhu & Ghahramani 2002), with the
  prior taken as the marker count per class — not an ad-hoc knob, and not inverse-frequency
  weighting. Two places where grassp goes beyond the papers are currently undocumented: CMN is
  applied on the `spreading` path too (Zhou et al. have no such step), and with a soft seed the
  prior becomes total soft mass per class rather than a count. Worth naming in the docstring so
  the parameter stops looking like an open question.
