# Two-head CatBoost status
Training-only episode OOF one-hot Brier improved from 0.556885 to 0.553193. On previously examined validation probabilities, standalone official points improved from 31.398446 to 31.589327. Stable and short-history ratios improved; drift and shared-change ratios worsened. Exploratory paired normalized-error interval: [-0.028734, 0.007852]. The 0x normalized contribution improved from 0.614627 to 0.606956.

Unfinished: this model has not been substituted into the frozen four-recipe stack, compared against that complete stack on identical OOF queries, or packaged for offline submission. The 31.589327 figure is standalone exploratory evidence, not a deployed score. Existing releases remain unchanged. No upload occurred.

Reproduce from the project root with the existing CatBoost/NumPy environment: `python scripts/dense_catboost.py`, `python scripts/report_dense_catboost.py`, `python scripts/two_head_catboost.py`, `python scripts/report_two_head.py`. Cached matrices are not committed.
