# All-In round-one probability forecaster

The active submission candidate is the **equal-weight average of four frozen
fold recipes**, with fixed deployment weights 0.25 each. The user explicitly
revised the replacement policy to compare the complete candidate directly with
the deployed HMM. No model, internal blend weight, preprocessing or calibration
was refitted. See [promotion and evaluation scope](docs/blend-promotion.md).

Upload pair: `artifacts/blend_release/submission.csv` and
`artifacts/blend_release/entry.zip`. The organizer checker passed all 1,536 rows;
clean offline reproduction was byte-identical in 5.75 seconds. The preserved
31.702/60 score is the official combined held-out OOF result, **not an independent
validation score of the deployed four-recipe average**. Nothing was uploaded.

The HMM remains an intact fallback at `artifacts/final/submission.csv` and
`artifacts/final/entry.zip`. The earlier phase-two decision and protocol remain
unchanged in [phase-two results](docs/phase2-results.md) and
[evaluation protocol](docs/phase2-protocol.md). The active selection and checksums
are recorded in `artifacts/active_champion.json`.

Predicts payout probabilities in order **0, 1, 2, 5, 20** using the released
main participant kit. The controlled Dynamax partial-pooling experiment is
complete. Its small holdout advantage did not pass the frozen retention gate,
so its source modification is not used. The frozen blend recipes reuse the shared
Dynamax HMM checkpoint together with local-history and CatBoost components.

The original HMM upload files remain unchanged in `artifacts/final/` as a fallback.
Uploading through the organizers' form is separate from preparing these files.
See `docs/results.md` for official scores, uncertainty and limitations,
`docs/source-map.md` for source tracing, and `docs/evidence-ledger.md` for evidence.

## Reproduce the CSV offline

Python 3.13.7 was tested. For the active blend, install the exact dependencies in
`artifacts/blend_release/requirements.txt` before the offline run:

```sh
.venv/bin/python predict_blend.py --data /absolute/path/to/main-participants-60-40/public --out /tmp/submission.csv
```

From the new extracted ZIP, run `python predict.py --data /data --out
/output/submission.csv --model model`. No JAX, REA, Databricks, MLflow or credentials
are needed. The old HMM fallback still uses the standard library only:

```sh
python3 predict.py --data /absolute/path/to/main-participants-60-40/public --out /tmp/submission.csv
python3 organizer/check_submission.py /tmp/submission.csv --data /absolute/path/to/main-participants-60-40/public
```

The saved population model is `artifacts/experiment/population.json`, and the
deployment choice is in `artifacts/experiment/holdout.json`. The portable forward
recurrence is checked against upstream JAX predictions on all validation and
test queries. Inference does not require REA, Databricks, MLflow or network access.

## Reproduce the investigation

From this project root, install the exact training environment and set the kit path:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-training.lock
.venv/bin/python scripts/restore_vendor.py
DATA=/absolute/path/to/main-participants-60-40/public
.venv/bin/python organizer/run_starter.py --data "$DATA" --out artifacts/baseline-replay
.venv/bin/python scripts/verify.py --data "$DATA" --out artifacts/verification-replay
.venv/bin/python scripts/window_probe.py
.venv/bin/python scripts/experiment.py develop --data "$DATA" --out artifacts/experiment-replay
.venv/bin/python scripts/experiment.py audit --data "$DATA" --out artifacts/experiment-replay
.venv/bin/python predict.py --data "$DATA" --out /tmp/dynamax.csv --engine dynamax
.venv/bin/python scripts/package_entry.py --data "$DATA"
```

Use fresh output directories; existing finalist and holdout files are deliberately
protected from silent overwrite. `scripts/verify.py` uses the preserved original
experiment checkpoint to test the exact delivered population. Its clean-source
replay trains the unmodified class. The experiment uses the committed split list,
not a newly randomized partition. Repeating the audit is a numerical reproduction,
not another independent holdout experiment.

`artifacts/splits.json` fixes 48 development and 48 holdout episodes (12 per
condition in each). One population fit on training histories supports the shared
control, unshrunk adaptation, and three shrinkage settings. No validation outcome
or target is used in fitting; development targets select precision. Holdout is
used once for the predeclared confirmatory gate. No later tuning occurred.

The inherited project contained the split, patch and development results when
this work began; they were audited and preserved rather than overwritten.
At the start of the original investigation, the supplied workspace had no
Databricks integration or CatBoost benchmark; the subsequent CatBoost experiments
are preserved under `artifacts/phase2/`.
The existing experiment's local MLflow fallback was retained. Package versions,
data hashes, source revision, patch hash, parameters, predictions, diagnostics and
run ID are under `artifacts/`. No secrets are packaged.

Upstream Dynamax is pinned to `ffc626bf89bf07c7ae7064922f6a373d013c4b17`.
The readable patch is `artifacts/dynamax.patch`; the original categorical module
is `research/upstream_categorical_glm_hmm.py`. MIT attribution and citation are in
`THIRD_PARTY_NOTICES.md` and the vendored license files.

Git preserves the pinned, patched Dynamax Python source, license and citation in
`research/dynamax-source.tar.gz`; `scripts/restore_vendor.py` restores it without
network access. The local upstream checkout, its Git metadata, feature caches,
virtual environment, credentials and MLflow file store are excluded from Git.
MLflow run identifiers and exported experiment artifacts remain versioned.

<!-- Remote repository description preserved: # All-In all in on black -->
