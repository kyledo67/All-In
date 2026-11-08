# Deployed TabPFN component

Built with PriorLabs-TabPFN. License: Prior Labs License v1.1, copy at
`research/tabpfn/LICENSE`. See `THIRD_PARTY_NOTICES.md`.

Two saved contexts, seeds 20261108 and 20261109, both the `hmm` feature variant
(182 observable columns plus the 7 portable HMM summary columns). `TabPFNEnsemble`
averages them, which is exactly what `scripts/tabpfn_evaluate.py` scored.

The `hmm` variant was selected on the fitting folds in all four development folds, so
no per-fold switching is needed. Each `context.npz` holds only the derived feature
table and its payout labels from `public/train`; no upstream model weights are stored
here, and no validation target was used to build it.

The pre-trained checkpoint itself is NOT in this directory. Fetch it once with
`scripts/tabpfn_fetch_checkpoint.py`, after which prediction runs with
`HF_HUB_OFFLINE=1` and no network access.
