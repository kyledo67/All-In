"""Audit what the 15 'RAW' pool members' own producing scripts read, and how much of each
member's q head is a verbatim copy of a non-RAW candidate.

Family X's bank uses, from each member, (a) the q head and (b) a = r[payout 1]. If a member
copies rows from a FOLD_FITTED or DEPLOY_FIT candidate as a fallback, those rows carry
validation-target fitting into the bank, even though the member is declared RAW.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/all-in-mpl')
for _v in ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))
import numpy as np

from slot_model import harness as H
from slot_forecasting.io import read_submission
from scripts.stack_manifest import RAW_SOURCES, classify

PRODUCERS = {
    'artifacts/beta_cells/predictions/oof_beta_hard.csv': 'scripts/beta_cells.py',
    'artifacts/beta_cells/predictions/oof_beta_average.csv': 'scripts/beta_cells.py',
    'artifacts/beta_templates/predictions/oof_parity_templates.csv': 'scripts/beta_templates.py',
    'artifacts/beta_subsets/predictions/oof_parity_templates.csv': 'scripts/beta_subsets.py',
    'artifacts/beta_memory/predictions/oof_beta_memory.csv': 'scripts/beta_memory.py',
    'artifacts/two_head_01/validation.csv': 'scripts/two_head_catboost.py',
    'artifacts/dense_catboost_01/validation_dense.csv': 'scripts/dense_catboost.py',
    'artifacts/dense_catboost_01/validation_sparse.csv': 'scripts/dense_catboost.py',
    'artifacts/mtd_pairs_01/validation_additive.csv': 'scripts/evaluate_mtd_pairs.py',
    'artifacts/mtd_pairs_01/validation_control.csv': 'scripts/evaluate_mtd_pairs.py',
    'artifacts/mtd_pairs_01/validation_lag7.csv': 'scripts/evaluate_mtd_pairs.py',
    'artifacts/tabpfn/predictions/tabpfn_hmm_20261108.csv': 'scripts/tabpfn_evaluate.py',
    'artifacts/tabpfn/predictions/tabpfn_hmm_20261109.csv': 'scripts/tabpfn_evaluate.py',
    'artifacts/tabpfn/predictions/tabpfn_observable_20261108.csv': 'scripts/tabpfn_evaluate.py',
    'artifacts/tabpfn/predictions/tabpfn_observable_20261109.csv': 'scripts/tabpfn_evaluate.py',
}

PAT = re.compile(r"read_submission\(\s*ROOT\s*/\s*'([^']+)'")


def main():
    b = H.load(ROOT / 'main-participants-60-40/public')
    names, P, Q, R, _ = H.raw_pool(b)
    A = R[:, :, 0]
    ids = b.ids
    rep = {'producer_reads': {}, 'fallback_rows': {}}

    # ---- what does each producing script read from artifacts/? -------------------------
    for mem, prod in sorted(PRODUCERS.items()):
        f = ROOT / prod
        if not f.exists():
            rep['producer_reads'][prod] = 'producer script missing'
            continue
        reads = sorted(set(PAT.findall(f.read_text())))
        rep['producer_reads'][prod] = [{'reads': r, 'class': classify(r),
                                        'exists': (ROOT / r).exists()} for r in reads]

    # ---- how many rows of each member's q head equal a non-RAW candidate's q? ----------
    cands = {}
    for p in sorted({r for v in rep['producer_reads'].values() if isinstance(v, list)
                     for r in [x['reads'] for x in v]}):
        if classify(p) in ('RAW',) or not (ROOT / p).exists():
            continue
        d = read_submission(ROOT / p)
        if set(d) != set(ids):
            cands[p] = {'class': classify(p), 'status': 'misaligned'}
            continue
        Z = np.array([d[i] for i in ids])
        Z = Z / Z.sum(axis=1, keepdims=True)
        cands[p] = {'class': classify(p), 'q': 1.0 - Z[:, 0],
                    'a': (Z[:, 1] / np.maximum(1.0 - Z[:, 0], 1e-12))}

    rows = {}
    for i, nm in enumerate(names):
        rec = {}
        for p, c in sorted(cands.items()):
            if 'q' not in c:
                continue
            eq = np.isclose(Q[i], c['q'], rtol=0, atol=1e-12)
            rec[p] = {'class': c['class'], 'q_rows_identical': int(eq.sum()),
                      'q_rows_total': len(ids),
                      'q_frac_identical': float(eq.mean())}
            if eq.any():
                rec[p]['by_condition'] = {}
                for cond in sorted(set(b.groups.values())):
                    m = np.array([b.groups[q] == cond for q in ids])
                    rec[p]['by_condition'][cond] = int((eq & m).sum())
                rec[p]['by_type'] = {t: int((eq & (b.types == t)).sum())
                                     for t in sorted(set(b.types.tolist()))}
        rows[nm] = rec
    rep['fallback_rows'] = rows

    # ---- shared a heads: which members collapse onto which canonical a head? ----------
    import hashlib
    groups = {}
    for i, nm in enumerate(names):
        groups.setdefault(hashlib.sha256(A[i].round(12).tobytes()).hexdigest()[:12], []).append(nm)
    rep['a_head_groups'] = {k: v for k, v in sorted(groups.items())}

    out = ROOT / 'artifacts/verify_family_x'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pool_inputs.json').write_text(json.dumps(rep, indent=2, sort_keys=True) + '\n')

    print('=== non-RAW artifacts read by the producing scripts ===')
    for prod, v in sorted(rep['producer_reads'].items()):
        if not isinstance(v, list):
            print(f'{prod}: {v}'); continue
        bad = [x for x in v if x['class'] != 'RAW']
        if bad:
            print(f'{prod}:')
            for x in bad:
                print(f"   {x['class']:<12} {x['reads']}  exists={x['exists']}")
    print('\n=== member q rows copied verbatim from a non-RAW candidate ===')
    for nm, rec in sorted(rows.items()):
        hits = {p: v for p, v in rec.items() if v['q_rows_identical'] > 0}
        if hits:
            print(f'{nm}:')
            for p, v in sorted(hits.items()):
                print(f"   {v['q_rows_identical']:>5}/{v['q_rows_total']} rows "
                      f"({v['q_frac_identical']:.1%})  <- {v['class']}  {p}")
                print(f"        by condition {v['by_condition']}")
    print('\n=== a-head equivalence groups ===')
    for k, v in sorted(groups.items()):
        print(f'{k}: {len(v)} member(s)')
        for x in v:
            print(f'    {x}')


if __name__ == '__main__':
    main()
