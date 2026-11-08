"""Extract a built candidate ZIP into a fresh directory and run it with the network down.

The point is to prove the package needs nothing it does not ship. HF_HUB_OFFLINE=1 alone
would not prove that, because it only asks huggingface_hub to behave; so this harness

  * rebuilds the environment from scratch with a minimal variable set, which removes
    every HF_*, TABPFN_*, PIP_*, no_proxy and credential variable this shell holds,
  * sets HF_HUB_OFFLINE=0, i.e. downloads are permitted as far as the library knows,
  * and points http_proxy / https_proxy / all_proxy / HF_ENDPOINT at 127.0.0.1:1, a port
    nothing listens on, so any outbound HTTP attempt fails with a connection error
    instead of silently succeeding.

The extraction goes through the organizer's own `submission_contract.extract_archive`,
so the ZIP is validated against the real upload contract before anything runs.
"""
import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'organizer'))

BLOCKED = 'http://127.0.0.1:1'


def closed_port(host='127.0.0.1', port=1):
    """Confirm the port used to break the network really refuses connections."""
    probe = socket.socket()
    probe.settimeout(2)
    try:
        probe.connect((host, port))
    except OSError as exc:
        return {'blocked': True, 'error': f'{type(exc).__name__}: {exc}'}
    finally:
        probe.close()
    return {'blocked': False, 'error': None}


def offline_environment(scratch, python):
    """A minimal environment: no HF_*, no proxy bypass, outbound HTTP pointed at a dead port."""
    home = scratch / 'home'
    tmp = scratch / 'tmp'
    for path in (home, tmp):
        path.mkdir(parents=True, exist_ok=True)
    return {
        'PATH': f'{Path(python).parent}:/usr/bin:/bin:/usr/sbin:/sbin',
        'HOME': str(home),
        'TMPDIR': str(tmp),
        'LANG': 'C.UTF-8',
        'LC_ALL': 'C.UTF-8',
        'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
        'VECLIB_MAXIMUM_THREADS': '2',
        'NUMEXPR_NUM_THREADS': '2',
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONHASHSEED': '0',
        # Downloads are ALLOWED as far as the libraries know ...
        'HF_HUB_OFFLINE': '0',
        'TRANSFORMERS_OFFLINE': '0',
        # ... and then every outbound route is pointed at a closed port.
        'http_proxy': BLOCKED, 'https_proxy': BLOCKED, 'all_proxy': BLOCKED,
        'HTTP_PROXY': BLOCKED, 'HTTPS_PROXY': BLOCKED, 'ALL_PROXY': BLOCKED,
        'HF_ENDPOINT': BLOCKED,
        'no_proxy': '', 'NO_PROXY': '',
    }


def directory_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def package_digest(root):
    """Hash of the extracted tree, to prove the read-only mount was never written."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True,
                        help='artifacts/win_head_release or artifacts/win_head_release_nofx')
    parser.add_argument('--data', type=Path, default=ROOT / 'main-participants-60-40/public')
    parser.add_argument('--scratch', type=Path, required=True,
                        help='A directory that does not exist yet; the ZIP is extracted inside it')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--splits', nargs='+', default=['test', 'validation'])
    args = parser.parse_args()

    import numpy as np
    from slot_forecasting.io import read_submission
    from slot_forecasting.submission_contract import extract_archive

    archive = args.package / 'entry.zip'
    content = archive.read_bytes()
    scratch = args.scratch.resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    extracted = scratch / 'pkg'
    config = extract_archive(content, extracted)

    port = closed_port()
    tree_digest = package_digest(extracted)
    environment = offline_environment(scratch, args.python)
    report = {
        'package': str(args.package.relative_to(ROOT)),
        'entry_zip': {'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()},
        'contract_validated_by': 'organizer/slot_forecasting/submission_contract.extract_archive',
        'reproduce_json': config,
        'extracted_to': str(extracted),
        'extracted_files': sorted(str(p.relative_to(extracted))
                                  for p in extracted.rglob('*') if p.is_file()),
        'extracted_tree_sha256': tree_digest,
        'network_block': {
            'method': 'fresh minimal environment (no HF_*/proxy inheritance); '
                      'HF_HUB_OFFLINE=0 so downloads are permitted; '
                      'http_proxy/https_proxy/all_proxy/HF_ENDPOINT -> http://127.0.0.1:1',
            'closed_port_probe': port,
            'environment': environment,
        },
        'runs': {},
    }
    if not port['blocked']:
        raise SystemExit('127.0.0.1:1 accepted a connection; pick a port that is really closed')

    for split in args.splits:
        output = scratch / 'output' / f'{split}_submission.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(args.python), 'predict.py', '--data', str(args.data.resolve()),
                   '--out', str(output), '--split', split]
        printable = ('env -i ' + ' '.join(f'{k}={v!r}' for k, v in sorted(environment.items()))
                     + ' ' + ' '.join(command) + f'   # cwd={extracted}')
        scratch_before = directory_bytes(scratch / 'tmp')
        start = time.perf_counter()
        done = subprocess.run(command, cwd=extracted, env=environment,
                              capture_output=True, text=True)
        seconds = time.perf_counter() - start
        scratch_after = directory_bytes(scratch / 'tmp')
        entry = {'command': printable, 'cwd': str(extracted), 'returncode': done.returncode,
                 'wall_clock_seconds': seconds, 'threads': 2,
                 'scratch_bytes_written': scratch_after - scratch_before,
                 'scratch_limit_bytes': 134217728,
                 'package_tree_unchanged': package_digest(extracted) == tree_digest,
                 'stdout_tail': done.stdout.strip().splitlines()[-40:],
                 'stderr_tail': done.stderr.strip().splitlines()[-20:]}
        if done.returncode != 0:
            entry['passed'] = False
            report['runs'][split] = entry
            (args.package / 'verification.json').write_text(
                json.dumps(report, indent=2, sort_keys=True) + '\n')
            raise SystemExit(f'offline run of {split} FAILED:\n{done.stdout}\n{done.stderr}')

        reference = args.package / f'{split}_submission.csv'
        fresh = read_submission(output)
        saved = read_submission(reference)
        ids = sorted(saved)
        if sorted(fresh) != ids:
            raise SystemExit('offline run produced a different query_id set')
        deviation = float(np.max(np.abs(np.array([fresh[i] for i in ids])
                                       - np.array([saved[i] for i in ids]))))
        entry.update({
            'in_repo_csv': str(reference.relative_to(ROOT)),
            'fresh_csv': str(output),
            'max_abs_probability_deviation_after_normalization': deviation,
            'tolerance': 1e-8,
            'within_tolerance': deviation <= 1e-8,
            'bitwise_identical': output.read_bytes() == reference.read_bytes(),
            'csv_bytes': output.stat().st_size,
            'csv_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'event_csv_limit_bytes': 10000000,
            'runtime_budget_seconds': 1800,
            'within_runtime_budget': seconds <= 1800,
        })
        check = subprocess.run(
            [str(args.python), 'check_submission.py', str(output), '--data', str(args.data.resolve()),
             '--split', split],
            cwd=ROOT / 'organizer', capture_output=True, text=True)
        entry['check_submission'] = {
            'command': f'python check_submission.py {output} --data {args.data.resolve()} '
                       f'--split {split}   # cwd=organizer/',
            'returncode': check.returncode,
            'stdout': check.stdout.strip().splitlines(),
            'stderr': check.stderr.strip().splitlines(),
        }
        if split == 'validation':
            entry['check_submission']['points_caveat'] = (
                'The reported validation points are an IN-SAMPLE diagnostic for this '
                'deployment configuration, not an achieved forecasting score: the '
                'mean-of-folds deployment weights were derived from fits on these same '
                '96 public-validation episodes, which were inspected repeatedly during '
                'development. It is printed because the checker prints it, not as a result.')
        entry['passed'] = bool(entry['within_tolerance'] and check.returncode == 0
                               and entry['within_runtime_budget']
                               and entry['package_tree_unchanged']
                               and entry['scratch_bytes_written'] <= entry['scratch_limit_bytes'])
        report['runs'][split] = entry
        print(f"{split}: {seconds:.1f}s  max abs dev {deviation:.3e}  "
              f"check_submission rc={check.returncode}  "
              f"{'PASS' if entry['passed'] else 'FAIL'}", flush=True)
        for line in entry['check_submission']['stdout']:
            print('   ', line)

    report['passed'] = all(r['passed'] for r in report['runs'].values())
    (args.package / 'verification.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"\n{args.package.name}: {'PASSED' if report['passed'] else 'FAILED'}  "
          f"-> {args.package / 'verification.json'}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
