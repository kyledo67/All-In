"""Corrected required set: the audited inventory plus organizer/slot_forecasting/baseline.py,
which transition.py imports. Re-measures bytes, a real ZIP, and importability."""
import hashlib, json, shutil, subprocess, sys, tempfile, zipfile, zlib
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In')
V = R / 'artifacts/audit/packaging-feasibility/verify'
sys.path.insert(0, str(R / 'organizer'))
from slot_forecasting.submission_contract import package_code, inspect_archive, MAX_EXPANDED

inv = json.loads((R / 'artifacts/audit/packaging-feasibility/inventory.json').read_text())
req = [f['path'] for f in inv['files'] if f['required'] and f.get('present')]
ADDED = ['organizer/slot_forecasting/baseline.py']
fixed = req + [p for p in ADDED if p not in req]
raw = sum((R / p).stat().st_size for p in fixed)
defl = sum(len(zlib.compress((R / p).read_bytes(), 9)) for p in fixed)

with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp) / 'source'; stage.mkdir()
    for p in fixed:
        d = stage / p; d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(R / p, d)
    probe = Path(tmp) / 'probe.py'
    probe.write_text(
        "import sys, json\n"
        "root = sys.argv[1]\n"
        "sys.path[:0] = [root, root + '/organizer']\n"
        "out = {}\n"
        "for mod in ('slot_model.features','slot_model.phase2_features','slot_model.phase2_predict',\n"
        "            'slot_model.portable','slot_model.mtd','slot_model.mtd_pairs',\n"
        "            'slot_model.frozen_blend','slot_model.tabpfn_model'):\n"
        "    try:\n"
        "        __import__(mod); out[mod]='OK'\n"
        "    except Exception as e:\n"
        "        out[mod]=type(e).__name__+': '+str(e)\n"
        "print(json.dumps(out))\n")
    run = subprocess.run([str(R / '.venv/bin/python'), '-I', '-B', str(probe), str(stage)],
                         capture_output=True, text=True, cwd=tmp)
    imports = json.loads(run.stdout) if run.stdout.strip().startswith('{') else {'error': run.stderr[-800:]}
    (stage / 'README.md').write_text('# audit measurement only\n\nNot a submission.\n')
    (stage / 'reproduce.json').write_text(json.dumps(
        {'image': 'python', 'command': ['python', 'predict.py', '--data', '/data', '--out', '/output/submission.csv']}) + '\n')
    out = Path(tmp) / 'fixed.zip'
    package_code(stage, out)
    content = out.read_bytes(); inspect_archive(content)
    with zipfile.ZipFile(out) as z: infos = z.infolist()

rep = {'baseline_py_bytes': (R / 'organizer/slot_forecasting/baseline.py').stat().st_size,
       'corrected_required_count': len(fixed),
       'corrected_required_raw_bytes': raw,
       'corrected_required_deflate_sum': defl,
       'audited_raw_bytes': 33922606, 'audited_deflate_sum': 31550807,
       'real_zip_bytes_with_readme_and_reproduce': len(content),
       'real_zip_entries': len(infos),
       'real_zip_unpacked_bytes': sum(i.file_size for i in infos),
       'pct_of_event_100MB': 100 * len(content) / 100_000_000,
       'pct_of_MAX_EXPANDED_512MiB': 100 * sum(i.file_size for i in infos) / MAX_EXPANDED,
       'inspect_archive': 'PASSED',
       'imports_from_corrected_set': imports,
       'ckpt_share_of_raw_pct': round(100 * 29009539 / raw, 2),
       'ckpt_deflate_share_of_deflate_pct': round(100 * 26930075 / defl, 2)}
print(json.dumps(rep, indent=2))
(V / 'fixed_inventory.json').write_text(json.dumps(rep, indent=2) + '\n')
