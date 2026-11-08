"""Build the REQUIRED inventory as a real ZIP via the organizer's own package_code,
then validate it with inspect_archive. Measures actual archive bytes, not a per-file
deflate sum. Writes nothing outside the verify directory."""
import json, shutil, sys, tempfile, zipfile
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In')
V = R / 'artifacts/audit/packaging-feasibility/verify'
sys.path.insert(0, str(R / 'organizer'))
from slot_forecasting.submission_contract import package_code, inspect_archive, MAX_ZIP, MAX_EXPANDED, MAX_FILES

inv = json.loads((R / 'artifacts/audit/packaging-feasibility/inventory.json').read_text())
req = [f for f in inv['files'] if f['required'] and f.get('present')]
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp) / 'source'; stage.mkdir()
    for f in req:
        dest = stage / f['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R / f['path'], dest)
    (stage / 'README.md').write_text('# audit measurement only\n\nNot a submission.\n')
    (stage / 'reproduce.json').write_text(json.dumps(
        {'image': 'python', 'command': ['python', 'predict.py', '--data', '/data', '--out', '/output/submission.csv']}) + '\n')
    out = Path(tmp) / 'measure.zip'
    package_code(stage, out)
    content = out.read_bytes()
    inspect_archive(content)
    with zipfile.ZipFile(out) as z:
        infos = z.infolist()
    rep = {'required_files_copied': len(req),
           'zip_entries': len(infos),
           'zip_bytes_real': len(content),
           'sum_file_size_unpacked': sum(i.file_size for i in infos),
           'sum_compress_size': sum(i.compress_size for i in infos),
           'inspect_archive': 'PASSED',
           'pct_of_event_100MB': 100 * len(content) / 100_000_000,
           'pct_of_kit_MAX_ZIP': 100 * len(content) / MAX_ZIP,
           'pct_of_MAX_EXPANDED': 100 * sum(i.file_size for i in infos) / MAX_EXPANDED,
           'pct_of_MAX_FILES': 100 * len(infos) / MAX_FILES,
           'auditor_claim_deflate_sum': 31550807,
           'auditor_claim_required_bytes': 33922606}
print(json.dumps(rep, indent=2))
(V / 'realzip.json').write_text(json.dumps(rep, indent=2) + '\n')
