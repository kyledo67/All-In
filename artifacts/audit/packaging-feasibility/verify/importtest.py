"""Stage EXACTLY the audited inventory's required file set and try to import the
inference path from it, with no fallback to the repo. Decides whether the claimed
'complete inference-only payload' is importable."""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path
R = Path('/Users/ayan/Downloads/All-In')
V = R / 'artifacts/audit/packaging-feasibility/verify'
inv = json.loads((R / 'artifacts/audit/packaging-feasibility/inventory.json').read_text())
req = [f['path'] for f in inv['files'] if f['required'] and f.get('present')]
rep = {'required_paths': len(req)}
with tempfile.TemporaryDirectory() as tmp:
    stage = Path(tmp) / 'zip-root'
    for p in req:
        d = stage / p
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(R / p, d)
    probe = Path(tmp) / 'probe.py'
    probe.write_text(
        "import sys, json\n"
        "root = sys.argv[1]\n"
        "sys.path[:0] = [root, root + '/organizer']\n"
        "out = {}\n"
        "for mod in ('slot_model.features', 'slot_model.phase2_features',\n"
        "            'slot_model.phase2_predict', 'slot_model.portable',\n"
        "            'slot_model.mtd', 'slot_model.mtd_pairs',\n"
        "            'slot_model.frozen_blend', 'slot_model.tabpfn_model'):\n"
        "    try:\n"
        "        __import__(mod); out[mod] = 'OK'\n"
        "    except Exception as e:\n"
        "        out[mod] = type(e).__name__ + ': ' + str(e)\n"
        "print(json.dumps(out, indent=2))\n")
    run = subprocess.run([str(R / '.venv/bin/python'), '-I', '-B', str(probe), str(stage)],
                         capture_output=True, text=True, cwd=tmp)
    rep['stdout'] = run.stdout
    rep['stderr'] = run.stderr[-2000:]
    rep['returncode'] = run.returncode
print(json.dumps(rep, indent=2))
(V / 'importtest.json').write_text(json.dumps(rep, indent=2) + '\n')
