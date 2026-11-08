"""Package the fixed stack+MTD candidate without changing the frozen release."""
import hashlib,json,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/mtd_01';base=ROOT/'artifacts/blend_release/entry.zip'
with zipfile.ZipFile(base) as src:
 files={name:src.read(name) for name in src.namelist() if not name.endswith('/')}
files['predict.py']=(ROOT/'predict_mtd_stack.py').read_bytes()
files['slot_model/mtd.py']=(ROOT/'slot_model/mtd.py').read_bytes()
files['mtd.json']=(OUT/'change.json').read_bytes()
files['MTD_LICENSE']=(ROOT/'research/mtd_reference/LICENSE').read_bytes()
files['docs/mtd-investigation.md']=(ROOT/'docs/mtd-investigation.md').read_bytes()
files['README.md']=b'''# Fixed stack plus changepoint MTD candidate\n\nInstall pinned requirements.txt in the organizer's Python image. Offline command:\n\npython predict.py --data /data --out /output/submission.csv --model model --mtd mtd.json\n\nPayout order: 0,1,2,5,20. This is a candidate, not a submitted entry.\n'''
files['reproduce.json']=json.dumps({'image':'python','command':'python predict.py --data /data --out /output/submission.csv --model model --mtd mtd.json','probability_order':[0,1,2,5,20]},indent=2).encode()
manifest={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n!='source_manifest.json'}
files['source_manifest.json']=json.dumps(manifest,indent=2,sort_keys=True).encode()
with zipfile.ZipFile(OUT/'entry.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as dst:
 for name in sorted(files):dst.writestr(name,files[name])
print(json.dumps({'entry_sha256':hashlib.sha256((OUT/'entry.zip').read_bytes()).hexdigest(),'bytes':(OUT/'entry.zip').stat().st_size,'members':len(files)}))
