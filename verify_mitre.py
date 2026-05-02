import json
from pathlib import Path

mitre_path  = Path('data/mitre_ics_mapping.json')
names_path  = Path('data/processed/class_names.json')

mitre_db = json.loads(mitre_path.read_text())

if names_path.exists():
    class_names = json.loads(names_path.read_text())['class_names']
    print(f"Class names from training: {len(class_names)}")
    missing_in_mitre = [n for n in class_names if n not in mitre_db]
    extra_in_mitre   = [k for k in mitre_db if k not in class_names]
    if missing_in_mitre:
        print(f"MISSING from MITRE JSON: {missing_in_mitre}")
    if extra_in_mitre:
        print(f"EXTRA in MITRE JSON (not in class_names): {extra_in_mitre}")
    if not missing_in_mitre and not extra_in_mitre:
        print("✅ All 25 class names match MITRE JSON keys exactly.")
else:
    print("class_names.json not found — run scripts/train.py first.")
    print(f"MITRE JSON has {len(mitre_db)} families.")
