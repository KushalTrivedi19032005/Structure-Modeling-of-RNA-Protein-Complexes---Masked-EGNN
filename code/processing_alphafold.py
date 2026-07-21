import os
import zipfile
import shutil
from pathlib import Path
from collections import defaultdict
import re

# Anchor all paths to this script's folder (code/), so it runs from any CWD.
BASE = Path(__file__).resolve().parent
SRC = BASE / "AlphaFold-FINAL"           # folder containing the .zip files
OUT = BASE / "AlphaFold-CIF3"       # output folder for extracted .cif files
LIST_FILE = BASE / "complexes4.txt"

OUT.mkdir(parents=True, exist_ok=True)

pattern = re.compile(r"fold_(.+)_model_0\.cif$")

# 1. Check duplicate zipped folders (by inner top-level folder name)
seen = defaultdict(list)
for z in SRC.glob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        top = {n.split("/")[0] for n in zf.namelist() if n.strip()}
        for t in top:
            seen[t].append(z.name)

dupes = {k: v for k, v in seen.items() if len(v) > 1}
if dupes:
    print("Duplicate folders found:")
    for k, v in dupes.items():
        print(f"  {k}: {v}")
else:
    print("No duplicate folders.")

# 2 + 3. Extract matching .cif files and collect names
names = []
for z in SRC.glob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            base = os.path.basename(n)
            m = pattern.match(base)
            if m:
                names.append(m.group(1))
                with zf.open(n) as src, open(OUT / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)

LIST_FILE.write_text(",".join(names))
print(f"Extracted {len(names)} files. Names written to {LIST_FILE}.")