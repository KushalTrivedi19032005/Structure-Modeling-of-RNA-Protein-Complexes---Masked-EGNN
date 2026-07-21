import io
import gzip
import zipfile
from pathlib import Path

# Anchor all paths to this script's folder (code/), so it runs from any CWD.
BASE = Path(__file__).resolve().parent
SRC = BASE / "PDB-Test"          # folder containing the .zip files (RCSB batch downloads)
OUT = BASE / "PDB-Test"      # output folder for extracted .cif files

OUT.mkdir(parents=True, exist_ok=True)


def _write_cif(name: str, data: bytes) -> None:
    """Write raw mmCIF bytes to OUT/<name>.cif (stripping any .gz suffix)."""
    stem = Path(name).name
    if stem.endswith(".gz"):
        stem = stem[:-3]
    (OUT / stem).write_bytes(data)


def walk(name: str, data: bytes, count: list) -> None:
    """Recurse through nested zips / gzip until we reach the .cif layer."""
    lower = name.lower()
    if lower.endswith(".zip"):
        # a nested zip: descend into every member
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                walk(member, zf.read(member), count)
    elif lower.endswith(".cif.gz"):
        _write_cif(name, gzip.decompress(data))
        count[0] += 1
    elif lower.endswith(".gz"):
        # some other gzipped file that decompresses to a .cif
        _write_cif(name[:-3], gzip.decompress(data))
        count[0] += 1
    elif lower.endswith(".cif"):
        _write_cif(name, data)
        count[0] += 1
    # anything else (json, txt, ...) is ignored


count = [0]
for z in sorted(SRC.glob("*.zip")):
    with open(z, "rb") as fh:
        walk(z.name, fh.read(), count)

print(f"Extracted {count[0]} .cif file(s) into {OUT}.")
