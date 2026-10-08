"""Package the contribution at paths that can be overlaid on the upstream repo."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ["README.md", ".gitignore", ".gitattributes", "requirements.txt",
         "requirements-tested.txt", "docs/DOWNSTREAM_LAYOUT.json", "docs/DOWNSTREAM.md",
         "docs/DOWNSTREAM_VALIDATION.md", "Others/verify_downstream_layout.py",
         "Others/smoke_downstream.py", "Others/package_downstream.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((ROOT / "docs/DOWNSTREAM_LAYOUT.json").read_text(encoding="utf-8"))
    paths = sorted({item["path"] for item in records} | set(EXTRA))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in paths:
            data = (ROOT / relative).read_bytes()
            archive.writestr(relative, data)
            manifest.append({"path": relative, "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, indent=2))
    print(f"Packaged {len(paths)} files at upstream-relative paths: {args.output}")


if __name__ == "__main__":
    main()
