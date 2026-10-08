"""Check source hashes and the exact directory/import-only transformation record."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path)
    args = parser.parse_args()
    records = json.loads((ROOT / "docs/DOWNSTREAM_LAYOUT.json").read_text(encoding="utf-8"))
    changed = 0
    for record in records:
        target = (ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(target).hexdigest() == record["sha256"], record["path"]
        changed += bool(record["layout_replacements"])
        if args.original_root:
            original = (args.original_root / record["original_path"]).read_bytes()
            assert hashlib.sha256(original).hexdigest() == record["original_sha256"]
            for edit in record["layout_replacements"]:
                before, after = edit["before"].encode(), edit["after"].encode()
                assert original.count(before) == edit["count"]
                original = original.replace(before, after)
            assert original == target, record["path"]
    print(f"PASS: {len(records)} files verified; {changed} have recorded path/import adjustments.")


if __name__ == "__main__":
    main()
