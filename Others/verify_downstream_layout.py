"""Verify downstream source files against the release checksums."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    records = json.loads((ROOT / "docs/DOWNSTREAM_LAYOUT.json").read_text(encoding="utf-8"))
    for record in records:
        target = (ROOT / record["path"]).read_bytes()
        assert hashlib.sha256(target).hexdigest() == record["sha256"], record["path"]
    print(f"PASS: {len(records)} downstream source files match their release checksums.")


if __name__ == "__main__":
    main()
