"""Verify all imported files, allowing only the documented launcher overrides."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    lock = json.loads((ROOT / "sources.lock.json").read_text())
    count = 0
    for source in lock["sources"]:
        base = ROOT / "vendor" / source["name"]
        for name, expected in source["files_sha256"].items():
            data = (base / name).read_bytes()
            if source["name"] == "dexbench" and name in (
                "scripts/teleop/run.sh", "scripts/teleop/run_master_slave.sh"
            ):
                data = data.replace(b'${DEXBENCH_PYTHON:-$root/.venv/bin/python}', b'$root/.venv/bin/python')
            if hashlib.sha256(data).hexdigest() != expected:
                raise SystemExit(f"Source mismatch: {source['name']}/{name}")
            count += 1
    print(f"Verified {count} imported files, including documented launcher adaptations.")


if __name__ == "__main__":
    main()
