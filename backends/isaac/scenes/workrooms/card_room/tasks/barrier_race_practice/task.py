"""Inspect a public static practice curriculum; no simulator or model."""
import json
from ...practice.cases import rules_for, selections

def main():
    print(json.dumps({"rules": rules_for('barrier_race_practice'), "selections": selections('barrier_race_practice')}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
