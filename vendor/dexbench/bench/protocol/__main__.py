"""Contract/schema/metrics CLI; physics lives in the separate Isaac runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import aggregate_results
from .schemas import (AgentObservation, EpisodeResult, EpisodeSpec,
                      EvaluationPlan, RobotCapabilities, parse_action, schemas)
from bench.paths import load_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("schema", help="export JSON Schema for public contracts")
    export.add_argument("--output", type=Path)
    validate = sub.add_parser("validate", help="validate one JSON contract")
    validate.add_argument("kind", choices=["action", "observation", "episode", "result", "capabilities", "plan"])
    validate.add_argument("input", type=Path)
    aggregate = sub.add_parser("aggregate", help="aggregate typed results with an optional preregistered plan")
    aggregate.add_argument("results", type=Path, nargs="+")
    aggregate.add_argument("--plan", type=Path)
    aggregate.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "schema":
        result = schemas()
    elif args.command == "validate":
        models = dict(observation=AgentObservation, episode=EpisodeSpec, result=EpisodeResult,
                      capabilities=RobotCapabilities, plan=EvaluationPlan)
        payload = load_json(args.input)
        value = parse_action(payload) if args.kind == "action" else models[args.kind].model_validate(payload)
        result = dict(valid=True, kind=args.kind, canonical=value.model_dump(mode="json"))
    else:
        plan = EvaluationPlan.model_validate_json(args.plan.read_text()) if args.plan else None
        result = aggregate_results([EpisodeResult.model_validate_json(path.read_text()) for path in args.results], plan=plan)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = getattr(args, "output", None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
