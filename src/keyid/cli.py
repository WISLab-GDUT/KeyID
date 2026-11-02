from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .core.config import load_config
from .core.errors import EXIT_FATAL, EXIT_OK, KeyIDError
from .core.logging import configure
from .core.registry import build_default_registry
from .core.runner import Runner, summarize_run


CONFIG_HELP = "pipeline YAML; required (e.g. configs/pipeline.track1.yaml)"


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    level = "WARNING" if args.quiet else config.runtime.log_level
    configure(level, console=True, files=(), run_id=None)

    registry = build_default_registry(config.runtime.stage_packages)
    runner = Runner(config, registry=registry)

    outcome = runner.run(
        target=args.target,
        only=args.only,
        resume=args.resume,
        force_stages=tuple(args.force_stage or ()),
        dry_run=args.dry_run,
        run_id=args.run_id,
    )

    if args.dry_run:
        plan = runner.plan_payload(target=args.target, only=args.only)


        print(
            json.dumps(
                {"dry_run": True, "run_dir": str(outcome.run_paths.root), **plan},
                ensure_ascii=False,
                indent=2,
            )
        )
        print(
            "dry-run ok: config, DAG, dataset, stage discovery and interpreters "
            "validated; no model loaded.",
            file=sys.stderr,
        )
        return EXIT_OK

    print(json.dumps(_outcome_payload(outcome), ensure_ascii=False, indent=2))
    return outcome.exit_code


def cmd_list_stages(args: argparse.Namespace) -> int:
    packages: tuple[str, ...] = ("keyid.stages",)
    if args.config:
        packages = load_config(args.config).runtime.stage_packages
    registry = build_default_registry(packages)
    entries = registry.describe()
    if args.json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return EXIT_OK
    width = max((len(entry["type"]) for entry in entries), default=0)
    for entry in entries:
        needs = ", ".join(entry["needs"]) or "-"
        print(f"{entry['type']:<{width}}  needs: {needs:<24} {entry['description']}")
    print(
        f"\n{len(entries)} stage type(s) discovered; none of them loaded a model to be listed.",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_status(args: argparse.Namespace) -> int:
    summary = summarize_run(args.run_dir)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return EXIT_OK

    print(f"run      : {summary['run_id']}")
    print(f"dir      : {summary['run_dir']}")
    print(f"created  : {summary.get('created_at')}")
    print(f"updated  : {summary.get('updated_at')}")
    dataset = summary.get("dataset") or {}
    if dataset:
        print(f"dataset  : {dataset.get('format')} {dataset.get('metadata')} ({dataset.get('samples')} samples)")
    print()
    print(f"{'stage':<20} {'type':<22} {'status':<10} {'ok':>5} {'fail':>5} {'skip':>5} {'reuse':>6}")
    for stage_id, entry in (summary.get("stages") or {}).items():
        counts = entry.get("summary") or {}
        print(
            f"{stage_id:<20} {str(entry.get('type') or '-'):<22} {str(entry.get('status') or '-'):<10} "
            f"{counts.get('succeeded', 0):>5} {counts.get('failed', 0):>5} "
            f"{counts.get('skipped', 0):>5} {counts.get('reused', 0):>6}"
        )
    totals = summary.get("totals") or {}
    print()
    print(
        "totals   : "
        f"succeeded={totals.get('succeeded', 0)} failed={totals.get('failed', 0)} "
        f"skipped={totals.get('skipped', 0)} reused={totals.get('reused', 0)}"
    )
    if summary.get("exit_code") is not None:
        print(f"exit code: {summary['exit_code']}")
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    configure(config.runtime.log_level, console=True, files=())
    runner = Runner(config)
    print(json.dumps(runner.plan_payload(target=args.target, only=args.only), ensure_ascii=False, indent=2))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyid",
        description="KeyID staged image/video generation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run stages (default command)")
    run.add_argument("--config", required=True, help=CONFIG_HELP)
    scope = run.add_mutually_exclusive_group()
    scope.add_argument("--target", help="run this stage and its required upstream stages")
    scope.add_argument("--only", help="run exactly this stage (needs an existing run/work_dir for inputs)")
    run.add_argument("--resume", type=Path, help="use this run directory (overrides runtime.work_dir)")
    run.add_argument(
        "--force-stage",
        action="append",
        metavar="STAGE",
        help="regenerate only this stage (repeatable)",
    )
    run.add_argument("--dry-run", action="store_true", help="validate everything without loading a model")
    run.add_argument("--run-id", help="explicit run id (default: <timestamp>_<suffix>)")
    run.add_argument("--quiet", action="store_true", help="only log warnings and errors")
    run.set_defaults(func=cmd_run)

    list_stages = subparsers.add_parser("list-stages", help="list discovered stage types")
    list_stages.add_argument("--json", action="store_true")
    list_stages.add_argument(
        "--config", default=None, help="read runtime.stage_packages from this YAML"
    )
    list_stages.set_defaults(func=cmd_list_stages)

    status = subparsers.add_parser("status", help="summarise an existing run directory")
    status.add_argument("run_dir", type=Path)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    plan = subparsers.add_parser("plan", help="print the execution plan for a config")
    plan.add_argument("--config", required=True, help=CONFIG_HELP)
    plan.add_argument("--target")
    plan.add_argument("--only")
    plan.set_defaults(func=cmd_plan)

    return parser


def _outcome_payload(outcome) -> dict:
    return {
        "run_id": outcome.run_paths.root.name,
        "run_dir": str(outcome.run_paths.root),
        "exit_code": outcome.exit_code,
        "stages": outcome.stage_statuses,
        "fatal_error": outcome.fatal_error,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        parser.print_help()
        return EXIT_FATAL
    if argv[0] not in {"run", "list-stages", "status", "plan", "-h", "--help"}:

        argv = ["run", *argv]
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_request:
        code = int(exit_request.code or EXIT_OK)


        return EXIT_OK if code == EXIT_OK else EXIT_FATAL

    if not hasattr(args, "func"):
        parser.print_help()
        return EXIT_FATAL

    try:
        return int(args.func(args))
    except KeyIDError as exc:
        print(f"keyid: {exc}", file=sys.stderr)
        return EXIT_FATAL
    except KeyboardInterrupt:
        print("keyid: interrupted", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":
    raise SystemExit(main())
