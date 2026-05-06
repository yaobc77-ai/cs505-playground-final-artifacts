from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agent.loop import SelfExplanatoryAgent
from env.toytextmdp import ToyTextMDP, load_tasks


def run_suite(
    *,
    tasks_path: Path,
    mode: str,
    out_path: Path,
    limit: int | None = None,
    seed: int = 7,
    memory_path: Path | None = None,
    audit_path: Path | None = None,
    selector_config: dict[str, float] | None = None,
    repeat_index: int | None = None,
) -> None:
    tasks = load_tasks(tasks_path)
    if limit is not None:
        tasks = tasks[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path = memory_path or out_path.with_name(f"{out_path.stem}_memory.jsonl")
    audit_path = audit_path or out_path.with_name(f"{out_path.stem}_audit.jsonl")
    memory_path.write_text("", encoding="utf-8")
    audit_path.write_text("", encoding="utf-8")

    agent = SelfExplanatoryAgent(
        mode=mode,
        memory_path=memory_path,
        audit_log_path=audit_path,
        seed=seed,
        selector_config=selector_config,
    )

    meta_record = {
        "record_type": "run_meta",
        "mode": mode,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tasks_path": str(tasks_path),
        "memory_path": str(memory_path),
        "audit_path": str(audit_path),
        "selector_config": selector_config or {},
        "repeat_index": repeat_index,
    }

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
        for task in tasks:
            env = ToyTextMDP(task)
            for record in agent.run_task(env):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ToyTextMDP evaluation suite.")
    parser.add_argument("--tasks", type=Path, required=True, help="Path to tasks.jsonl.")
    parser.add_argument(
        "--mode",
        choices=(
            "full",
            "no_reflection",
            "no_self_model",
            "no_cf_eval",
            "permute",
            "selector_penalty",
            "voi_coverage_only",
            "voi_gate_only",
            "voi_memory_v1",
            "subject_v0",
            "subject_no_other",
            "subject_no_commitment",
            "subject_no_memory_recompose",
        ),
        default="full",
    )
    parser.add_argument(
        "--cf_eval",
        choices=("standard", "permute"),
        default="standard",
        help="Compatibility flag for the PDF command examples.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Path to JSONL logs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional task limit.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--memory-path", type=Path, default=None)
    parser.add_argument("--audit-path", type=Path, default=None)
    parser.add_argument("--selector-penalty-base", type=float, default=0.10)
    parser.add_argument("--selector-penalty-memory-scale", type=float, default=0.45)
    parser.add_argument("--repeat-index", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = "permute" if args.cf_eval == "permute" else args.mode
    selector_config = None
    if mode == "selector_penalty":
        selector_config = {
            "penalty_base": args.selector_penalty_base,
            "penalty_memory_scale": args.selector_penalty_memory_scale,
        }
    run_suite(
        tasks_path=args.tasks,
        mode=mode,
        out_path=args.out,
        limit=args.limit,
        seed=args.seed,
        memory_path=args.memory_path,
        audit_path=args.audit_path,
        selector_config=selector_config,
        repeat_index=args.repeat_index,
    )


if __name__ == "__main__":
    main()
