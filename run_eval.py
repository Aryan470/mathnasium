from __future__ import annotations
import argparse, json, sys, os, pathlib
from typing import Any, Dict
from eval.eval_core import ProblemLoader, ProofEvaluator, PromptBuilder, LeanSession, summarize, load_model

# LeanSession.configure_local_repo("/home/aryan/lean/mathlib4")
LeanSession.disable_local_repo()


def parse_kv_pairs(pairs):
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Bad --model-arg '{p}', expected key=value")
        k, v = p.split("=", 1)
        # Best-effort JSON parse; fallback to string
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Run model-agnostic LeanDojo evaluation")
    ap.add_argument("--test-path", required=True, help="Path to JSON/JSONL test set (LeanDojo-style problems)")
    ap.add_argument("--out-path", required=True, help="Where to write results JSONL")
    ap.add_argument("--model", required=True, help="Dotted path to model class, e.g., 'my_pkg.models:MyModel'")
    ap.add_argument("--model-arg", action="append", help="Model constructor arg as key=value (JSON allowed)")
    ap.add_argument("--max-steps", type=int, default=64, help="Max tactic steps per problem")
    ap.add_argument("--sleep-between", type=float, default=0.0, help="Sleep seconds between model calls")
    ap.add_argument("--num-problems", type=int, default=None, help="Number of problems to evaluate")
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Load problems
    # -------------------------------------------------------------------------
    problems = ProblemLoader(args.test_path).load()
    # problems = [problems[9]]
    print(problems)
    if args.num_problems is not None:
        problems = problems[:args.num_problems]

    if not problems:
        print(f"[run_eval] No problems loaded from {args.test_path}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model_kwargs = parse_kv_pairs(args.model_arg)
    model = load_model(args.model, **model_kwargs)

    # -------------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------------
    evaluator = ProofEvaluator(
        model,
        lean=LeanSession(),
        prompt_builder=PromptBuilder(),
        max_steps=args.max_steps,
        sleep_between=args.sleep_between,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        attempts_per_worker=args.attempts_per_worker,
    )

    results = evaluator.run(problems)

    # Safety check: if nothing came back, warn loudly
    if not results:
        print("[run_eval] WARNING: evaluator.run returned 0 results", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Prepare results data (one JSON object per attempt)
    # -------------------------------------------------------------------------
    results_data = []
    for r in results:
        results_data.append({
            "problem": dataclasses.asdict(r.problem),
            "status": r.status,
            "steps": r.steps,
            "time_sec": r.time_sec,
            "proof_script": r.proof_script,
            "error": r.error,
            "meta": r.meta,
            "model": getattr(model, "name", args.model),
        })

    out_p = pathlib.Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Write results as JSONL (one object per line)
    # -------------------------------------------------------------------------
    with out_p.open("w", encoding="utf-8") as f:
        for row in results_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------------
    # Calculate and write summary statistics
    # -------------------------------------------------------------------------
    summary = summarize(results, problems)

    summary_path = out_p.parent / f"{out_p.stem}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Print summary to console
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Problems: {summary['total_problems']}")
    print(f"Total Attempts: {summary['total_attempts']}")
    print(f"\nProblem-Level Statistics:")
    print(f"  Problems Solved: {summary['problems_solved']} ({summary['problem_success_rate'] * 100:.2f}%)")
    print(f"  Problems Failed: {summary['problems_failed']}")
    print(f"  Problems Skipped: {summary['problems_skipped']}")
    total_categorized = summary['problems_solved'] + summary['problems_failed'] + summary['problems_skipped']
    if total_categorized != summary['total_problems']:
        print(f"  ⚠️  WARNING: Problem counts don't add up! ({total_categorized} != {summary['total_problems']})")
    print(f"\nAttempt-Level Statistics:")
    print(f"  Attempts Successful: {summary['success']} ({summary['success_rate'] * 100:.2f}%)")
    print(f"  Average Steps per Attempt: {summary['avg_steps']:.2f}")
    print(f"\nTiming Statistics:")
    print(f"  Total Time: {summary['time_total_sec']:.2f} seconds ({summary['time_total_sec'] / 60:.2f} minutes)")
    print(f"  Average Time per Attempt: {summary['time_avg_sec']:.2f} seconds")
    print(f"  Average Time per Problem: {summary['time_avg_per_problem']:.2f} seconds")
    print(f"\nStatus Breakdown:")
    for status, count in summary['by_status'].items():
        print(f"  {status}: {count}")
    total_by_status = sum(summary['by_status'].values())
    if total_by_status != summary['total_attempts']:
        print(f"  ⚠️  WARNING: Status breakdown doesn't match total attempts! ({total_by_status} != {summary['total_attempts']})")
    print("=" * 80)
    print(f"\nResults written to: {out_p}")
    print(f"Summary written to: {summary_path}")

if __name__ == "__main__":
    import dataclasses  # local import for JSON dump
    main()
