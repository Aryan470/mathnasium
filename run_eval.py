from __future__ import annotations
import argparse, json, sys, os, pathlib
from typing import Any, Dict
from eval.eval_core import ProblemLoader, ProofEvaluator, PromptBuilder, LeanSession, summarize, load_model

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

    problems = ProblemLoader(args.test_path).load()
    if args.num_problems is not None:
        problems = problems[:args.num_problems]
    model_kwargs = parse_kv_pairs(args.model_arg)
    model = load_model(args.model, **model_kwargs)
    evaluator = ProofEvaluator(model, lean=LeanSession(), prompt_builder=PromptBuilder(),
                               max_steps=args.max_steps, sleep_between=args.sleep_between)

    results = evaluator.run(problems)
    # Write JSONL
    out_p = pathlib.Path(args.out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "problem": dataclasses.asdict(r.problem),
                "status": r.status,
                "steps": r.steps,
                "time_sec": r.time_sec,
                "proof_script": r.proof_script,
                "error": r.error,
                "meta": r.meta,
                "model": getattr(model, "name", args.model),
            }) + "\n")

    summary = summarize(results)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import dataclasses  # local import for JSON dump
    main()
