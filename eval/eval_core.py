from __future__ import annotations
import json, time, importlib, dataclasses, traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -------- Model Interface (assumed implemented by you) --------
class BaseModelInterface:
    """Minimal interface that any model must implement."""
    name: str
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

# -------- Problem representation & loading --------
@dataclass
class Problem:
    url: str
    commit: str
    file_path: str
    full_name: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    traced_tactics: List[Dict[str, Any]] = field(default_factory=list)

class ProblemLoader:
    """Loads LeanDojo-style problems from a JSON or JSONL file.

    Expected schema for each item:
    {
      "url": "...",
      "commit": "...",
      "file_path": "Mathlib/...",
      "full_name": "Namespace.theorem_name",
      "start": [line, col],
      "end": [line, col],
      "traced_tactics": [ ... ]  # optional
    }
    """
    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Problem]:
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        items: List[Dict[str, Any]]
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                items = [data]
            else:
                items = list(data)
        except Exception:
            # try JSONL
            items = []
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                items.append(json.loads(ln))
        problems = [Problem(**item) for item in items]
        return problems

# -------- Lean session wrapper --------
class LeanSession:
    """Thin wrapper around LeanDojo/Lean 4.
    If LeanDojo is installed, use it. Otherwise, provide a no-op that marks attempts as 'skipped'.
    """
    def __init__(self):
        self.available = False
        self.env = None
        try:
            import leandojo  # type: ignore
            self.available = True
            self.leandojo = leandojo
        except Exception:
            self.available = False
            self.leandojo = None

    def reset(self, problem: Problem) -> Dict[str, Any]:
        """Initialize Lean state for a problem. Returns an initial proof_state dict."""
        if not self.available:
            return {"ok": False, "reason": "LeanDojo not available", "goals": None}
        # TODO: integrate with LeanDojo's APIs: checkout commit, open file, position at problem.start, etc.
        return {"ok": True, "goals": ["<goal>"], "cursor": problem.start}

    def apply_tactic(self, tactic: str) -> Dict[str, Any]:
        """Apply a tactic. Returns a dict with new state or error."""
        if not self.available:
            return {"ok": False, "error": "lean_not_available"}
        # TODO: call LeanDojo to apply the tactic and fetch next goals.
        # For now: pretend 'admit'/'exact'/'done' closes all goals.
        if "admit" in tactic or "exact" in tactic or "done" in tactic:
            return {"ok": True, "goals": []}
        return {"ok": True, "goals": ["<goal>"]}

# -------- Prompt building --------
class PromptBuilder:
    """Formats the prompt from a problem & current proof state."""
    def __init__(self, template: Optional[str] = None):
        self.template = template or (
            "You are a Lean 4 prover. Prove the theorem:\n"
            "full_name: {full_name}\n"
            "file: {file_path}\n"
            "commit: {commit}\n"
            "Current goals: {goals}\n"
            "Return ONE Lean tactic or a short line of Lean code to advance the proof."
        )

    def build(self, problem: Problem, goals: List[str]) -> str:
        return self.template.format(
            full_name=problem.full_name,
            file_path=problem.file_path,
            commit=problem.commit,
            goals=goals,
        )

# -------- Evaluator & metrics --------
@dataclass
class AttemptResult:
    problem: Problem
    status: str  # "success", "fail", "skipped", "timeout", "error"
    steps: int
    time_sec: float
    proof_script: List[str] = field(default_factory=list)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class ProofEvaluator:
    def __init__(
        self,
        model: BaseModelInterface,
        lean: Optional[LeanSession] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        max_steps: int = 64,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        sleep_between: float = 0.0,
    ):
        self.model = model
        self.lean = lean or LeanSession()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_steps = max_steps
        self.gen_kwargs = gen_kwargs or {}
        self.sleep_between = sleep_between

    def run_attempt(self, problem: Problem) -> AttemptResult:
        t0 = time.time()
        proof_script: List[str] = []
        try:
            state = self.lean.reset(problem)
            if not state.get("ok"):
                return AttemptResult(problem, "skipped", 0, time.time()-t0, [], state.get("reason"))
            goals = state.get("goals") or []
            steps = 0
            while goals and steps < self.max_steps:
                prompt = self.prompt_builder.build(problem, goals)
                tactic = self.model.generate(prompt, **self.gen_kwargs).strip()
                proof_script.append(tactic)
                res = self.lean.apply_tactic(tactic)
                if not res.get("ok"):
                    return AttemptResult(problem, "fail", steps+1, time.time()-t0, proof_script, res.get("error"))
                goals = res.get("goals") or []
                steps += 1
                if self.sleep_between:
                    time.sleep(self.sleep_between)
            status = "success" if not goals else "timeout"
            return AttemptResult(problem, status, steps, time.time()-t0, proof_script)
        except Exception as e:
            return AttemptResult(problem, "error", len(proof_script), time.time()-t0, proof_script, error=str(e))

    def run(self, problems: Iterable[Problem]) -> List[AttemptResult]:
        results: List[AttemptResult] = []
        for p in problems:
            results.append(self.run_attempt(p))
        return results

# -------- Aggregation --------
def summarize(results: List[AttemptResult]) -> Dict[str, Any]:
    total = len(results)
    succ = sum(1 for r in results if r.status == "success")
    time_sum = sum(r.time_sec for r in results)
    avg_steps = (sum(r.steps for r in results) / total) if total else 0.0
    return {
        "total": total,
        "success": succ,
        "success_rate": (succ / total) if total else 0.0,
        "avg_steps": avg_steps,
        "time_total_sec": time_sum,
        "time_avg_sec": (time_sum / total) if total else 0.0,
        "by_status": {s: sum(1 for r in results if r.status == s)
                      for s in sorted(set(r.status for r in results))},
    }

# -------- Utilities --------
def load_model(dotted_class: str, **kwargs) -> BaseModelInterface:
    """Load a model class from 'package.module:ClassName' and instantiate it with kwargs."""
    if ":" in dotted_class:
        module_name, cls_name = dotted_class.split(":", 1)
    else:
        parts = dotted_class.split(".")
        module_name, cls_name = ".".join(parts[:-1]), parts[-1]
    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    return cls(**kwargs)  # type: ignore
