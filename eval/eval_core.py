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
        self.leandojo = None
        self._dojo_ctx = None
        self._dojo = None
        self._state = None
        self._problem: Optional[Problem] = None
        self.last_error: Optional[str] = None
        try:
            import lean_dojo  # type: ignore[import]
            self.leandojo = lean_dojo
            # Cache commonly used classes/functions for convenience.
            self._LeanGitRepo = lean_dojo.LeanGitRepo
            self._Theorem = lean_dojo.Theorem
            self._Dojo = lean_dojo.Dojo
            self._TacticState = lean_dojo.TacticState
            self._ProofFinished = lean_dojo.ProofFinished
            self._ProofGivenUp = lean_dojo.ProofGivenUp
            self._LeanError = lean_dojo.LeanError
            self.available = True
        except Exception as exc:  # pragma: no cover - best effort handling
            self.last_error = str(exc)

    # ------------------------------------------------------------------ #
    # lifecycle helpers
    def close(self) -> None:
        """Close any active Dojo context."""
        if self._dojo_ctx is not None:
            try:
                self._dojo_ctx.__exit__(None, None, None)
            finally:
                self._dojo_ctx = None
                self._dojo = None
                self._state = None
                self._problem = None

    def __del__(self) -> None:  # pragma: no cover - cleanup on GC
        try:
            self.close()
        except Exception:
            pass

    def reset(self, problem: Problem) -> Dict[str, Any]:
        """Initialize Lean state for a problem. Returns an initial proof_state dict."""
        if not self.available:
            return {
                "ok": False,
                "reason": self.last_error or "LeanDojo not available",
                "goals": None,
            }

        # Ensure previous sessions are cleaned up before starting a new one.
        self.close()
        self._problem = problem

        try:
            repo = self._LeanGitRepo(problem.url, problem.commit)
            theorem = self._Theorem(repo, problem.file_path, problem.full_name)
            dojo_ctx = self._Dojo(theorem)
            dojo, state = dojo_ctx.__enter__()
        except Exception as exc:
            self.last_error = str(exc)
            self.close()
            return {
                "ok": False,
                "reason": self.last_error,
                "goals": None,
            }

        self._dojo_ctx = dojo_ctx
        self._dojo = dojo
        self._state = state
        goals = self._format_goals(state)
        return {"ok": True, "goals": goals, "cursor": problem.start}

    def apply_tactic(self, tactic: str) -> Dict[str, Any]:
        """Apply a tactic. Returns a dict with new state or error."""
        if not self.available:
            return {"ok": False, "error": "lean_not_available"}
        if self._dojo is None or self._state is None:
            return {"ok": False, "error": "session_not_initialized"}

        try:
            result = self._dojo.run_tac(self._state, tactic)
        except Exception as exc:
            self.last_error = str(exc)
            return {"ok": False, "error": self.last_error}

        # Handle LeanDojo result types.
        if isinstance(result, self._LeanError):
            self.last_error = result.error
            return {"ok": False, "error": result.error}

        if isinstance(result, self._ProofGivenUp):
            self.last_error = "proof_given_up"
            self._state = None
            return {"ok": False, "error": "proof_given_up"}

        if isinstance(result, self._ProofFinished):
            self._state = None
            return {"ok": True, "goals": [], "finished": True}

        if isinstance(result, self._TacticState):
            self._state = result
            goals = self._format_goals(result)
            return {"ok": True, "goals": goals}

        # Fallback: unknown result type, attempt to stringify and continue.
        self._state = result  # keep reference for potential debugging
        display = getattr(result, "pp", None)
        if callable(display):
            try:
                display = display()
            except Exception:  # pragma: no cover
                display = None
        goals = [display] if isinstance(display, str) and display else []
        return {"ok": True, "goals": goals}

    # ------------------------------------------------------------------ #
    # internal helpers
    def _format_goals(self, state: Any) -> List[str]:
        """Convert a LeanDojo tactic state into a list of human-readable goals."""
        if state is None:
            return []
        goals: List[str] = []

        # Prefer an explicit goals list when available.
        raw_goals = getattr(state, "goals", None)
        if raw_goals:
            for g in raw_goals:
                text = getattr(g, "pp", None)
                if callable(text):
                    try:
                        text = text()
                    except Exception:  # pragma: no cover
                        text = None
                if isinstance(text, str) and text.strip():
                    goals.append(text)
                else:
                    goals.append(str(g))

        if goals:
            return goals

        # Fall back to pretty-printed state.
        text = getattr(state, "pp", None)
        if callable(text):
            try:
                text = text()
            except Exception:  # pragma: no cover
                text = None
        if isinstance(text, str) and text.strip():
            return [text]
        return []

    def is_successful(self) -> Optional[bool]:
        """Return LeanDojo's success flag if a session is active."""
        if self._dojo is None:
            return None
        return getattr(self._dojo, "is_successful", None)

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
        meta: Dict[str, Any] = {}
        try:
            state = self.lean.reset(problem)
            if not state.get("ok"):
                meta["reset_error"] = state.get("reason")
                meta["lean_success"] = self.lean.is_successful()
                return AttemptResult(
                    problem,
                    "skipped",
                    0,
                    time.time() - t0,
                    proof_script,
                    state.get("reason"),
                    meta,
                )
            goals = state.get("goals") or []
            steps = 0
            while goals and steps < self.max_steps:
                prompt = self.prompt_builder.build(problem, goals)
                tactic = self.model.generate(prompt, **self.gen_kwargs).strip()
                print(f"Prompt: {prompt}")
                print(f"Tactic: {tactic}")
                proof_script.append(tactic)
                res = self.lean.apply_tactic(tactic)
                if not res.get("ok"):
                    meta["lean_error"] = res.get("error")
                    meta["lean_success"] = self.lean.is_successful()
                    return AttemptResult(
                        problem,
                        "fail",
                        steps + 1,
                        time.time() - t0,
                        proof_script,
                        res.get("error"),
                        meta,
                    )
                goals = res.get("goals") or []
                steps += 1
                if self.sleep_between:
                    time.sleep(self.sleep_between)
            status = "success" if not goals else "timeout"
            meta["lean_success"] = self.lean.is_successful()
            return AttemptResult(
                problem,
                status,
                steps,
                time.time() - t0,
                proof_script,
                meta=meta,
            )
        except Exception as e:
            meta["exception_type"] = type(e).__name__
            meta["lean_success"] = self.lean.is_successful()
            return AttemptResult(
                problem,
                "error",
                len(proof_script),
                time.time() - t0,
                proof_script,
                error=str(e),
                meta=meta,
            )
        finally:
            self.lean.close()

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
