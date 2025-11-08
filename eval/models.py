# my_pkg/models.py
import json
from eval.eval_core import BaseModelInterface
from misc.test_qwen import generate_with_tps, load_model

class MockModel(BaseModelInterface):
    def __init__(self, name="mock", **kwargs):
        self._name = name
    @property
    def name(self): return self._name
    def generate(self, prompt: str, **kwargs) -> str:
        # Naive behavior: try to "finish" quickly
        return "admit"

    
class QwenModel(BaseModelInterface):
    def __init__(
        self,
        model="Qwen/Qwen2.5-7B-Instruct",
        fourbit=False,
        system="You are a mathematical proof expert.",
        max_new=512,
        temp=0.7,
        top_p=0.9,
        response_template=(
            "You are a Lean 4 prover. You must output exactly ONE line of Lean code that advances the proof."
            "Output rules (HARD):"
            "- Output a single line, no backticks, no explanation."
            "- ASCII only. Never use dagger/superscripts or fancy Unicode (e.g. ‘inst✝’). "
            "- Never reference pretty-printed names like inst✝, inst✝¹, etc. Use `inferInstance`, `assumption`, or class projections."
            "- Prefer method-style lemmas on hypotheses (e.g. `hg.comp_*`) and `simpa [Function.comp]` to match `(g ∘ f)`."
            "- For class fields: use `ClassName.field` or `(inferInstance : ClassName _ _).field`."
            "- If goal matches a hypothesis: use `assumption` or `exact ‹_›`."
            "- If rewriting inside `<`/`≤` goals fails, use `mul_lt_mul_left` / `mul_le_mul_left` with positivity side-conditions."
            "- Do NOT invent identifiers. Only use names that appear in the context or in mathlib."
            "- Acceptable atoms: `intro`, `apply`, `refine`, `exact`, `simp`, `rw`, `have`, `calc`, `convert`, `cases`, `rcases`, `linarith`, `nlinarith`, `ring`, `field_simp`, `norm_cast`, `conv`, `simpa`."
            "- If unsure about instance names: use `exact inferInstance`."
            "Return exactly one tactic/term line. No comments."

        ),
        **kwargs,
    ):
        self._name = "qwen"
        self.model = model
        self.fourbit = fourbit
        self.system = system
        self.max_new = max_new
        self.temp = temp
        self.top_p = top_p
        self.response_template = response_template
        self.tok, self.model = load_model(self.model, self.fourbit)
    @property
    def name(self): return self._name
    def generate(self, prompt: str, **kwargs) -> str:
        messages = [
            {
                "role": "system",
                "content": f"{self.system}\n{self.response_template}",
            },
            {"role": "user", "content": prompt},
        ]
        reply, gen_len, tok_per_sec = generate_with_tps(
            self.model,
            self.tok,
            messages,
            self.max_new,
            self.temp,
            self.top_p,
        )

        reply = reply.strip()
        # Attempt to parse a JSON object with a `tactic` field.
        tactic = None
        try:
            data = json.loads(reply)
            if isinstance(data, dict):
                tactic = data.get("tactic") or data.get("command") or data.get("lean_code")
        except json.JSONDecodeError:
            # fallback: look for first JSON block in text
            start = reply.find("{")
            end = reply.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = reply[start : end + 1]
                try:
                    data = json.loads(snippet)
                    if isinstance(data, dict):
                        tactic = data.get("tactic") or data.get("command") or data.get("lean_code")
                except Exception:
                    pass

        if not tactic:
            # As a last resort, fall back to the raw reply (trim to first line).
            return reply.splitlines()[0].strip()

        return str(tactic).strip()