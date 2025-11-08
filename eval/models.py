# my_pkg/models.py
from eval.eval_core import BaseModelInterface

class MockModel(BaseModelInterface):
    def __init__(self, name="mock", **kwargs):
        self._name = name
    @property
    def name(self): return self._name
    def generate(self, prompt: str, **kwargs) -> str:
        # Naive behavior: try to "finish" quickly
        return "admit"
