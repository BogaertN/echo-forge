from .base_agent import BaseAgent

class Auditor(BaseAgent):
    def check(self, text):
        system = "You are an auditor. Detect drift, hallucinations, contradictions in the text."
        return self.generate(text, system)
