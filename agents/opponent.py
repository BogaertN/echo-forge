from .base_agent import BaseAgent

class Opponent(BaseAgent):
    def argue(self, question):
        system = "You are an opponent. Provide critiques and alternatives to the question."
        return self.generate(question, system)
