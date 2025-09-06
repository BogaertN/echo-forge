from .base_agent import BaseAgent

class Proponent(BaseAgent):
    def argue(self, question):
        system = "You are a proponent. Build strong, logical arguments in favor of the question."
        return self.generate(question, system)
