from .base_agent import BaseAgent

class Decider(BaseAgent):
    def synthesize(self, pro, opp):
        system = "You are a neutral decider. Synthesize pro and opp arguments into a balanced insight."
        prompt = f"Pro: {pro}\nOpp: {opp}"
        return self.generate(prompt, system)
