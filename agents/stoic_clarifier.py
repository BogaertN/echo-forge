from .base_agent import BaseAgent

class StoicClarifier(BaseAgent):
    def clarify(self, question):
        system = """
        You are a Stoic clarifier using Socratic method. Do not answer the question. Ask probing questions to refine and clarify the user's intent. Continue until the question is crystal clear. Output only the clarifying questions.
        """
        response = self.generate(question, system)
        # Loop logic (simplified; in full, check if clarified)
        return response
