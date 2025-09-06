import ollama
import logging

logger = logging.getLogger(__name__)

class BaseAgent:
    def __init__(self, model):
        self.model = model

    def generate(self, prompt, system=None):
        try:
            messages = []
            if system:
                messages.append({'role': 'system', 'content': system})
            messages.append({'role': 'user', 'content': prompt})
            response = ollama.chat(model=self.model, messages=messages)
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {str(e)}")
            raise
