from .base_agent import BaseAgent
import whisper

class JournalingAssistant(BaseAgent):
    def rephrase(self, text):
        system = "Rephrase the text for clarity and add metadata (tags, weights)."
        return self.generate(text, system)

    def transcribe_voice(self, audio_path):
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path)
        return result["text"]
