import yaml
import multiprocessing as mp
import logging
from logging.handlers import RotatingFileHandler
import time
import os
from db import init_db, DB_PATH  # Import DB init
import sqlite3

logger = logging.getLogger(__name__)
handler = RotatingFileHandler(os.path.join(os.environ['ECHO_FORGE_DATA_DIR'], 'echo_forge.log'), maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Orchestrator:
    def __init__(self):
        init_db()  # Ensure DB ready
        self.config = self.load_config('configs/agent_routing.yaml')
        self.conn = sqlite3.connect(DB_PATH)

    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def isolate_agent_call(self, agent_name, prompt, timeout=300):
        def run_agent(queue):
            try:
                model = self.config['agents'][agent_name]['model']
                # Simulate Ollama call (replace with ollama-python API)
                response = f"Mock response from {model} for prompt: {prompt}"  # Placeholder
                queue.put(response)
            except Exception as e:
                queue.put(str(e))

        queue = mp.Queue()
        p = mp.Process(target=run_agent, args=(queue,))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f"Agent {agent_name} timed out")
        result = queue.get()
        if isinstance(result, str) and result.startswith("Error"):
            raise RuntimeError(result)
        return result

    def run_flow(self, user_input):
        try:
            # Clarify
            clarified = self.isolate_agent_call('clarifier', user_input)
            logger.info(f"Clarified: {clarified}")
            # Debate (simplified)
            proponent = self.isolate_agent_call('proponent', clarified)
            opponent = self.isolate_agent_call('opponent', clarified)
            # Synthesize
            synthesis = self.isolate_agent_call('decider', f"Pro: {proponent} Opp: {opponent}")
            # Audit
            self.isolate_agent_call('auditor', synthesis)
            # Journal
            self.log_to_journal(synthesis)
            return synthesis
        except Exception as e:
            logger.error(f"Flow error: {str(e)}")
            self.log_audit('error', str(e))
            raise

    def log_to_journal(self, content):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO journal_entries (content) VALUES (?)", (content,))
        self.conn.commit()

    def log_audit(self, event, details):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO audit_logs (event_type, details) VALUES (?, ?)", (event, details))
        self.conn.commit()

if __name__ == "__main__":
    orch = Orchestrator()
    result = orch.run_flow("Test input")
    print(result)
