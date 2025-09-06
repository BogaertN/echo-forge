import yaml
import multiprocessing as mp
import logging
from logging.handlers import RotatingFileHandler
import os
from db import init_db, DB_PATH
import sqlite3

from agents.stoic_clarifier import StoicClarifier
from agents.proponent import Proponent
from agents.opponent import Opponent
from agents.decider import Decider
from agents.auditor import Auditor
from agents.journaling_assistant import JournalingAssistant
from journal import Journal
from resonance_map import ResonanceMap
from tools import ToolManager, web_search  # Added for step 130

logger = logging.getLogger(__name__)
handler = RotatingFileHandler(os.path.join(os.environ['ECHO_FORGE_DATA_DIR'], 'echo_forge.log'), maxBytes=10*1024*1024, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class Orchestrator:
    def __init__(self):
        init_db()
        self.config = self.load_config('configs/agent_routing.yaml')
        self.conn = sqlite3.connect(DB_PATH)
        self.journal_assistant = JournalingAssistant(self.config['agents']['journaling_assistant']['model'])
        self.journal = Journal()
        self.map = ResonanceMap()
        self.tools = ToolManager()  # Added for step 130
        self.tools.register_tool('web_search', web_search)  # Added for step 130

    def load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def isolate_agent_call(self, agent_instance, method, args, timeout=300):
        def run_agent(queue):
            try:
                response = getattr(agent_instance, method)(*args)
                queue.put(response)
            except Exception as e:
                queue.put(str(e))

        queue = mp.Queue()
        p = mp.Process(target=run_agent, args=(queue,))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            raise TimeoutError(f"Agent timed out")
        result = queue.get()
        if isinstance(result, str) and "Error" in result:
            raise RuntimeError(result)
        return result

    def run_flow(self, user_input):
        try:
            clarifier = StoicClarifier(self.config['agents']['clarifier']['model'])
            clarified = self.isolate_agent_call(clarifier, 'clarify', (user_input,))
            logger.info(f"Clarified: {clarified}")

            proponent = Proponent(self.config['agents']['proponent']['model'])
            pro_arg = self.isolate_agent_call(proponent, 'argue', (clarified,))

            opponent = Opponent(self.config['agents']['opponent']['model'])
            opp_arg = self.isolate_agent_call(opponent, 'argue', (clarified,))

            decider = Decider(self.config['agents']['decider']['model'])
            synthesis = self.isolate_agent_call(decider, 'synthesize', (pro_arg, opp_arg))

            # Added for step 130: Example tool use if enabled
            if self.tools.enabled:
                search_result = self.tools.call_tool('web_search', clarified)
                synthesis += f"\nAdditional info from web: {search_result}"

            auditor = Auditor(self.config['agents']['auditor']['model'])
            self.isolate_agent_call(auditor, 'check', (synthesis,))

            journal_entry = self.journal_assistant.rephrase(synthesis)
            journal_id = self.journal.create_entry(journal_entry, metadata='{"example": "data"}', tags='test,tag', ghost_loop=False)
            self.map.visualize()
            return synthesis
        except Exception as e:
            logger.error(f"Flow error: {str(e)}")
            self.log_audit('error', str(e))
            raise

    def log_audit(self, event, details):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO audit_logs (event_type, details) VALUES (?, ?)", (event, details))
        self.conn.commit()

if __name__ == "__main__":
    orch = Orchestrator()
    result = orch.run_flow("Test input")
    print(result)
