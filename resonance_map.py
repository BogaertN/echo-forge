import networkx as nx
from pyvis.network import Network
import sqlite3
from db import DB_PATH
import logging

logger = logging.getLogger(__name__)

class ResonanceMap:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.G = nx.Graph()

    def build_graph(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content, ghost_loop FROM journal_entries")
        entries = cursor.fetchall()
        cursor.execute("SELECT id, question, journal_entry_id FROM debate_logs")
        debates = cursor.fetchall()

        for entry in entries:
            self.G.add_node(f"entry_{entry[0]}", label=entry[1][:50], type='entry', ghost=entry[2])

        for debate in debates:
            self.G.add_node(f"debate_{debate[0]}", label=debate[1][:50], type='debate')
            if debate[2]:
                self.G.add_edge(f"debate_{debate[0]}", f"entry_{debate[2]}", type='resolution')

        # Add edges for ghost loops or tags (expand as needed)

    def visualize(self, output_file='resonance_map.html'):
        self.build_graph()
        nt = Network('800px', '1200px', notebook=True)
        nt.from_nx(self.G)
        nt.show_buttons(filter_=['physics'])
        nt.show(output_file)
        logger.info(f"Map saved to {output_file}")

    def export(self, format='graphml'):
        nx.write_graphml(self.G, 'resonance_map.graphml')
        logger.info("Graph exported")
