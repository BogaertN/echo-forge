import logging
import json
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from encrypted_db import EncryptedDB
from agents import JournalingAssistant  # For metadata generation in linking
from config import load_config
from utils import calculate_edge_strength  # Assume utility for strength calc

logger = logging.getLogger(__name__)

class ResonanceMap:
    """
    Manages resonance mapping: graph structure for entries/debates/ghost loops,
    visualization UI logic, linking, navigation of open loops, ontology growth,
    personal knowledge graph, and exporting maps/snapshots.
    Integrates with DB for persistence.
    """
    
    def __init__(self, db: EncryptedDB):
        self.db = db
        self.config = load_config()
        self.assistant = JournalingAssistant(self.config['models']['journaling_assistant'])
        
        logger.info("ResonanceMap initialized")
    
    # Graph Structure Management
    
    def add_node(self, node_type: str, content_summary: str, related_id: Optional[str] = None) -> str:
        """
        Add a new node to the resonance map.
        
        Args:
            node_type: 'entry', 'debate', 'ghost_loop'
            content_summary: Brief summary for visualization
            related_id: ID of related journal entry or debate
        
        Returns:
            Node ID
        """
        node_id = self.db.add_resonance_node(node_type, content_summary)
        
        if related_id:
            # Auto-link to related item
            self.add_edge(related_id, node_id, 'references', strength=1.0)
        
        logger.info(f"Node added: {node_id} ({node_type})")
        return node_id
    
    def add_edge(self, from_id: str, to_id: str, relation_type: str, strength: Optional[float] = None):
        """
        Add or update edge between nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            relation_type: 'resolves', 'contradicts', 'relates_to', 'builds_on', etc.
            strength: Edge strength (0-1), auto-calculated if None
        """
        if strength is None:
            # Auto-calculate strength using utility or agent
            from_summary = self.db.get_node_summary(from_id)  # Assume DB method
            to_summary = self.db.get_node_summary(to_id)
            strength = calculate_edge_strength(from_summary, to_summary)
        
        self.db.add_resonance_edge(from_id, to_id, relation_type, strength)
        
        logger.info(f"Edge added: {from_id} -> {to_id} ({relation_type}, strength={strength})")
    
    def auto_link_entries(self, new_entry_id: str):
        """Automatically link new entry to existing nodes using agent"""
        new_entry = self.db.get_journal_entry(new_entry_id)
        existing_nodes = self.db.get_all_nodes()  # Assume DB method
        
        for node in existing_nodes:
            similarity = self.assistant.calculate_similarity(  # Assume extended agent method
                new_entry['summary'],
                node['content_summary']
            )
            if similarity > self.config['resonance']['link_threshold']:
                relation = self.assistant.determine_relation(
                    new_entry['summary'],
                    node['content_summary']
                )
                self.add_edge(new_entry_id, node['id'], relation, similarity)
    
    # Visualization and UI Logic
    
    def generate_map_view(self, center_node: Optional[str] = None, filter_types: List[str] = None,
                          highlight_ghosts: bool = True) -> Dict:
        """
        Generate data for map visualization.
        
        Args:
            center_node: Optional center node for filtered view
            filter_types: List of node types to include
            highlight_ghosts: Highlight ghost loops
        
        Returns:
            Graph data for UI (nodes, edges, layout hints)
        """
        graph = self.db.get_resonance_map(center_node)
        
        # Apply filters
        if filter_types:
            graph['nodes'] = {k: v for k, v in graph['nodes'].items() if v['type'] in filter_types}
            graph['edges'] = [e for e in graph['edges'] if e['from'] in graph['nodes'] and e['to'] in graph['nodes']]
        
        # Add highlights
        if highlight_ghosts:
            for node_id, node in graph['nodes'].items():
                if self.db.is_ghost_loop(node_id):  # Assume DB method
                    node['highlight'] = True
        
        # Generate layout hints (simple force-directed simulation placeholder)
        graph['layout'] = self._compute_layout(graph)
        
        return graph
    
    def _compute_layout(self, graph: Dict) -> Dict:
        """Compute simple layout for visualization (placeholder for graph lib)"""
        # TODO: Integrate networkx or similar for real layout
        layout = {}
        for i, node_id in enumerate(graph['nodes']):
            layout[node_id] = {
                'x': i * 100,  # Simple linear layout
                'y': 0
            }
        return layout
    
    # Ghost Loop Management
    
    def navigate_open_loops(self) -> List[Dict]:
        """Get navigable list of open ghost loops"""
        ghost_loops = self.db.find_ghost_loops()
        
        # Sort by priority/age
        ghost_loops.sort(key=lambda g: (g['weights']['priority'], g['created_at']), reverse=True)
        
        return ghost_loops
    
    def close_loop(self, loop_id: str, resolution_entry_id: str):
        """Close a ghost loop by linking to resolution"""
        self.add_edge(resolution_entry_id, loop_id, 'resolves')
        self.db.update_journal_entry(loop_id, {'ghost_loop': False})
        
        # Award gamification
        self.gamification.award_loop_closure()  # Assume integration
        
        logger.info(f"Ghost loop closed: {loop_id} resolved by {resolution_entry_id}")
    
    # Ontology Growth and Knowledge Graph
    
    def grow_ontology(self, new_entry_id: str):
        """Grow personal ontology by extracting concepts and linking"""
        entry = self.db.get_journal_entry(new_entry_id)
        concepts = self.assistant.extract_concepts(entry['content'])  # Assume agent method
        
        for concept in concepts:
            concept_node_id = self.add_node('concept', concept)
            self.add_edge(new_entry_id, concept_node_id, 'mentions')
        
        logger.info(f"Ontology grown with {len(concepts)} new concepts from entry {new_entry_id}")
    
    def query_knowledge_graph(self, query: str) -> List[Dict]:
        """Query the knowledge graph for related knowledge"""
        # Simple search; extend with graph queries
        results = self.db.search_graph(query)  # Assume DB method for graph search
        return results
    
    # Exporting
    
    def export_map(self, format: str = 'json', node_id: Optional[str] = None) -> str:
        """Export resonance map or snapshot"""
        graph = self.generate_map_view(center_node=node_id)
        data = json.dumps(graph, indent=2)
        
        file_path = f"resonance_map_{datetime.now().strftime('%Y%m%d')}.{format}"
        with open(file_path, 'w') as f:
            f.write(data)
        
        logger.info(f"Resonance map exported to {file_path}")
        return file_path
    
    def export_knowledge_snapshot(self) -> str:
        """Export current knowledge graph snapshot"""
        full_graph = self.db.get_resonance_map()
        return self.export_map(graph=full_graph)  # Overload or separate
    
    # Maintenance
    
    def prune_weak_edges(self, threshold: float = 0.3):
        """Prune weak edges to optimize graph"""
        edges = self.db.get_all_edges()  # Assume DB method
        for edge in edges:
            if edge['strength'] < threshold:
                self.db.remove_edge(edge['from_id'], edge['to_id'], edge['relation_type'])
        
        logger.info(f"Pruned edges below strength {threshold}")
    
    def visualize_closures(self) -> Dict:
        """Visualize ghost loop closures over time"""
        closures = self.db.get_closed_loops()  # Assume DB query
        timeline = {}
        for closure in closures:
            date = closure['updated_at'][:10]
            timeline[date] = timeline.get(date, 0) + 1
        
        return timeline
