import asyncio
import json
import logging
import math
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import hashlib
import statistics

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

from utils import (
    extract_key_concepts, calculate_similarity, generate_uuid,
    clean_text, normalize_text, extract_noun_phrases
)

logger = logging.getLogger(__name__)

@dataclass
class ResonanceNode:
    """Node in the resonance map representing a concept, theme, or insight"""
    id: str
    session_id: str
    node_type: str  # concept, theme, question, insight, debate_point
    title: str
    description: str
    content_hash: str
    strength: float  # Node importance/strength (0.0 - 1.0)
    frequency: int  # How often referenced
    last_activated: datetime
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            "last_activated": self.last_activated.isoformat(),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResonanceNode':
        """Create from dictionary"""
        data = data.copy()
        data["last_activated"] = datetime.fromisoformat(data["last_activated"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

@dataclass
class ResonanceConnection:
    """Connection between two nodes in the resonance map"""
    id: str
    session_id: str
    source_node_id: str
    target_node_id: str
    connection_type: str  # relates_to, contradicts, supports, leads_to, etc.
    strength: float  # Connection strength (0.0 - 1.0)
    created_from: str  # Source that created this connection
    created_at: datetime
    last_strengthened: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "last_strengthened": self.last_strengthened.isoformat()
        }

@dataclass
class GraphCluster:
    """Cluster of related nodes in the graph"""
    id: str
    nodes: List[str]
    central_concept: str
    strength: float
    density: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphMetrics:
    """Graph analysis metrics"""
    total_nodes: int
    total_edges: int
    density: float
    clustering_coefficient: float
    average_path_length: float
    connected_components: int
    largest_component_size: int
    central_nodes: List[Tuple[str, float]]  # (node_id, centrality_score)
    communities: List[GraphCluster]
    ghost_loops: List[Dict[str, Any]]

class ConceptExtractor:
    """Advanced concept extraction and processing"""
    
    def __init__(self):
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'this', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two',
            'more', 'very', 'when', 'come', 'its', 'now', 'over', 'think',
            'also', 'back', 'after', 'use', 'year', 'work', 'first', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
            'most', 'us'
        }
    
    def extract_concepts(self, text: str, min_length: int = 3, max_concepts: int = 20) -> List[Tuple[str, float]]:
        """
        Extract key concepts from text with importance scores.
        
        Args:
            text: Input text
            min_length: Minimum concept length
            max_concepts: Maximum number of concepts to return
            
        Returns:
            List of (concept, importance_score) tuples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        # Clean and normalize text
        cleaned_text = clean_text(text)
        
        # Extract noun phrases and key terms
        noun_phrases = extract_noun_phrases(cleaned_text)
        basic_concepts = extract_key_concepts(cleaned_text)
        
        # Combine and score concepts
        all_concepts = set()
        
        # Add noun phrases (higher weight)
        for phrase in noun_phrases:
            if len(phrase) >= min_length and phrase.lower() not in self.stop_words:
                all_concepts.add(phrase.lower().strip())
        
        # Add basic concepts
        for concept in basic_concepts:
            if len(concept) >= min_length and concept.lower() not in self.stop_words:
                all_concepts.add(concept.lower().strip())
        
        # Calculate importance scores using TF-IDF-like approach
        concept_scores = []
        word_freq = Counter(cleaned_text.lower().split())
        total_words = len(cleaned_text.split())
        
        for concept in all_concepts:
            # Calculate frequency-based score
            concept_words = concept.split()
            freq_score = sum(word_freq.get(word, 0) for word in concept_words)
            
            # Normalize by length and total words
            tf_score = freq_score / total_words
            
            # Length bonus (longer phrases often more meaningful)
            length_bonus = min(1.5, 1.0 + (len(concept_words) - 1) * 0.2)
            
            # Position bonus (concepts appearing early often more important)
            position_score = 1.0
            first_occurrence = cleaned_text.lower().find(concept)
            if first_occurrence >= 0:
                position_score = 1.0 - (first_occurrence / len(cleaned_text)) * 0.3
            
            final_score = tf_score * length_bonus * position_score
            concept_scores.append((concept, final_score))
        
        # Sort by score and return top concepts
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        return concept_scores[:max_concepts]
    
    def find_concept_relationships(self, concepts: List[str], text: str) -> List[Tuple[str, str, float]]:
        """
        Find relationships between concepts based on co-occurrence and proximity.
        
        Args:
            concepts: List of concepts to analyze
            text: Source text
            
        Returns:
            List of (concept1, concept2, relationship_strength) tuples
        """
        relationships = []
        
        if len(concepts) < 2:
            return relationships
        
        # Normalize text for analysis
        normalized_text = normalize_text(text)
        
        # Calculate co-occurrence matrix
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i + 1:], i + 1):
                strength = self._calculate_relationship_strength(
                    concept1, concept2, normalized_text
                )
                
                if strength > 0.1:  # Minimum threshold
                    relationships.append((concept1, concept2, strength))
        
        return relationships
    
    def _calculate_relationship_strength(self, concept1: str, concept2: str, text: str) -> float:
        """Calculate relationship strength between two concepts"""
        # Find all occurrences
        concept1_positions = self._find_concept_positions(concept1, text)
        concept2_positions = self._find_concept_positions(concept2, text)
        
        if not concept1_positions or not concept2_positions:
            return 0.0
        
        # Calculate proximity-based strength
        min_distance = float('inf')
        proximity_scores = []
        
        for pos1 in concept1_positions:
            for pos2 in concept2_positions:
                distance = abs(pos1 - pos2)
                min_distance = min(min_distance, distance)
                
                # Proximity score (closer = stronger relationship)
                if distance < 50:  # Within 50 characters
                    proximity_scores.append(1.0 - (distance / 50))
                elif distance < 200:  # Within 200 characters
                    proximity_scores.append(0.5 - (distance / 400))
        
        if not proximity_scores:
            return 0.0
        
        # Average proximity score
        avg_proximity = statistics.mean(proximity_scores)
        
        # Co-occurrence frequency bonus
        co_occurrence_bonus = min(1.0, len(proximity_scores) / 3.0)
        
        return min(1.0, avg_proximity * co_occurrence_bonus)
    
    def _find_concept_positions(self, concept: str, text: str) -> List[int]:
        """Find all positions where concept appears in text"""
        positions = []
        concept_lower = concept.lower()
        text_lower = text.lower()
        
        start = 0
        while True:
            pos = text_lower.find(concept_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions

class GraphAnalyzer:
    """Advanced graph analysis and metrics calculation"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
    
    def calculate_metrics(self) -> GraphMetrics:
        """Calculate comprehensive graph metrics"""
        try:
            # Basic metrics
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()
            
            if total_nodes == 0:
                return GraphMetrics(
                    total_nodes=0, total_edges=0, density=0.0,
                    clustering_coefficient=0.0, average_path_length=0.0,
                    connected_components=0, largest_component_size=0,
                    central_nodes=[], communities=[], ghost_loops=[]
                )
            
            # Density
            density = nx.density(self.graph) if total_nodes > 1 else 0.0
            
            # Clustering coefficient
            clustering_coeff = nx.average_clustering(self.graph) if total_nodes > 2 else 0.0
            
            # Connected components
            components = list(nx.connected_components(self.graph))
            connected_components = len(components)
            largest_component_size = len(max(components, key=len)) if components else 0
            
            # Average path length (for largest component)
            average_path_length = 0.0
            if largest_component_size > 1:
                largest_component = self.graph.subgraph(max(components, key=len))
                if nx.is_connected(largest_component):
                    average_path_length = nx.average_shortest_path_length(largest_component)
            
            # Centrality measures
            central_nodes = self._calculate_centrality()
            
            # Community detection
            communities = self._detect_communities()
            
            # Ghost loop detection
            ghost_loops = self._detect_ghost_loops()
            
            return GraphMetrics(
                total_nodes=total_nodes,
                total_edges=total_edges,
                density=density,
                clustering_coefficient=clustering_coeff,
                average_path_length=average_path_length,
                connected_components=connected_components,
                largest_component_size=largest_component_size,
                central_nodes=central_nodes,
                communities=communities,
                ghost_loops=ghost_loops
            )
            
        except Exception as e:
            logger.error(f"Error calculating graph metrics: {e}")
            return GraphMetrics(
                total_nodes=total_nodes, total_edges=total_edges, density=0.0,
                clustering_coefficient=0.0, average_path_length=0.0,
                connected_components=0, largest_component_size=0,
                central_nodes=[], communities=[], ghost_loops=[]
            )
    
    def _calculate_centrality(self) -> List[Tuple[str, float]]:
        """Calculate node centrality measures"""
        try:
            if self.graph.number_of_nodes() < 2:
                return []
            
            # Combine multiple centrality measures
            betweenness = nx.betweenness_centrality(self.graph)
            degree = nx.degree_centrality(self.graph)
            closeness = nx.closeness_centrality(self.graph)
            
            # Calculate composite centrality score
            centrality_scores = {}
            for node in self.graph.nodes():
                composite_score = (
                    betweenness.get(node, 0) * 0.4 +
                    degree.get(node, 0) * 0.3 +
                    closeness.get(node, 0) * 0.3
                )
                centrality_scores[node] = composite_score
            
            # Return top 10 most central nodes
            sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:10]
            
        except Exception as e:
            logger.error(f"Error calculating centrality: {e}")
            return []
    
    def _detect_communities(self) -> List[GraphCluster]:
        """Detect communities/clusters in the graph"""
        try:
            communities = []
            
            if self.graph.number_of_nodes() < 3:
                return communities
            
            # Use Louvain algorithm for community detection
            try:
                import community  # python-louvain
                partition = community.best_partition(self.graph)
                
                # Group nodes by community
                community_groups = defaultdict(list)
                for node, comm_id in partition.items():
                    community_groups[comm_id].append(node)
                
                # Create cluster objects
                for comm_id, nodes in community_groups.items():
                    if len(nodes) >= 2:  # Minimum cluster size
                        cluster = self._create_cluster(comm_id, nodes)
                        communities.append(cluster)
                        
            except ImportError:
                # Fallback: use connected components as communities
                components = nx.connected_components(self.graph)
                for i, component in enumerate(components):
                    if len(component) >= 2:
                        cluster = self._create_cluster(i, list(component))
                        communities.append(cluster)
            
            return communities
            
        except Exception as e:
            logger.error(f"Error detecting communities: {e}")
            return []
    
    def _create_cluster(self, cluster_id: int, nodes: List[str]) -> GraphCluster:
        """Create cluster object from nodes"""
        # Calculate cluster metrics
        subgraph = self.graph.subgraph(nodes)
        density = nx.density(subgraph) if len(nodes) > 1 else 0.0
        
        # Find central concept (highest degree node in cluster)
        degrees = dict(subgraph.degree())
        central_concept = max(degrees, key=degrees.get) if degrees else nodes[0]
        
        # Calculate cluster strength (average edge weight)
        edge_weights = []
        for u, v, data in subgraph.edges(data=True):
            edge_weights.append(data.get('strength', 1.0))
        
        strength = statistics.mean(edge_weights) if edge_weights else 0.0
        
        return GraphCluster(
            id=f"cluster_{cluster_id}",
            nodes=nodes,
            central_concept=central_concept,
            strength=strength,
            density=density,
            metadata={
                "node_count": len(nodes),
                "edge_count": subgraph.number_of_edges(),
                "avg_degree": statistics.mean(degrees.values()) if degrees else 0.0
            }
        )
    
    def _detect_ghost_loops(self) -> List[Dict[str, Any]]:
        """Detect ghost loops (circular reasoning patterns) in the graph"""
        ghost_loops = []
        
        try:
            # Find cycles in the graph
            cycles = list(nx.simple_cycles(self.graph.to_directed()))
            
            for cycle in cycles:
                if len(cycle) >= 3:  # Minimum cycle length
                    # Analyze cycle strength
                    cycle_strength = self._calculate_cycle_strength(cycle)
                    
                    if cycle_strength > 0.5:  # Strong circular pattern
                        ghost_loop = {
                            "id": generate_uuid(),
                            "type": "circular_reasoning",
                            "nodes": cycle,
                            "strength": cycle_strength,
                            "length": len(cycle),
                            "detected_at": datetime.now().isoformat(),
                            "description": f"Circular reasoning pattern involving {len(cycle)} concepts"
                        }
                        ghost_loops.append(ghost_loop)
            
            return ghost_loops[:5]  # Limit to top 5 ghost loops
            
        except Exception as e:
            logger.error(f"Error detecting ghost loops: {e}")
            return []
    
    def _calculate_cycle_strength(self, cycle: List[str]) -> float:
        """Calculate the strength of a cycle based on edge weights"""
        try:
            total_strength = 0.0
            edge_count = 0
            
            for i in range(len(cycle)):
                current_node = cycle[i]
                next_node = cycle[(i + 1) % len(cycle)]
                
                if self.graph.has_edge(current_node, next_node):
                    edge_data = self.graph.get_edge_data(current_node, next_node)
                    strength = edge_data.get('strength', 1.0)
                    total_strength += strength
                    edge_count += 1
            
            return total_strength / edge_count if edge_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating cycle strength: {e}")
            return 0.0

class ResonanceMapManager:
    """
    Main resonance map management system.
    
    Handles graph construction, analysis, and visualization data generation.
    """
    
    def __init__(self, db):
        self.db = db
        self.concept_extractor = ConceptExtractor()
        self._graph_cache = {}  # Cache for session graphs
        self._cache_timeout = 600  # 10 minutes
        
        logger.info("ResonanceMapManager initialized")
    
    async def update_from_entry(self, entry_id: str) -> bool:
        """
        Update resonance map from a journal entry.
        
        Args:
            entry_id: Journal entry ID
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get journal entry
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM journal_entries WHERE id = ?
                """, (entry_id,))
                
                entry_row = cursor.fetchone()
                if not entry_row:
                    logger.warning(f"Journal entry {entry_id} not found")
                    return False
                
                session_id = entry_row["session_id"]
            
            # Extract concepts from entry
            combined_text = f"{entry_row['title']} {entry_row['content']} {entry_row['summary']}"
            concepts = self.concept_extractor.extract_concepts(combined_text)
            
            if not concepts:
                logger.info(f"No concepts extracted from entry {entry_id}")
                return True
            
            # Create or update nodes
            node_ids = []
            for concept, importance in concepts:
                node_id = await self._create_or_update_node(
                    session_id=session_id,
                    title=concept,
                    node_type="concept",
                    description=f"Concept from journal entry: {entry_row['title']}",
                    strength=min(1.0, importance * 2.0),  # Scale importance to strength
                    source_entry=entry_id
                )
                node_ids.append(node_id)
            
            # Create insights nodes from journal insights
            if entry_row["insights"]:
                try:
                    insights = json.loads(entry_row["insights"])
                    for insight in insights:
                        if len(insight.strip()) > 10:  # Minimum insight length
                            insight_node_id = await self._create_or_update_node(
                                session_id=session_id,
                                title=insight[:100],  # Truncate title
                                node_type="insight",
                                description=insight,
                                strength=0.8,  # High strength for insights
                                source_entry=entry_id
                            )
                            node_ids.append(insight_node_id)
                except json.JSONDecodeError:
                    pass
            
            # Find and create connections between concepts
            concept_names = [concept for concept, _ in concepts]
            relationships = self.concept_extractor.find_concept_relationships(
                concept_names, combined_text
            )
            
            for concept1, concept2, strength in relationships:
                await self._create_or_update_connection(
                    session_id=session_id,
                    source_concept=concept1,
                    target_concept=concept2,
                    connection_type="relates_to",
                    strength=strength,
                    created_from=f"journal_entry:{entry_id}"
                )
            
            # Clear graph cache for this session
            self._clear_graph_cache(session_id)
            
            logger.info(f"Updated resonance map from entry {entry_id}: {len(node_ids)} nodes, {len(relationships)} connections")
            return True
            
        except Exception as e:
            logger.error(f"Error updating resonance map from entry {entry_id}: {e}")
            return False
    
    async def update_from_debate(self, debate_id: str) -> bool:
        """
        Update resonance map from a debate session.
        
        Args:
            debate_id: Debate ID
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Get debate and agent responses
            with self.db.get_connection() as conn:
                # Get debate info
                cursor = conn.execute("""
                    SELECT * FROM debates WHERE id = ?
                """, (debate_id,))
                
                debate_row = cursor.fetchone()
                if not debate_row:
                    return False
                
                session_id = debate_row["session_id"]
                
                # Get agent responses
                cursor = conn.execute("""
                    SELECT * FROM agent_responses 
                    WHERE debate_id = ?
                    ORDER BY round_number, created_at
                """, (debate_id,))
                
                responses = cursor.fetchall()
                
                # Get synthesis if available
                cursor = conn.execute("""
                    SELECT * FROM syntheses WHERE debate_id = ?
                """, (debate_id,))
                
                synthesis_row = cursor.fetchone()
            
            # Extract concepts from all debate content
            all_content = []
            debate_positions = {"proponent": [], "opponent": []}
            
            for response in responses:
                content = response["content"]
                agent_role = response["agent_role"]
                
                all_content.append(content)
                
                if agent_role in debate_positions:
                    debate_positions[agent_role].append(content)
            
            # Add synthesis content
            if synthesis_row:
                all_content.append(synthesis_row["content"])
            
            # Extract concepts from combined content
            combined_text = " ".join(all_content)
            concepts = self.concept_extractor.extract_concepts(combined_text, max_concepts=30)
            
            # Create debate question node
            question_node_id = await self._create_or_update_node(
                session_id=session_id,
                title=debate_row.get("clarified_question", "Debate Question"),
                node_type="question",
                description=f"Question from debate {debate_id}",
                strength=1.0,
                source_entry=debate_id
            )
            
            # Create concept nodes
            concept_node_ids = []
            for concept, importance in concepts:
                node_id = await self._create_or_update_node(
                    session_id=session_id,
                    title=concept,
                    node_type="debate_point",
                    description=f"Concept from debate: {debate_row.get('clarified_question', '')}",
                    strength=min(1.0, importance * 1.5),
                    source_entry=debate_id
                )
                concept_node_ids.append(node_id)
                
                # Connect concepts to question
                await self._create_or_update_connection(
                    session_id=session_id,
                    source_node_id=question_node_id,
                    target_node_id=node_id,
                    connection_type="explores",
                    strength=0.7,
                    created_from=f"debate:{debate_id}"
                )
            
            # Analyze opposing positions
            if debate_positions["proponent"] and debate_positions["opponent"]:
                await self._analyze_debate_opposition(
                    session_id=session_id,
                    proponent_content=" ".join(debate_positions["proponent"]),
                    opponent_content=" ".join(debate_positions["opponent"]),
                    debate_id=debate_id
                )
            
            # Create synthesis node if available
            if synthesis_row:
                synthesis_node_id = await self._create_or_update_node(
                    session_id=session_id,
                    title="Synthesis: " + synthesis_row["content"][:50] + "...",
                    node_type="synthesis",
                    description=synthesis_row["content"],
                    strength=0.9,
                    source_entry=debate_id
                )
                
                # Connect synthesis to question
                await self._create_or_update_connection(
                    session_id=session_id,
                    source_node_id=question_node_id,
                    target_node_id=synthesis_node_id,
                    connection_type="synthesizes",
                    strength=0.8,
                    created_from=f"debate:{debate_id}"
                )
            
            # Clear graph cache
            self._clear_graph_cache(session_id)
            
            logger.info(f"Updated resonance map from debate {debate_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating resonance map from debate {debate_id}: {e}")
            return False
    
    async def _analyze_debate_opposition(self, session_id: str, proponent_content: str,
                                        opponent_content: str, debate_id: str):
        """Analyze opposing positions in a debate and create contradiction connections"""
        try:
            # Extract concepts from each position
            pro_concepts = self.concept_extractor.extract_concepts(proponent_content, max_concepts=15)
            opp_concepts = self.concept_extractor.extract_concepts(opponent_content, max_concepts=15)
            
            # Find opposing concept pairs based on semantic similarity and context
            for pro_concept, pro_importance in pro_concepts:
                for opp_concept, opp_importance in opp_concepts:
                    # Check if concepts are semantically related but contextually opposed
                    semantic_similarity = calculate_similarity(pro_concept, opp_concept)
                    
                    if 0.3 < semantic_similarity < 0.8:  # Related but not identical
                        # Create contradiction connection
                        await self._create_or_update_connection(
                            session_id=session_id,
                            source_concept=pro_concept,
                            target_concept=opp_concept,
                            connection_type="contradicts",
                            strength=min(1.0, (pro_importance + opp_importance) / 2),
                            created_from=f"debate:{debate_id}"
                        )
                        
        except Exception as e:
            logger.error(f"Error analyzing debate opposition: {e}")
    
    async def _create_or_update_node(self, session_id: str, title: str, node_type: str,
                                    description: str, strength: float,
                                    source_entry: str = None) -> str:
        """Create new node or update existing one"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.sha256(f"{title.lower()}:{node_type}".encode()).hexdigest()
            
            with self.db.get_connection() as conn:
                # Check if node already exists
                cursor = conn.execute("""
                    SELECT id, strength, frequency FROM resonance_nodes 
                    WHERE session_id = ? AND content_hash = ?
                """, (session_id, content_hash))
                
                existing_node = cursor.fetchone()
                
                if existing_node:
                    # Update existing node
                    node_id = existing_node["id"]
                    new_strength = min(1.0, existing_node["strength"] + (strength * 0.1))
                    new_frequency = existing_node["frequency"] + 1
                    
                    conn.execute("""
                        UPDATE resonance_nodes 
                        SET strength = ?, frequency = ?, last_activated = ?
                        WHERE id = ?
                    """, (new_strength, new_frequency, datetime.now(), node_id))
                    
                else:
                    # Create new node
                    node_id = generate_uuid()
                    
                    conn.execute("""
                        INSERT INTO resonance_nodes (
                            id, session_id, node_type, title, description,
                            content_hash, strength, frequency, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node_id, session_id, node_type, title, description,
                        content_hash, strength, 1,
                        json.dumps({"source_entry": source_entry} if source_entry else {})
                    ))
                
                conn.commit()
                return node_id
                
        except Exception as e:
            logger.error(f"Error creating/updating node: {e}")
            return generate_uuid()  # Return dummy ID on error
    
    async def _create_or_update_connection(self, session_id: str, 
                                          source_concept: str = None, target_concept: str = None,
                                          source_node_id: str = None, target_node_id: str = None,
                                          connection_type: str = "relates_to",
                                          strength: float = 1.0, created_from: str = ""):
        """Create or update connection between nodes"""
        try:
            # Get node IDs if not provided
            if source_node_id is None and source_concept:
                source_node_id = await self._find_node_by_concept(session_id, source_concept)
            
            if target_node_id is None and target_concept:
                target_node_id = await self._find_node_by_concept(session_id, target_concept)
            
            if not source_node_id or not target_node_id:
                logger.warning(f"Could not find nodes for connection: {source_concept} -> {target_concept}")
                return
            
            with self.db.get_connection() as conn:
                # Check if connection already exists
                cursor = conn.execute("""
                    SELECT id, strength FROM resonance_connections 
                    WHERE session_id = ? AND source_node_id = ? AND target_node_id = ? 
                    AND connection_type = ?
                """, (session_id, source_node_id, target_node_id, connection_type))
                
                existing_conn = cursor.fetchone()
                
                if existing_conn:
                    # Update existing connection
                    new_strength = min(1.0, existing_conn["strength"] + (strength * 0.1))
                    
                    conn.execute("""
                        UPDATE resonance_connections 
                        SET strength = ?, last_strengthened = ?
                        WHERE id = ?
                    """, (new_strength, datetime.now(), existing_conn["id"]))
                    
                else:
                    # Create new connection
                    connection_id = generate_uuid()
                    
                    conn.execute("""
                        INSERT INTO resonance_connections (
                            id, session_id, source_node_id, target_node_id,
                            connection_type, strength, created_from
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        connection_id, session_id, source_node_id, target_node_id,
                        connection_type, strength, created_from
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error creating/updating connection: {e}")
    
    async def _find_node_by_concept(self, session_id: str, concept: str) -> Optional[str]:
        """Find node ID by concept title"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id FROM resonance_nodes 
                    WHERE session_id = ? AND LOWER(title) = LOWER(?)
                    ORDER BY strength DESC, frequency DESC
                    LIMIT 1
                """, (session_id, concept.strip()))
                
                result = cursor.fetchone()
                return result["id"] if result else None
                
        except Exception as e:
            logger.error(f"Error finding node by concept: {e}")
            return None
    
    async def get_map_data(self, session_id: str, filter_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get visualization data for the resonance map.
        
        Args:
            session_id: User session ID
            filter_params: Filtering parameters
            
        Returns:
            Visualization data for frontend
        """
        try:
            # Check cache first
            cache_key = f"{session_id}:{hash(str(filter_params))}"
            if cache_key in self._graph_cache:
                cache_entry = self._graph_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self._cache_timeout:
                    return cache_entry["data"]
            
            # Build graph from database
            graph = await self._build_graph(session_id, filter_params)
            
            if graph.number_of_nodes() == 0:
                return {
                    "nodes": [],
                    "edges": [],
                    "metrics": {},
                    "clusters": [],
                    "has_data": False
                }
            
            # Analyze graph
            analyzer = GraphAnalyzer(graph)
            metrics = analyzer.calculate_metrics()
            
            # Generate visualization data
            vis_data = await self._generate_visualization_data(graph, metrics, filter_params)
            
            # Cache result
            self._graph_cache[cache_key] = {
                "data": vis_data,
                "timestamp": time.time()
            }
            
            return vis_data
            
        except Exception as e:
            logger.error(f"Error getting map data: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    async def _build_graph(self, session_id: str, filter_params: Dict[str, Any] = None) -> nx.Graph:
        """Build NetworkX graph from database"""
        graph = nx.Graph()
        
        try:
            filter_params = filter_params or {}
            
            with self.db.get_connection() as conn:
                # Build WHERE clause for nodes
                node_conditions = ["session_id = ?"]
                node_params = [session_id]
                
                # Filter by node type
                if filter_params.get("node_types"):
                    placeholders = ",".join("?" * len(filter_params["node_types"]))
                    node_conditions.append(f"node_type IN ({placeholders})")
                    node_params.extend(filter_params["node_types"])
                
                # Filter by minimum strength
                min_strength = filter_params.get("min_strength", 0.1)
                node_conditions.append("strength >= ?")
                node_params.append(min_strength)
                
                # Filter by time range
                if filter_params.get("date_from"):
                    node_conditions.append("created_at >= ?")
                    node_params.append(filter_params["date_from"])
                
                if filter_params.get("date_to"):
                    node_conditions.append("created_at <= ?")
                    node_params.append(filter_params["date_to"])
                
                # Get nodes
                cursor = conn.execute(f"""
                    SELECT * FROM resonance_nodes 
                    WHERE {' AND '.join(node_conditions)}
                    ORDER BY strength DESC
                    LIMIT ?
                """, node_params + [filter_params.get("max_nodes", 100)])
                
                nodes = cursor.fetchall()
                
                # Add nodes to graph
                for node in nodes:
                    graph.add_node(node["id"], **{
                        "title": node["title"],
                        "type": node["node_type"],
                        "strength": node["strength"],
                        "frequency": node["frequency"],
                        "description": node["description"],
                        "created_at": node["created_at"]
                    })
                
                # Get connections for these nodes
                if nodes:
                    node_ids = [node["id"] for node in nodes]
                    placeholders = ",".join("?" * len(node_ids))
                    
                    cursor = conn.execute(f"""
                        SELECT * FROM resonance_connections 
                        WHERE session_id = ? 
                        AND source_node_id IN ({placeholders})
                        AND target_node_id IN ({placeholders})
                        AND strength >= ?
                    """, [session_id] + node_ids + node_ids + [filter_params.get("min_connection_strength", 0.1)])
                    
                    connections = cursor.fetchall()
                    
                    # Add edges to graph
                    for conn in connections:
                        if graph.has_node(conn["source_node_id"]) and graph.has_node(conn["target_node_id"]):
                            graph.add_edge(
                                conn["source_node_id"],
                                conn["target_node_id"],
                                **{
                                    "type": conn["connection_type"],
                                    "strength": conn["strength"],
                                    "created_from": conn["created_from"],
                                    "created_at": conn["created_at"]
                                }
                            )
            
            return graph
            
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return nx.Graph()
    
    async def _generate_visualization_data(self, graph: nx.Graph, metrics: GraphMetrics,
                                          filter_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate data structure for frontend visualization"""
        try:
            # Calculate layout positions using spring layout
            if graph.number_of_nodes() > 0:
                pos = nx.spring_layout(
                    graph,
                    k=1.0,
                    iterations=50,
                    weight='strength'
                )
            else:
                pos = {}
            
            # Prepare nodes for visualization
            nodes = []
            for node_id, node_data in graph.nodes(data=True):
                position = pos.get(node_id, (0, 0))
                
                # Calculate node size based on centrality and strength
                centrality_score = 0.0
                for central_node, score in metrics.central_nodes:
                    if central_node == node_id:
                        centrality_score = score
                        break
                
                size = max(10, min(50, 20 + (node_data.get("strength", 0) * 20) + (centrality_score * 30)))
                
                nodes.append({
                    "id": node_id,
                    "title": node_data.get("title", ""),
                    "type": node_data.get("type", "concept"),
                    "strength": node_data.get("strength", 0),
                    "frequency": node_data.get("frequency", 1),
                    "description": node_data.get("description", ""),
                    "x": position[0] * 300,  # Scale for visualization
                    "y": position[1] * 300,
                    "size": size,
                    "centrality": centrality_score
                })
            
            # Prepare edges for visualization
            edges = []
            for source, target, edge_data in graph.edges(data=True):
                strength = edge_data.get("strength", 1.0)
                
                edges.append({
                    "source": source,
                    "target": target,
                    "type": edge_data.get("type", "relates_to"),
                    "strength": strength,
                    "width": max(1, strength * 5),  # Visual width
                    "created_from": edge_data.get("created_from", ""),
                    "opacity": max(0.3, strength)
                })
            
            # Prepare clusters for visualization
            clusters = []
            for cluster in metrics.communities:
                # Calculate cluster center
                cluster_nodes = [pos.get(node_id, (0, 0)) for node_id in cluster.nodes if node_id in pos]
                if cluster_nodes:
                    center_x = statistics.mean([p[0] for p in cluster_nodes]) * 300
                    center_y = statistics.mean([p[1] for p in cluster_nodes]) * 300
                    
                    clusters.append({
                        "id": cluster.id,
                        "nodes": cluster.nodes,
                        "central_concept": cluster.central_concept,
                        "strength": cluster.strength,
                        "density": cluster.density,
                        "center_x": center_x,
                        "center_y": center_y,
                        "node_count": len(cluster.nodes)
                    })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "clusters": clusters,
                "metrics": {
                    "total_nodes": metrics.total_nodes,
                    "total_edges": metrics.total_edges,
                    "density": metrics.density,
                    "clustering_coefficient": metrics.clustering_coefficient,
                    "connected_components": metrics.connected_components,
                    "central_nodes": [{"id": node_id, "score": score} for node_id, score in metrics.central_nodes[:5]]
                },
                "ghost_loops": metrics.ghost_loops,
                "has_data": True,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return {"nodes": [], "edges": [], "error": str(e)}
    
    async def prune_weak_connections(self, session_id: str, min_strength: float = 0.1) -> int:
        """Remove weak connections to optimize graph performance"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM resonance_connections 
                    WHERE session_id = ? AND strength < ?
                """, (session_id, min_strength))
                
                removed_count = cursor.rowcount
                conn.commit()
                
                # Clear cache
                self._clear_graph_cache(session_id)
                
                logger.info(f"Pruned {removed_count} weak connections for session {session_id}")
                return removed_count
                
        except Exception as e:
            logger.error(f"Error pruning connections: {e}")
            return 0
    
    async def decay_node_strengths(self, session_id: str, decay_rate: float = 0.01) -> int:
        """Apply time-based decay to node strengths"""
        try:
            # Calculate decay based on time since last activation
            cutoff_date = datetime.now() - timedelta(days=30)
            
            with self.db.get_connection() as conn:
                cursor = conn.execute("""
                    UPDATE resonance_nodes 
                    SET strength = strength * (1 - ?)
                    WHERE session_id = ? AND last_activated < ?
                    AND strength > 0.1
                """, (decay_rate, session_id, cutoff_date))
                
                updated_count = cursor.rowcount
                conn.commit()
                
                # Clear cache
                self._clear_graph_cache(session_id)
                
                logger.info(f"Applied decay to {updated_count} nodes for session {session_id}")
                return updated_count
                
        except Exception as e:
            logger.error(f"Error applying decay: {e}")
            return 0
    
    def _clear_graph_cache(self, session_id: str):
        """Clear cached graphs for a session"""
        keys_to_remove = [key for key in self._graph_cache.keys() if key.startswith(session_id)]
        for key in keys_to_remove:
            del self._graph_cache[key]
