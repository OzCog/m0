"""
Tensor Fragment Architecture

Implements tensor fragment data structures and bidirectional translation
between ML primitives and AtomSpace hypergraph patterns.

This module provides the core tensor fragment management and optimization
capabilities for the cognitive architecture.
"""

from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
import json
from abc import ABC, abstractmethod

from .primitives import CognitivePrimitive, TensorSignature, ModalityType


@dataclass
class TensorFragment:
    """
    Core tensor fragment representing a localized cognitive pattern.
    
    Tensor fragments are composable units that maintain both tensor representations
    for ML operations and hypergraph structures for symbolic reasoning.
    """
    id: str
    signature: TensorSignature
    tensor_data: np.ndarray
    hypergraph_structure: Dict[str, Any]
    primitives: List[CognitivePrimitive] = field(default_factory=list)
    connections: Dict[str, float] = field(default_factory=dict)  # id -> strength
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate tensor fragment consistency"""
        if self.tensor_data.size == 0:
            raise ValueError("Tensor data cannot be empty")
        
        # Ensure hypergraph structure has required fields
        required_fields = {"nodes", "edges", "patterns"}
        if not all(field in self.hypergraph_structure for field in required_fields):
            missing = required_fields - set(self.hypergraph_structure.keys())
            raise ValueError(f"Hypergraph structure missing required fields: {missing}")
    
    def get_tensor_features(self) -> np.ndarray:
        """Extract ML-compatible tensor features"""
        # Combine signature vector with tensor data
        signature_vector = self.signature.to_vector()
        
        # Flatten tensor data and combine with signature
        flat_tensor = self.tensor_data.flatten()
        
        # Add connection strengths as features
        connection_features = np.array(list(self.connections.values())) if self.connections else np.array([])
        
        # Combine all features
        features = [signature_vector, flat_tensor]
        if len(connection_features) > 0:
            features.append(connection_features)
        
        return np.concatenate(features)
    
    def to_atomspace_hypergraph(self) -> Dict[str, Any]:
        """Convert tensor fragment to AtomSpace hypergraph representation"""
        # Create hypergraph with nodes, edges, and patterns
        atomspace_graph = {
            "fragment_id": self.id,
            "signature": {
                "modality": self.signature.modality.value,
                "depth": self.signature.depth,
                "context": self.signature.context,
                "salience": self.signature.salience,
                "autonomy_index": self.signature.autonomy_index
            },
            "nodes": [],
            "edges": [],
            "patterns": [],
            "tensor_encoding": self.tensor_data.tolist(),
            "metadata": self.metadata
        }
        
        # Add primitive nodes
        for primitive in self.primitives:
            node = {
                "id": primitive.id,
                "type": "ConceptNode",
                "name": primitive.id,
                "tv": {
                    "strength": primitive.signature.salience,
                    "confidence": primitive.signature.context
                },
                "properties": primitive.to_atomspace_pattern()
            }
            atomspace_graph["nodes"].append(node)
        
        # Add connection edges
        for target_id, strength in self.connections.items():
            edge = {
                "type": "SimilarityLink",
                "source": self.id,
                "target": target_id,
                "tv": {
                    "strength": strength,
                    "confidence": 0.8
                }
            }
            atomspace_graph["edges"].append(edge)
        
        # Add hypergraph patterns
        for pattern_name, pattern_data in self.hypergraph_structure.get("patterns", {}).items():
            pattern = {
                "name": pattern_name,
                "type": "PatternNode",
                "structure": pattern_data,
                "fragment_id": self.id
            }
            atomspace_graph["patterns"].append(pattern)
        
        return atomspace_graph
    
    @classmethod
    def from_atomspace_hypergraph(cls, hypergraph: Dict[str, Any]) -> 'TensorFragment':
        """Create tensor fragment from AtomSpace hypergraph representation"""
        fragment_id = hypergraph.get("fragment_id", "")
        
        # Reconstruct signature
        sig_data = hypergraph.get("signature", {})
        signature = TensorSignature(
            modality=ModalityType(sig_data.get("modality", "symbolic")),
            depth=sig_data.get("depth", 0),
            context=sig_data.get("context", 0.5),
            salience=sig_data.get("salience", 0.5),
            autonomy_index=sig_data.get("autonomy_index", 0.5)
        )
        
        # Reconstruct tensor data
        tensor_encoding = hypergraph.get("tensor_encoding", [])
        tensor_data = np.array(tensor_encoding) if tensor_encoding else np.array([0.0])
        
        # Reconstruct primitives
        primitives = []
        for node in hypergraph.get("nodes", []):
            if node.get("type") == "ConceptNode":
                primitive = CognitivePrimitive.from_atomspace_pattern(node.get("properties", {}))
                primitives.append(primitive)
        
        # Reconstruct connections
        connections = {}
        for edge in hypergraph.get("edges", []):
            if edge.get("type") == "SimilarityLink" and edge.get("source") == fragment_id:
                target_id = edge.get("target")
                strength = edge.get("tv", {}).get("strength", 0.5)
                connections[target_id] = strength
        
        # Reconstruct hypergraph structure
        hypergraph_structure = {
            "nodes": hypergraph.get("nodes", []),
            "edges": hypergraph.get("edges", []),
            "patterns": {p["name"]: p["structure"] for p in hypergraph.get("patterns", [])}
        }
        
        return cls(
            id=fragment_id,
            signature=signature,
            tensor_data=tensor_data,
            hypergraph_structure=hypergraph_structure,
            primitives=primitives,
            connections=connections,
            metadata=hypergraph.get("metadata", {})
        )
    
    def merge_with(self, other: 'TensorFragment', merge_strategy: str = "weighted_average") -> 'TensorFragment':
        """Merge this tensor fragment with another"""
        if merge_strategy == "weighted_average":
            return self._weighted_average_merge(other)
        elif merge_strategy == "concatenate":
            return self._concatenate_merge(other)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
    
    def _weighted_average_merge(self, other: 'TensorFragment') -> 'TensorFragment':
        """Merge using weighted average of tensor data and signatures"""
        # Calculate weights based on salience
        w1 = self.signature.salience
        w2 = other.signature.salience
        total_weight = w1 + w2
        
        if total_weight == 0:
            w1 = w2 = 0.5
        else:
            w1 /= total_weight
            w2 /= total_weight
        
        # Merge signatures
        merged_signature = TensorSignature(
            modality=self.signature.modality if w1 >= w2 else other.signature.modality,
            depth=max(self.signature.depth, other.signature.depth),
            context=w1 * self.signature.context + w2 * other.signature.context,
            salience=w1 * self.signature.salience + w2 * other.signature.salience,
            autonomy_index=w1 * self.signature.autonomy_index + w2 * other.signature.autonomy_index
        )
        
        # Merge tensor data
        if self.tensor_data.shape == other.tensor_data.shape:
            merged_tensor = w1 * self.tensor_data + w2 * other.tensor_data
        else:
            # Pad to same shape and then merge
            max_shape = tuple(max(s1, s2) for s1, s2 in zip(self.tensor_data.shape, other.tensor_data.shape))
            
            padded_tensor1 = np.zeros(max_shape)
            padded_tensor2 = np.zeros(max_shape)
            
            padded_tensor1[:self.tensor_data.shape[0]] = self.tensor_data
            padded_tensor2[:other.tensor_data.shape[0]] = other.tensor_data
            
            merged_tensor = w1 * padded_tensor1 + w2 * padded_tensor2
        
        # Merge primitives and connections
        merged_primitives = self.primitives + other.primitives
        merged_connections = {**self.connections, **other.connections}
        
        # Merge hypergraph structures
        merged_hypergraph = {
            "nodes": self.hypergraph_structure.get("nodes", []) + other.hypergraph_structure.get("nodes", []),
            "edges": self.hypergraph_structure.get("edges", []) + other.hypergraph_structure.get("edges", []),
            "patterns": {**self.hypergraph_structure.get("patterns", {}), **other.hypergraph_structure.get("patterns", {})}
        }
        
        return TensorFragment(
            id=f"merged_{self.id}_{other.id}",
            signature=merged_signature,
            tensor_data=merged_tensor,
            hypergraph_structure=merged_hypergraph,
            primitives=merged_primitives,
            connections=merged_connections,
            metadata={"merge_strategy": "weighted_average", "source_fragments": [self.id, other.id]}
        )
    
    def _concatenate_merge(self, other: 'TensorFragment') -> 'TensorFragment':
        """Merge by concatenating tensor data"""
        # Use signature of more salient fragment
        primary_signature = self.signature if self.signature.salience >= other.signature.salience else other.signature
        
        # Concatenate tensor data
        merged_tensor = np.concatenate([self.tensor_data.flatten(), other.tensor_data.flatten()])
        
        # Merge other components
        merged_primitives = self.primitives + other.primitives
        merged_connections = {**self.connections, **other.connections}
        
        merged_hypergraph = {
            "nodes": self.hypergraph_structure.get("nodes", []) + other.hypergraph_structure.get("nodes", []),
            "edges": self.hypergraph_structure.get("edges", []) + other.hypergraph_structure.get("edges", []),
            "patterns": {**self.hypergraph_structure.get("patterns", {}), **other.hypergraph_structure.get("patterns", {})}
        }
        
        return TensorFragment(
            id=f"concat_{self.id}_{other.id}",
            signature=primary_signature,
            tensor_data=merged_tensor,
            hypergraph_structure=merged_hypergraph,
            primitives=merged_primitives,
            connections=merged_connections,
            metadata={"merge_strategy": "concatenate", "source_fragments": [self.id, other.id]}
        )


class HypergraphEncoder:
    """
    Bidirectional encoder between ML primitives and AtomSpace hypergraph patterns.
    
    Provides translation capabilities between tensor representations and
    symbolic hypergraph structures for cognitive processing.
    """
    
    def __init__(self):
        self.primitive_registry: Dict[str, CognitivePrimitive] = {}
        self.fragment_registry: Dict[str, TensorFragment] = {}
        self.pattern_templates: Dict[str, Dict[str, Any]] = self._initialize_pattern_templates()
    
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard hypergraph pattern templates"""
        return {
            "simple_concept": {
                "type": "ConceptNode",
                "structure": {
                    "node_type": "concept",
                    "properties": ["name", "truth_value"],
                    "connections": []
                }
            },
            "inheritance_link": {
                "type": "InheritanceLink",
                "structure": {
                    "link_type": "inheritance",
                    "source": "child_concept",
                    "target": "parent_concept",
                    "strength": 0.8
                }
            },
            "similarity_link": {
                "type": "SimilarityLink",
                "structure": {
                    "link_type": "similarity",
                    "source": "concept_a",
                    "target": "concept_b",
                    "strength": 0.7
                }
            },
            "composition_pattern": {
                "type": "PatternNode",
                "structure": {
                    "pattern_type": "composition",
                    "components": [],
                    "composition_rule": "weighted_merge"
                }
            }
        }
    
    def encode_primitive_to_hypergraph(self, primitive: CognitivePrimitive) -> Dict[str, Any]:
        """Encode cognitive primitive to hypergraph pattern"""
        self.primitive_registry[primitive.id] = primitive
        
        # Determine appropriate pattern template
        pattern_type = self._determine_pattern_type(primitive)
        template = self.pattern_templates.get(pattern_type, self.pattern_templates["simple_concept"])
        
        # Create hypergraph pattern
        hypergraph_pattern = {
            "id": primitive.id,
            "type": template["type"],
            "primitive_signature": {
                "modality": primitive.signature.modality.value,
                "depth": primitive.signature.depth,
                "context": primitive.signature.context,
                "salience": primitive.signature.salience,
                "autonomy_index": primitive.signature.autonomy_index
            },
            "structure": self._customize_pattern_structure(template["structure"], primitive),
            "content": primitive.content,
            "metadata": primitive.metadata,
            "tensor_encoding": primitive.get_ml_features().tolist()
        }
        
        return hypergraph_pattern
    
    def decode_hypergraph_to_primitive(self, hypergraph_pattern: Dict[str, Any]) -> CognitivePrimitive:
        """Decode hypergraph pattern to cognitive primitive"""
        # Reconstruct signature
        sig_data = hypergraph_pattern.get("primitive_signature", {})
        signature = TensorSignature(
            modality=ModalityType(sig_data.get("modality", "symbolic")),
            depth=sig_data.get("depth", 0),
            context=sig_data.get("context", 0.5),
            salience=sig_data.get("salience", 0.5),
            autonomy_index=sig_data.get("autonomy_index", 0.5)
        )
        
        # Create primitive
        primitive = CognitivePrimitive(
            id=hypergraph_pattern.get("id", ""),
            signature=signature,
            content=hypergraph_pattern.get("content"),
            metadata=hypergraph_pattern.get("metadata", {}),
            relationships=hypergraph_pattern.get("structure", {}).get("connections", [])
        )
        
        self.primitive_registry[primitive.id] = primitive
        return primitive
    
    def encode_fragment_to_hypergraph(self, fragment: TensorFragment) -> Dict[str, Any]:
        """Encode tensor fragment to complete hypergraph representation"""
        self.fragment_registry[fragment.id] = fragment
        return fragment.to_atomspace_hypergraph()
    
    def decode_hypergraph_to_fragment(self, hypergraph: Dict[str, Any]) -> TensorFragment:
        """Decode hypergraph representation to tensor fragment"""
        fragment = TensorFragment.from_atomspace_hypergraph(hypergraph)
        self.fragment_registry[fragment.id] = fragment
        return fragment
    
    def create_fragment_from_primitives(self, primitives: List[CognitivePrimitive], 
                                      fragment_id: Optional[str] = None) -> TensorFragment:
        """Create tensor fragment from a collection of cognitive primitives"""
        if not primitives:
            raise ValueError("Cannot create fragment from empty primitive list")
        
        if fragment_id is None:
            fragment_id = f"fragment_{hash('_'.join(p.id for p in primitives)) % 10000}"
        
        # Calculate fragment signature from primitives
        avg_context = sum(p.signature.context for p in primitives) / len(primitives)
        avg_salience = sum(p.signature.salience for p in primitives) / len(primitives)
        avg_autonomy = sum(p.signature.autonomy_index for p in primitives) / len(primitives)
        max_depth = max(p.signature.depth for p in primitives)
        
        # Use modality of most salient primitive
        primary_primitive = max(primitives, key=lambda p: p.signature.salience)
        
        fragment_signature = TensorSignature(
            modality=primary_primitive.signature.modality,
            depth=max_depth,
            context=avg_context,
            salience=avg_salience,
            autonomy_index=avg_autonomy
        )
        
        # Create tensor data from primitive features
        feature_vectors = [p.get_ml_features() for p in primitives]
        max_length = max(len(fv) for fv in feature_vectors)
        
        # Pad all vectors to same length
        padded_vectors = []
        for fv in feature_vectors:
            if len(fv) < max_length:
                padded = np.zeros(max_length)
                padded[:len(fv)] = fv
                padded_vectors.append(padded)
            else:
                padded_vectors.append(fv)
        
        tensor_data = np.array(padded_vectors)
        
        # Create hypergraph structure
        hypergraph_structure = {
            "nodes": [p.to_atomspace_pattern() for p in primitives],
            "edges": self._compute_primitive_connections(primitives),
            "patterns": {
                "fragment_pattern": {
                    "type": "composition",
                    "components": [p.id for p in primitives],
                    "fragment_id": fragment_id
                }
            }
        }
        
        # Calculate connections between primitives
        connections = {}
        for i, p1 in enumerate(primitives):
            for j, p2 in enumerate(primitives):
                if i != j:
                    compatibility = p1.calculate_compatibility(p2)
                    if compatibility > 0.5:  # Only store significant connections
                        connections[p2.id] = compatibility
        
        return TensorFragment(
            id=fragment_id,
            signature=fragment_signature,
            tensor_data=tensor_data,
            hypergraph_structure=hypergraph_structure,
            primitives=primitives,
            connections=connections,
            metadata={
                "creation_method": "from_primitives",
                "primitive_count": len(primitives)
            }
        )
    
    def _determine_pattern_type(self, primitive: CognitivePrimitive) -> str:
        """Determine appropriate hypergraph pattern type for primitive"""
        if primitive.signature.modality == ModalityType.RELATIONAL:
            if len(primitive.relationships) == 2:
                return "similarity_link"
            else:
                return "composition_pattern"
        else:
            return "simple_concept"
    
    def _customize_pattern_structure(self, template_structure: Dict[str, Any], 
                                   primitive: CognitivePrimitive) -> Dict[str, Any]:
        """Customize pattern structure template for specific primitive"""
        structure = template_structure.copy()
        
        # Add primitive-specific properties
        structure["primitive_id"] = primitive.id
        structure["modality"] = primitive.signature.modality.value
        structure["depth"] = primitive.signature.depth
        structure["connections"] = primitive.relationships
        
        # Add content-specific customizations
        if isinstance(primitive.content, dict) and "source" in primitive.content:
            # Relational primitive
            structure["source"] = primitive.content["source"]
            structure["target"] = primitive.content["target"]
            structure["relation_type"] = primitive.content.get("relation", "unknown")
        
        return structure
    
    def _compute_primitive_connections(self, primitives: List[CognitivePrimitive]) -> List[Dict[str, Any]]:
        """Compute connection edges between primitives"""
        edges = []
        
        for i, p1 in enumerate(primitives):
            for j, p2 in enumerate(primitives):
                if i < j:  # Avoid duplicate edges
                    compatibility = p1.calculate_compatibility(p2)
                    if compatibility > 0.3:  # Threshold for creating edge
                        edge = {
                            "type": "SimilarityLink",
                            "source": p1.id,
                            "target": p2.id,
                            "tv": {
                                "strength": compatibility,
                                "confidence": 0.8
                            },
                            "properties": {
                                "compatibility_score": compatibility,
                                "edge_type": "primitive_similarity"
                            }
                        }
                        edges.append(edge)
        
        return edges
    
    def optimize_fragment_representation(self, fragment: TensorFragment) -> TensorFragment:
        """Optimize tensor fragment representation for efficiency"""
        # Reduce tensor dimensionality if possible
        optimized_tensor = self._compress_tensor_data(fragment.tensor_data)
        
        # Simplify hypergraph structure
        optimized_hypergraph = self._simplify_hypergraph(fragment.hypergraph_structure)
        
        # Update fragment signature if needed
        optimized_signature = self._optimize_signature(fragment.signature, optimized_tensor)
        
        return TensorFragment(
            id=f"{fragment.id}_optimized",
            signature=optimized_signature,
            tensor_data=optimized_tensor,
            hypergraph_structure=optimized_hypergraph,
            primitives=fragment.primitives,
            connections=fragment.connections,
            metadata={**fragment.metadata, "optimization": "applied"}
        )
    
    def _compress_tensor_data(self, tensor_data: np.ndarray) -> np.ndarray:
        """Compress tensor data using dimensionality reduction"""
        # Simple compression: remove near-zero values
        threshold = 1e-6
        compressed = tensor_data.copy()
        compressed[np.abs(compressed) < threshold] = 0.0
        
        # Additional compression could use PCA or other techniques
        return compressed
    
    def _simplify_hypergraph(self, hypergraph_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify hypergraph structure by removing redundant patterns"""
        simplified = hypergraph_structure.copy()
        
        # Remove duplicate edges
        edges = simplified.get("edges", [])
        unique_edges = []
        seen_edges = set()
        
        for edge in edges:
            edge_key = (edge.get("source"), edge.get("target"), edge.get("type"))
            if edge_key not in seen_edges:
                unique_edges.append(edge)
                seen_edges.add(edge_key)
        
        simplified["edges"] = unique_edges
        
        return simplified
    
    def _optimize_signature(self, signature: TensorSignature, tensor_data: np.ndarray) -> TensorSignature:
        """Optimize tensor signature based on tensor data characteristics"""
        # Adjust salience based on tensor magnitude
        tensor_magnitude = np.linalg.norm(tensor_data)
        adjusted_salience = min(1.0, signature.salience * (1.0 + tensor_magnitude / 10.0))
        
        return TensorSignature(
            modality=signature.modality,
            depth=signature.depth,
            context=signature.context,
            salience=adjusted_salience,
            autonomy_index=signature.autonomy_index
        )