"""
Cognitive Primitives Module

Implements the foundational cognitive primitives with tensor signature:
[modality, depth, context, salience, autonomy_index]

These primitives serve as the atomic building blocks for agentic kernel ML operations
and bidirectional translation to AtomSpace hypergraph patterns.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ModalityType(Enum):
    """Defines the different modalities for cognitive processing"""
    SYMBOLIC = "symbolic"
    NUMERICAL = "numerical"
    TEXTUAL = "textual"
    VISUAL = "visual"
    AUDITORY = "auditory"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    RELATIONAL = "relational"


@dataclass
class TensorSignature:
    """
    Core tensor signature: [modality, depth, context, salience, autonomy_index]
    
    This represents the foundational encoding for cognitive primitives that enables
    bidirectional translation between ML primitives and AtomSpace hypergraph patterns.
    """
    modality: ModalityType
    depth: int
    context: float
    salience: float
    autonomy_index: float
    
    def __post_init__(self):
        """Validate tensor signature constraints"""
        if self.depth < 0:
            raise ValueError("Depth must be non-negative")
        if not 0.0 <= self.context <= 1.0:
            raise ValueError("Context must be between 0.0 and 1.0")
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError("Salience must be between 0.0 and 1.0")
        if not 0.0 <= self.autonomy_index <= 1.0:
            raise ValueError("Autonomy index must be between 0.0 and 1.0")
    
    def to_vector(self) -> np.ndarray:
        """Convert tensor signature to numerical vector representation"""
        modality_encoding = list(ModalityType).index(self.modality)
        return np.array([
            modality_encoding,
            self.depth,
            self.context,
            self.salience,
            self.autonomy_index
        ])
    
    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'TensorSignature':
        """Create tensor signature from numerical vector"""
        if len(vector) != 5:
            raise ValueError("Vector must have exactly 5 elements")
        
        modality = list(ModalityType)[int(vector[0])]
        return cls(
            modality=modality,
            depth=int(vector[1]),
            context=float(vector[2]),
            salience=float(vector[3]),
            autonomy_index=float(vector[4])
        )


@dataclass 
class CognitivePrimitive:
    """
    Atomic cognitive primitive that serves as building block for agentic operations.
    
    Each primitive carries a tensor signature and can be translated bidirectionally
    between ML representations and AtomSpace hypergraph patterns.
    """
    id: str
    signature: TensorSignature
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    
    def to_atomspace_pattern(self) -> Dict[str, Any]:
        """
        Convert cognitive primitive to AtomSpace hypergraph pattern representation.
        
        This creates a structured pattern that can be used in OpenCog AtomSpace
        for symbolic reasoning and graph-based operations.
        """
        return {
            "atom_type": "ConceptNode",
            "name": self.id,
            "tv": {
                "strength": self.signature.salience,
                "confidence": self.signature.context
            },
            "signature": {
                "modality": self.signature.modality.value,
                "depth": self.signature.depth,
                "context": self.signature.context,
                "salience": self.signature.salience,
                "autonomy_index": self.signature.autonomy_index
            },
            "content": self._serialize_content(),
            "metadata": self.metadata,
            "relationships": self.relationships
        }
    
    @classmethod
    def from_atomspace_pattern(cls, pattern: Dict[str, Any]) -> 'CognitivePrimitive':
        """
        Create cognitive primitive from AtomSpace hypergraph pattern.
        
        This enables bidirectional translation from symbolic representations
        back to cognitive primitives for ML processing.
        """
        signature_data = pattern.get("signature", {})
        modality = ModalityType(signature_data.get("modality", "symbolic"))
        
        signature = TensorSignature(
            modality=modality,
            depth=signature_data.get("depth", 0),
            context=signature_data.get("context", 0.5),
            salience=signature_data.get("salience", 0.5),
            autonomy_index=signature_data.get("autonomy_index", 0.5)
        )
        
        return cls(
            id=pattern.get("name", ""),
            signature=signature,
            content=pattern.get("content"),
            metadata=pattern.get("metadata", {}),
            relationships=pattern.get("relationships", [])
        )
    
    def _serialize_content(self) -> Any:
        """Serialize content for AtomSpace representation"""
        if isinstance(self.content, np.ndarray):
            return self.content.tolist()
        return self.content
    
    def get_ml_features(self) -> np.ndarray:
        """Extract ML-compatible feature vector from cognitive primitive"""
        base_features = self.signature.to_vector()
        
        # Add content-based features if content is numerical
        if isinstance(self.content, (int, float)):
            content_features = np.array([self.content])
        elif isinstance(self.content, np.ndarray):
            content_features = self.content.flatten()
        elif isinstance(self.content, list) and all(isinstance(x, (int, float)) for x in self.content):
            content_features = np.array(self.content)
        else:
            # For non-numerical content, use hash-based encoding
            content_features = np.array([hash(str(self.content)) % 10000])
        
        return np.concatenate([base_features, content_features])
    
    def calculate_compatibility(self, other: 'CognitivePrimitive') -> float:
        """
        Calculate compatibility score between two cognitive primitives.
        
        This is used for determining relationships and clustering in the hypergraph.
        """
        if self.signature.modality != other.signature.modality:
            modality_score = 0.3
        else:
            modality_score = 1.0
        
        depth_score = 1.0 / (1.0 + abs(self.signature.depth - other.signature.depth))
        context_score = 1.0 - abs(self.signature.context - other.signature.context)
        salience_score = 1.0 - abs(self.signature.salience - other.signature.salience)
        autonomy_score = 1.0 - abs(self.signature.autonomy_index - other.signature.autonomy_index)
        
        return (modality_score + depth_score + context_score + salience_score + autonomy_score) / 5.0


class CognitivePrimitiveFactory:
    """Factory for creating cognitive primitives from various input types"""
    
    @staticmethod
    def from_text(text: str, salience: float = 0.5, context: float = 0.5) -> CognitivePrimitive:
        """Create cognitive primitive from text input"""
        signature = TensorSignature(
            modality=ModalityType.TEXTUAL,
            depth=0,
            context=context,
            salience=salience,
            autonomy_index=0.5
        )
        
        return CognitivePrimitive(
            id=f"text_{hash(text) % 10000}",
            signature=signature,
            content=text
        )
    
    @staticmethod
    def from_numerical(data: Union[float, int, List[float], np.ndarray], 
                      salience: float = 0.5, context: float = 0.5) -> CognitivePrimitive:
        """Create cognitive primitive from numerical data"""
        signature = TensorSignature(
            modality=ModalityType.NUMERICAL,
            depth=0,
            context=context,
            salience=salience,
            autonomy_index=0.5
        )
        
        if isinstance(data, (list, np.ndarray)):
            content = np.array(data)
            primitive_id = f"num_{hash(str(data)) % 10000}"
        else:
            content = data
            primitive_id = f"num_{hash(str(data)) % 10000}"
        
        return CognitivePrimitive(
            id=primitive_id,
            signature=signature,
            content=content
        )
    
    @staticmethod
    def from_relation(source: str, relation: str, target: str, 
                     salience: float = 0.5, context: float = 0.5) -> CognitivePrimitive:
        """Create cognitive primitive representing a relationship"""
        signature = TensorSignature(
            modality=ModalityType.RELATIONAL,
            depth=1,
            context=context,
            salience=salience,
            autonomy_index=0.7  # Relations have higher autonomy
        )
        
        return CognitivePrimitive(
            id=f"rel_{source}_{relation}_{target}",
            signature=signature,
            content={
                "source": source,
                "relation": relation,
                "target": target
            },
            relationships=[source, target]
        )