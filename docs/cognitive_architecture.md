# Cognitive Architecture Phase 1: Implementation Guide

This document describes the implementation of **Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding** for the mem0 cognitive architecture.

## Overview

Phase 1 establishes the foundational vocabulary and bidirectional translation between agentic kernel ML primitives and AtomSpace hypergraph patterns using the tensor signature:

**`[modality, depth, context, salience, autonomy_index]`**

## Implementation Components

### 1.1 Scheme Cognitive Grammar Microservices

The cognitive grammar provides Scheme-like functional composition for cognitive operations:

#### Core Classes

- **`CognitivePrimitive`**: Atomic cognitive building blocks with tensor signatures
- **`TensorSignature`**: Five-dimensional cognitive encoding
- **`ModalityType`**: Enumeration of cognitive modalities (symbolic, numerical, textual, visual, etc.)
- **`CognitivePrimitiveFactory`**: Factory for creating primitives from various inputs

#### Scheme Operations

- **`compose`**: Combine multiple primitives into composite structures
- **`relate`**: Create relational connections between primitives
- **`abstract`**: Extract common patterns from primitive collections
- **`specialize`**: Apply concrete details to abstract primitives

#### Example Usage

```python
from mem0.cognitive import CognitivePrimitiveFactory, CognitiveGrammarService

# Create primitives
p1 = CognitivePrimitiveFactory.from_text("Machine learning processes data")
p2 = CognitivePrimitiveFactory.from_numerical([0.9, 0.8, 0.7])

# Initialize grammar service
service = CognitiveGrammarService()
service.register_primitive(p1)
service.register_primitive(p2)

# Compose primitives
result = service.process_request('compose', {
    'primitive_ids': [p1.id, p2.id]
})

print(f"Composed: {result['result']['scheme_expression']}")
# Output: (compose (atom text_1234) (atom num_5678))
```

### 1.2 Tensor Fragment Architecture

Tensor fragments provide bidirectional translation between ML representations and AtomSpace hypergraph patterns:

#### Core Classes

- **`TensorFragment`**: Composite structure containing tensor data and hypergraph patterns
- **`HypergraphEncoder`**: Bidirectional ML ↔ AtomSpace translator
- **Pattern Templates**: Standard hypergraph structures for common cognitive patterns

#### Key Features

- **Bidirectional Conversion**: Seamless translation between tensor and symbolic representations
- **Fragment Merging**: Combine fragments using weighted averaging or concatenation
- **Optimization**: Automatic tensor compression and hypergraph simplification

#### Example Usage

```python
from mem0.cognitive import HypergraphEncoder, CognitivePrimitiveFactory

# Create primitives
primitives = [
    CognitivePrimitiveFactory.from_text("Neural networks learn patterns"),
    CognitivePrimitiveFactory.from_relation("network", "learns", "patterns")
]

# Create tensor fragment
encoder = HypergraphEncoder()
fragment = encoder.create_fragment_from_primitives(primitives, "neural_knowledge")

# Convert to AtomSpace
atomspace_graph = fragment.to_atomspace_hypergraph()
print(f"Nodes: {len(atomspace_graph['nodes'])}")
print(f"Edges: {len(atomspace_graph['edges'])}")

# Convert back to fragment
reconstructed = TensorFragment.from_atomspace_hypergraph(atomspace_graph)
assert reconstructed.id == fragment.id
```

### 1.3 Verification & Visualization

Comprehensive validation and visualization tools for cognitive structures:

#### Verification Components

- **`CognitiveVerifier`**: Validates primitive consistency and collections
- **`HypergraphValidator`**: Validates hypergraph structural integrity
- **Validator Classes**: Specialized validators for signatures, content, and relationships

#### Visualization Tools

- **`CognitiveVisualizer`**: Text-based visualization of primitives and fragments
- **Validation Reports**: Formatted validation results with errors and warnings
- **Hypergraph Summaries**: Statistical overviews of graph structures

#### Example Usage

```python
from mem0.cognitive import CognitiveVerifier, CognitivePrimitiveFactory
from mem0.cognitive.verification import CognitiveVisualizer

# Create and verify primitive
primitive = CognitivePrimitiveFactory.from_text("Test cognitive primitive")
verifier = CognitiveVerifier()
result = verifier.verify_primitive(primitive)

print(f"Valid: {result.is_valid}, Score: {result.score:.3f}")

# Visualize primitive
visualizer = CognitiveVisualizer()
print(visualizer.visualize_primitive(primitive))
```

## Tensor Signature Specification

The five-dimensional tensor signature encodes cognitive state:

| Dimension | Range | Description |
|-----------|-------|-------------|
| `modality` | enum | Type of cognitive content (symbolic, numerical, textual, etc.) |
| `depth` | 0+ | Hierarchical depth in cognitive structure |
| `context` | [0.0, 1.0] | Contextual specificity and relevance |
| `salience` | [0.0, 1.0] | Attention weight and importance |
| `autonomy_index` | [0.0, 1.0] | Degree of autonomous processing capability |

### Example Tensor Signatures

```python
# High-salience textual concept
TensorSignature(
    modality=ModalityType.TEXTUAL,
    depth=0,
    context=0.8,
    salience=0.9,
    autonomy_index=0.5
)
# Vector: [2.0, 0.0, 0.8, 0.9, 0.5]

# Deep relational structure
TensorSignature(
    modality=ModalityType.RELATIONAL,
    depth=3,
    context=0.6,
    salience=0.7,
    autonomy_index=0.8
)
# Vector: [7.0, 3.0, 0.6, 0.7, 0.8]
```

## AtomSpace Hypergraph Patterns

The system supports standard OpenCog AtomSpace patterns:

### Node Types

- **`ConceptNode`**: Basic cognitive concepts
- **`PatternNode`**: Complex cognitive patterns

### Link Types

- **`SimilarityLink`**: Similarity relationships between concepts
- **`InheritanceLink`**: Hierarchical concept relationships
- **`CompositionLink`**: Part-whole relationships

### Truth Values

All atoms include truth values with:
- **Strength**: Derived from primitive salience
- **Confidence**: Derived from primitive context

## API Reference

### Core Classes

#### `CognitivePrimitive`
```python
class CognitivePrimitive:
    def __init__(self, id: str, signature: TensorSignature, content: Any, 
                 metadata: Dict[str, Any] = None, relationships: List[str] = None)
    
    def to_atomspace_pattern(self) -> Dict[str, Any]
    def get_ml_features(self) -> np.ndarray
    def calculate_compatibility(self, other: 'CognitivePrimitive') -> float
    
    @classmethod
    def from_atomspace_pattern(cls, pattern: Dict[str, Any]) -> 'CognitivePrimitive'
```

#### `TensorFragment`
```python
class TensorFragment:
    def __init__(self, id: str, signature: TensorSignature, tensor_data: np.ndarray,
                 hypergraph_structure: Dict[str, Any], primitives: List[CognitivePrimitive],
                 connections: Dict[str, float], metadata: Dict[str, Any])
    
    def to_atomspace_hypergraph(self) -> Dict[str, Any]
    def merge_with(self, other: 'TensorFragment', strategy: str = "weighted_average") -> 'TensorFragment'
    
    @classmethod
    def from_atomspace_hypergraph(cls, hypergraph: Dict[str, Any]) -> 'TensorFragment'
```

#### `CognitiveGrammarService`
```python
class CognitiveGrammarService:
    def register_primitive(self, primitive: CognitivePrimitive)
    def process_request(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]
    
    # Supported operations: 'compose', 'relate', 'abstract', 'specialize', 'evaluate'
```

### Factory Methods

#### `CognitivePrimitiveFactory`
```python
@staticmethod
def from_text(text: str, salience: float = 0.5, context: float = 0.5) -> CognitivePrimitive

@staticmethod  
def from_numerical(data: Union[float, int, List[float], np.ndarray],
                  salience: float = 0.5, context: float = 0.5) -> CognitivePrimitive

@staticmethod
def from_relation(source: str, relation: str, target: str,
                 salience: float = 0.5, context: float = 0.5) -> CognitivePrimitive
```

## Integration with mem0

The cognitive architecture integrates seamlessly with existing mem0 components:

### Graph Memory Enhancement

```python
from mem0.cognitive import HypergraphEncoder
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT

# Enhance existing graph memory with cognitive primitives
encoder = HypergraphEncoder()
# Integration with existing mem0 graph utilities...
```

### Memory Operations

The cognitive primitives can enhance mem0's memory operations:

- **Enhanced Retrieval**: Use tensor signatures for semantic similarity
- **Cognitive Filtering**: Filter memories by modality and salience  
- **Hierarchical Organization**: Organize memories by depth and autonomy

## Testing

Run the cognitive architecture tests:

```bash
# Simple functional tests
python tests/cognitive/simple_test.py

# Full demonstration
python examples/cognitive_demo.py
```

## Future Extensions

Phase 1 provides the foundation for future cognitive capabilities:

- **Phase 2**: Advanced reasoning patterns and inference chains
- **Phase 3**: Learning and adaptation mechanisms
- **Phase 4**: Multi-agent cognitive coordination

## License

This implementation follows the same Apache 2.0 license as the main mem0 project.