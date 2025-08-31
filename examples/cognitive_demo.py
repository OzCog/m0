#!/usr/bin/env python3
"""
Cognitive Architecture Demo

This example demonstrates Phase 1 implementation of the cognitive primitives
and foundational hypergraph encoding, including:

1.1 Scheme Cognitive Grammar Microservices
1.2 Tensor Fragment Architecture
1.3 Verification & Visualization

Tensor signature: [modality, depth, context, salience, autonomy_index]
"""

from mem0.cognitive import (
    CognitivePrimitive, TensorSignature, ModalityType, CognitivePrimitiveFactory,
    SchemeGrammar, CognitiveGrammarService,
    TensorFragment, HypergraphEncoder,
    CognitiveVerifier, HypergraphValidator
)
from mem0.cognitive.verification import CognitiveVisualizer
import json


def demonstrate_cognitive_primitives():
    """Demonstrate core cognitive primitive functionality"""
    print("=== 1. Cognitive Primitives Demo ===\n")
    
    # Create different types of primitives
    text_primitive = CognitivePrimitiveFactory.from_text(
        "Artificial intelligence is transforming society",
        salience=0.9, context=0.8
    )
    
    numerical_primitive = CognitivePrimitiveFactory.from_numerical(
        [0.1, 0.5, 0.9, 0.3], salience=0.7, context=0.6
    )
    
    relation_primitive = CognitivePrimitiveFactory.from_relation(
        "AI", "transforms", "society", salience=0.8, context=0.7
    )
    
    primitives = [text_primitive, numerical_primitive, relation_primitive]
    
    # Demonstrate tensor signatures
    print("Cognitive Primitives Created:")
    for primitive in primitives:
        vector = primitive.signature.to_vector()
        print(f"- {primitive.id}: {primitive.signature.modality.value}")
        print(f"  Tensor signature: {vector}")
        print(f"  Content: {primitive.content}")
        print()
    
    # Demonstrate AtomSpace conversion
    print("AtomSpace Hypergraph Patterns:")
    for primitive in primitives:
        pattern = primitive.to_atomspace_pattern()
        print(f"- {primitive.id}:")
        print(f"  Type: {pattern['atom_type']}")
        print(f"  Strength: {pattern['tv']['strength']:.3f}")
        print(f"  Confidence: {pattern['tv']['confidence']:.3f}")
        print()
    
    return primitives


def demonstrate_scheme_grammar(primitives):
    """Demonstrate Scheme cognitive grammar microservices"""
    print("=== 2. Scheme Cognitive Grammar Demo ===\n")
    
    # Initialize grammar service
    service = CognitiveGrammarService()
    
    # Register primitives
    for primitive in primitives:
        service.register_primitive(primitive)
    
    # Demonstrate composition
    print("Composition Operation:")
    compose_result = service.process_request('compose', {
        'primitive_ids': [p.id for p in primitives[:2]]
    })
    
    if compose_result['status'] == 'success':
        composed = compose_result['result']['composed_primitive']
        print(f"✓ Composed primitive created")
        print(f"  ID: {composed['name']}")
        print(f"  Scheme: {compose_result['result']['scheme_expression']}")
        print()
    
    # Demonstrate relation
    print("Relation Operation:")
    relate_result = service.process_request('relate', {
        'source_id': primitives[0].id,
        'target_id': primitives[1].id
    })
    
    if relate_result['status'] == 'success':
        relation = relate_result['result']['relation_primitive']
        print(f"✓ Relation primitive created")
        print(f"  ID: {relation['name']}")
        print(f"  Scheme: {relate_result['result']['scheme_expression']}")
        print()
    
    # Demonstrate abstraction
    print("Abstraction Operation:")
    abstract_result = service.process_request('abstract', {
        'primitive_ids': [p.id for p in primitives]
    })
    
    if abstract_result['status'] == 'success':
        abstract = abstract_result['result']['abstract_primitive']
        print(f"✓ Abstract primitive created")
        print(f"  ID: {abstract['name']}")
        print(f"  Scheme: {abstract_result['result']['scheme_expression']}")
        print()


def demonstrate_tensor_fragments(primitives):
    """Demonstrate tensor fragment architecture"""
    print("=== 3. Tensor Fragment Architecture Demo ===\n")
    
    # Create hypergraph encoder
    encoder = HypergraphEncoder()
    
    # Create tensor fragment from primitives
    fragment = encoder.create_fragment_from_primitives(
        primitives, "demo_fragment"
    )
    
    print("Tensor Fragment Created:")
    print(f"- ID: {fragment.id}")
    print(f"- Signature: {fragment.signature.modality.value} (depth={fragment.signature.depth})")
    print(f"- Tensor shape: {fragment.tensor_data.shape}")
    print(f"- Primitives: {len(fragment.primitives)}")
    print(f"- Connections: {len(fragment.connections)}")
    print()
    
    # Demonstrate bidirectional conversion
    print("Bidirectional AtomSpace Conversion:")
    
    # Fragment -> AtomSpace
    atomspace_graph = fragment.to_atomspace_hypergraph()
    print(f"✓ Fragment → AtomSpace: {len(atomspace_graph['nodes'])} nodes, {len(atomspace_graph['edges'])} edges")
    
    # AtomSpace -> Fragment
    reconstructed_fragment = TensorFragment.from_atomspace_hypergraph(atomspace_graph)
    print(f"✓ AtomSpace → Fragment: {reconstructed_fragment.id}")
    print(f"  Tensor shape: {reconstructed_fragment.tensor_data.shape}")
    print()
    
    # Demonstrate fragment merging
    print("Fragment Merging:")
    
    # Create another fragment
    second_fragment = encoder.create_fragment_from_primitives(
        primitives[1:], "demo_fragment_2"
    )
    
    # Merge fragments
    merged_fragment = fragment.merge_with(second_fragment, "weighted_average")
    print(f"✓ Merged fragment created: {merged_fragment.id}")
    print(f"  Original shapes: {fragment.tensor_data.shape}, {second_fragment.tensor_data.shape}")
    print(f"  Merged shape: {merged_fragment.tensor_data.shape}")
    print()
    
    return fragment


def demonstrate_verification_visualization(primitives, fragment):
    """Demonstrate verification and visualization"""
    print("=== 4. Verification & Visualization Demo ===\n")
    
    # Initialize verification components
    cognitive_verifier = CognitiveVerifier()
    hypergraph_validator = HypergraphValidator()
    visualizer = CognitiveVisualizer()
    
    # Verify primitives
    print("Cognitive Primitive Verification:")
    for primitive in primitives:
        result = cognitive_verifier.verify_primitive(primitive)
        status = "✓ VALID" if result.is_valid else "✗ INVALID"
        print(f"- {primitive.id}: {status} (score: {result.score:.3f})")
    print()
    
    # Verify fragment
    print("Tensor Fragment Validation:")
    fragment_result = hypergraph_validator.validate_fragment(fragment)
    status = "✓ VALID" if fragment_result.is_valid else "✗ INVALID"
    print(f"- {fragment.id}: {status} (score: {fragment_result.score:.3f})")
    print()
    
    # Demonstrate visualization
    print("Primitive Visualization:")
    print("-" * 40)
    print(visualizer.visualize_primitive(primitives[0]))
    print("-" * 40)
    print()
    
    print("Fragment Visualization:")
    print("-" * 40)
    print(visualizer.visualize_fragment(fragment))
    print("-" * 40)
    print()
    
    # Create validation report
    print("Validation Report:")
    print("-" * 40)
    print(visualizer.create_validation_report(fragment_result))
    print("-" * 40)
    print()


def demonstrate_full_workflow():
    """Demonstrate complete cognitive architecture workflow"""
    print("=== 5. Complete Workflow Demo ===\n")
    
    # Create a knowledge scenario
    knowledge_primitives = [
        CognitivePrimitiveFactory.from_text("Machine learning enables pattern recognition"),
        CognitivePrimitiveFactory.from_text("Neural networks are computational models"),
        CognitivePrimitiveFactory.from_text("Deep learning uses multiple layers"),
        CognitivePrimitiveFactory.from_relation("machine_learning", "uses", "neural_networks"),
        CognitivePrimitiveFactory.from_relation("neural_networks", "implement", "deep_learning"),
        CognitivePrimitiveFactory.from_numerical([0.9, 0.8, 0.7, 0.6])  # Confidence scores
    ]
    
    # Create hypergraph encoder and fragment
    encoder = HypergraphEncoder()
    knowledge_fragment = encoder.create_fragment_from_primitives(
        knowledge_primitives, "knowledge_graph_fragment"
    )
    
    # Apply Scheme operations
    service = CognitiveGrammarService()
    for primitive in knowledge_primitives:
        service.register_primitive(primitive)
    
    # Create an abstraction of the knowledge
    abstraction_result = service.process_request('abstract', {
        'primitive_ids': [p.id for p in knowledge_primitives[:3]]
    })
    
    print("Knowledge Processing Workflow:")
    print(f"✓ Created {len(knowledge_primitives)} knowledge primitives")
    print(f"✓ Assembled into tensor fragment: {knowledge_fragment.id}")
    print(f"✓ Applied abstraction operation")
    print(f"✓ Fragment tensor shape: {knowledge_fragment.tensor_data.shape}")
    print(f"✓ Hypergraph nodes: {len(knowledge_fragment.hypergraph_structure['nodes'])}")
    print(f"✓ Hypergraph edges: {len(knowledge_fragment.hypergraph_structure['edges'])}")
    print()
    
    # Verify the complete knowledge structure
    verifier = CognitiveVerifier()
    collection_result = verifier.verify_primitive_collection(knowledge_primitives)
    
    print("Knowledge Structure Validation:")
    print(f"- Collection validity: {'✓ VALID' if collection_result.is_valid else '✗ INVALID'}")
    print(f"- Overall score: {collection_result.score:.3f}")
    print(f"- Individual primitive scores: {len(collection_result.metadata['individual_scores'])} evaluated")
    print(f"- Collection consistency: {collection_result.metadata['collection_consistency']:.3f}")
    print()
    
    return knowledge_fragment


def main():
    """Main demonstration function"""
    print("🧬 Phase 1: Cognitive Primitives & Foundational Hypergraph Encoding")
    print("=" * 70)
    print()
    
    # Run demonstrations
    primitives = demonstrate_cognitive_primitives()
    demonstrate_scheme_grammar(primitives)
    fragment = demonstrate_tensor_fragments(primitives)
    demonstrate_verification_visualization(primitives, fragment)
    knowledge_fragment = demonstrate_full_workflow()
    
    print("=== Summary ===")
    print()
    print("✅ 1.1 Scheme Cognitive Grammar Microservices")
    print("   - Cognitive primitives with tensor signatures")
    print("   - Scheme-like grammar operations (compose, relate, abstract)")
    print("   - Microservice interface for cognitive processing")
    print()
    print("✅ 1.2 Tensor Fragment Architecture")
    print("   - Bidirectional ML ↔ AtomSpace hypergraph translation")
    print("   - Tensor fragment composition and optimization")
    print("   - Hypergraph encoding and pattern management")
    print()
    print("✅ 1.3 Verification & Visualization")
    print("   - Cognitive primitive and fragment validation")
    print("   - Hypergraph consistency checking")
    print("   - Text-based visualization and reporting")
    print()
    print("🎯 Tensor signature [modality, depth, context, salience, autonomy_index] implemented")
    print("🔗 AtomSpace hypergraph patterns fully supported")
    print("🛠️  Microservice architecture ready for cognitive operations")
    print()
    print("Phase 1 implementation complete! 🚀")


if __name__ == "__main__":
    main()