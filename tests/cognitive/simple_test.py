#!/usr/bin/env python3
"""
Simple tests for Cognitive Architecture Phase 1 implementation

Tests core functionality without external dependencies.
"""

import sys
import traceback
import numpy as np

# Add current directory to path
sys.path.insert(0, '.')

from mem0.cognitive import (
    CognitivePrimitive, TensorSignature, ModalityType, CognitivePrimitiveFactory,
    SchemeGrammar, CognitiveGrammarService,
    TensorFragment, HypergraphEncoder,
    CognitiveVerifier, HypergraphValidator
)
from mem0.cognitive.verification import CognitiveVisualizer


def test_cognitive_primitives():
    """Test cognitive primitives functionality"""
    print("Testing Cognitive Primitives...")
    
    # Test tensor signature creation
    signature = TensorSignature(
        modality=ModalityType.TEXTUAL,
        depth=1,
        context=0.8,
        salience=0.9,
        autonomy_index=0.6
    )
    assert signature.modality == ModalityType.TEXTUAL
    
    # Test vector conversion
    vector = signature.to_vector()
    assert len(vector) == 5
    
    # Test round-trip conversion
    reconstructed = TensorSignature.from_vector(vector)
    assert reconstructed.modality == signature.modality
    
    # Test primitive factory
    text_primitive = CognitivePrimitiveFactory.from_text("test text", salience=0.8)
    assert text_primitive.signature.modality == ModalityType.TEXTUAL
    assert text_primitive.content == "test text"
    
    num_primitive = CognitivePrimitiveFactory.from_numerical([1, 2, 3])
    assert num_primitive.signature.modality == ModalityType.NUMERICAL
    
    rel_primitive = CognitivePrimitiveFactory.from_relation("A", "relates_to", "B")
    assert rel_primitive.signature.modality == ModalityType.RELATIONAL
    
    # Test AtomSpace conversion
    pattern = text_primitive.to_atomspace_pattern()
    assert pattern["atom_type"] == "ConceptNode"
    
    reconstructed_primitive = CognitivePrimitive.from_atomspace_pattern(pattern)
    assert reconstructed_primitive.id == text_primitive.id
    
    print("✓ Cognitive Primitives tests passed")
    return [text_primitive, num_primitive, rel_primitive]


def test_scheme_grammar(primitives):
    """Test Scheme cognitive grammar"""
    print("Testing Scheme Grammar...")
    
    service = CognitiveGrammarService()
    
    # Register primitives
    for primitive in primitives:
        service.register_primitive(primitive)
    
    # Test composition
    compose_result = service.process_request('compose', {
        'primitive_ids': [primitives[0].id, primitives[1].id]
    })
    assert compose_result['status'] == 'success'
    
    # Test relation
    relate_result = service.process_request('relate', {
        'source_id': primitives[0].id,
        'target_id': primitives[1].id
    })
    assert relate_result['status'] == 'success'
    
    # Test abstraction
    abstract_result = service.process_request('abstract', {
        'primitive_ids': [p.id for p in primitives]
    })
    assert abstract_result['status'] == 'success'
    
    print("✓ Scheme Grammar tests passed")


def test_tensor_fragments(primitives):
    """Test tensor fragment architecture"""
    print("Testing Tensor Fragments...")
    
    encoder = HypergraphEncoder()
    
    # Create fragment from primitives
    fragment = encoder.create_fragment_from_primitives(primitives, "test_fragment")
    assert fragment.id == "test_fragment"
    assert len(fragment.primitives) == len(primitives)
    assert fragment.tensor_data.size > 0
    
    # Test AtomSpace conversion
    atomspace_graph = fragment.to_atomspace_hypergraph()
    assert "fragment_id" in atomspace_graph
    assert "nodes" in atomspace_graph
    assert "edges" in atomspace_graph
    
    # Test reconstruction
    reconstructed = TensorFragment.from_atomspace_hypergraph(atomspace_graph)
    assert reconstructed.id == fragment.id
    
    # Test merging
    fragment2 = encoder.create_fragment_from_primitives(primitives[:2], "test_fragment_2")
    merged = fragment.merge_with(fragment2, "weighted_average")
    assert "merged_" in merged.id
    
    print("✓ Tensor Fragments tests passed")
    return fragment


def test_verification_visualization(primitives, fragment):
    """Test verification and visualization"""
    print("Testing Verification & Visualization...")
    
    # Test cognitive verifier
    verifier = CognitiveVerifier()
    for primitive in primitives:
        result = verifier.verify_primitive(primitive)
        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'score')
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
    
    # Test collection verification
    collection_result = verifier.verify_primitive_collection(primitives)
    assert hasattr(collection_result, 'is_valid')
    assert hasattr(collection_result, 'score')
    
    # Test hypergraph validator
    validator = HypergraphValidator()
    fragment_result = validator.validate_fragment(fragment)
    assert hasattr(fragment_result, 'is_valid')
    assert hasattr(fragment_result, 'score')
    
    # Test visualizer
    visualizer = CognitiveVisualizer()
    
    # Test primitive visualization
    primitive_viz = visualizer.visualize_primitive(primitives[0])
    assert isinstance(primitive_viz, str)
    assert len(primitive_viz) > 0
    
    # Test fragment visualization
    fragment_viz = visualizer.visualize_fragment(fragment)
    assert isinstance(fragment_viz, str)
    assert len(fragment_viz) > 0
    
    # Test validation report
    report = visualizer.create_validation_report(fragment_result)
    assert isinstance(report, str)
    assert len(report) > 0
    
    print("✓ Verification & Visualization tests passed")


def test_integration():
    """Test complete integration workflow"""
    print("Testing Integration...")
    
    # Create knowledge scenario
    knowledge_primitives = [
        CognitivePrimitiveFactory.from_text("Artificial intelligence enables automation"),
        CognitivePrimitiveFactory.from_text("Machine learning is a subset of AI"),
        CognitivePrimitiveFactory.from_relation("AI", "includes", "machine_learning"),
        CognitivePrimitiveFactory.from_numerical([0.9, 0.8, 0.7])
    ]
    
    # Create tensor fragment
    encoder = HypergraphEncoder()
    knowledge_fragment = encoder.create_fragment_from_primitives(
        knowledge_primitives, "knowledge_integration_test"
    )
    
    # Apply Scheme operations
    service = CognitiveGrammarService()
    for primitive in knowledge_primitives:
        service.register_primitive(primitive)
    
    # Test composition of knowledge
    compose_result = service.process_request('compose', {
        'primitive_ids': [knowledge_primitives[0].id, knowledge_primitives[1].id]
    })
    assert compose_result['status'] == 'success'
    
    # Verify the complete structure
    verifier = CognitiveVerifier()
    validator = HypergraphValidator()
    
    collection_result = verifier.verify_primitive_collection(knowledge_primitives)
    fragment_result = validator.validate_fragment(knowledge_fragment)
    
    assert hasattr(collection_result, 'score')
    assert hasattr(fragment_result, 'score')
    
    # Test AtomSpace roundtrip
    atomspace_graph = knowledge_fragment.to_atomspace_hypergraph()
    reconstructed_fragment = TensorFragment.from_atomspace_hypergraph(atomspace_graph)
    
    assert reconstructed_fragment.id == knowledge_fragment.id
    assert len(reconstructed_fragment.primitives) == len(knowledge_fragment.primitives)
    
    print("✓ Integration tests passed")


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("🧬 Cognitive Architecture Phase 1 Tests")
    print("=" * 60)
    print()
    
    try:
        # Run test suites
        primitives = test_cognitive_primitives()
        test_scheme_grammar(primitives)
        fragment = test_tensor_fragments(primitives)
        test_verification_visualization(primitives, fragment)
        test_integration()
        
        print()
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("✅ 1.1 Scheme Cognitive Grammar Microservices")
        print("✅ 1.2 Tensor Fragment Architecture")
        print("✅ 1.3 Verification & Visualization")
        print()
        print("🎯 Tensor signature [modality, depth, context, salience, autonomy_index] ✓")
        print("🔗 AtomSpace hypergraph patterns ✓")
        print("🛠️  Cognitive microservice architecture ✓")
        print()
        return True
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        print("Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)