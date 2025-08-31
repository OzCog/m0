"""
Tests for Cognitive Primitives & Foundational Hypergraph Encoding

Tests the Phase 1 implementation components:
- Cognitive primitives and tensor signatures
- Scheme cognitive grammar operations
- Tensor fragment architecture
- Verification and visualization
"""

import pytest
import numpy as np
from unittest.mock import Mock

from mem0.cognitive import (
    CognitivePrimitive, TensorSignature, ModalityType, CognitivePrimitiveFactory,
    SchemeGrammar, CognitiveGrammarService,
    TensorFragment, HypergraphEncoder,
    CognitiveVerifier, HypergraphValidator
)
from mem0.cognitive.verification import CognitiveVisualizer


class TestCognitivePrimitives:
    """Test cognitive primitives and tensor signatures"""
    
    def test_tensor_signature_creation(self):
        """Test tensor signature creation and validation"""
        signature = TensorSignature(
            modality=ModalityType.TEXTUAL,
            depth=2,
            context=0.7,
            salience=0.8,
            autonomy_index=0.6
        )
        
        assert signature.modality == ModalityType.TEXTUAL
        assert signature.depth == 2
        assert signature.context == 0.7
        assert signature.salience == 0.8
        assert signature.autonomy_index == 0.6
    
    def test_tensor_signature_validation(self):
        """Test tensor signature validation constraints"""
        # Test invalid depth
        with pytest.raises(ValueError, match="Depth must be non-negative"):
            TensorSignature(ModalityType.TEXTUAL, -1, 0.5, 0.5, 0.5)
        
        # Test invalid context
        with pytest.raises(ValueError, match="Context must be between 0.0 and 1.0"):
            TensorSignature(ModalityType.TEXTUAL, 0, 1.5, 0.5, 0.5)
        
        # Test invalid salience
        with pytest.raises(ValueError, match="Salience must be between 0.0 and 1.0"):
            TensorSignature(ModalityType.TEXTUAL, 0, 0.5, -0.1, 0.5)
        
        # Test invalid autonomy index
        with pytest.raises(ValueError, match="Autonomy index must be between 0.0 and 1.0"):
            TensorSignature(ModalityType.TEXTUAL, 0, 0.5, 0.5, 2.0)
    
    def test_tensor_signature_vector_conversion(self):
        """Test tensor signature to vector conversion"""
        signature = TensorSignature(
            modality=ModalityType.NUMERICAL,
            depth=1,
            context=0.6,
            salience=0.9,
            autonomy_index=0.4
        )
        
        vector = signature.to_vector()
        assert len(vector) == 5
        assert vector[0] == 1  # NUMERICAL is at index 1
        assert vector[1] == 1
        assert vector[2] == 0.6
        assert vector[3] == 0.9
        assert vector[4] == 0.4
        
        # Test round-trip conversion
        reconstructed = TensorSignature.from_vector(vector)
        assert reconstructed.modality == signature.modality
        assert reconstructed.depth == signature.depth
        assert reconstructed.context == signature.context
        assert reconstructed.salience == signature.salience
        assert reconstructed.autonomy_index == signature.autonomy_index
    
    def test_cognitive_primitive_factory(self):
        """Test cognitive primitive factory methods"""
        # Test text primitive
        text_primitive = CognitivePrimitiveFactory.from_text("test text", salience=0.8)
        assert text_primitive.signature.modality == ModalityType.TEXTUAL
        assert text_primitive.signature.salience == 0.8
        assert text_primitive.content == "test text"
        
        # Test numerical primitive
        num_primitive = CognitivePrimitiveFactory.from_numerical([1, 2, 3], salience=0.7)
        assert num_primitive.signature.modality == ModalityType.NUMERICAL
        assert num_primitive.signature.salience == 0.7
        np.testing.assert_array_equal(num_primitive.content, np.array([1, 2, 3]))
        
        # Test relational primitive
        rel_primitive = CognitivePrimitiveFactory.from_relation("A", "relates_to", "B")
        assert rel_primitive.signature.modality == ModalityType.RELATIONAL
        assert rel_primitive.content["source"] == "A"
        assert rel_primitive.content["relation"] == "relates_to"
        assert rel_primitive.content["target"] == "B"
    
    def test_atomspace_conversion(self):
        """Test bidirectional AtomSpace conversion"""
        primitive = CognitivePrimitiveFactory.from_text("test content")
        
        # Convert to AtomSpace pattern
        pattern = primitive.to_atomspace_pattern()
        assert pattern["atom_type"] == "ConceptNode"
        assert pattern["name"] == primitive.id
        assert "signature" in pattern
        assert "content" in pattern
        
        # Convert back to primitive
        reconstructed = CognitivePrimitive.from_atomspace_pattern(pattern)
        assert reconstructed.id == primitive.id
        assert reconstructed.signature.modality == primitive.signature.modality
        assert reconstructed.content == primitive.content
    
    def test_compatibility_calculation(self):
        """Test primitive compatibility calculation"""
        p1 = CognitivePrimitiveFactory.from_text("test 1", salience=0.8, context=0.7)
        p2 = CognitivePrimitiveFactory.from_text("test 2", salience=0.6, context=0.9)
        
        compatibility = p1.calculate_compatibility(p2)
        assert 0.0 <= compatibility <= 1.0
        
        # Same primitive should have high compatibility with itself
        self_compatibility = p1.calculate_compatibility(p1)
        assert self_compatibility == 1.0


class TestSchemeGrammar:
    """Test Scheme cognitive grammar operations"""
    
    def test_cognitive_grammar_service(self):
        """Test cognitive grammar service operations"""
        service = CognitiveGrammarService()
        
        # Create and register primitives
        p1 = CognitivePrimitiveFactory.from_text("concept A")
        p2 = CognitivePrimitiveFactory.from_text("concept B")
        
        service.register_primitive(p1)
        service.register_primitive(p2)
        
        # Test composition
        result = service.process_request('compose', {
            'primitive_ids': [p1.id, p2.id]
        })
        
        assert result['status'] == 'success'
        assert 'result' in result
        assert 'composed_primitive' in result['result']
        assert 'scheme_expression' in result['result']
        
        # Test relation
        result = service.process_request('relate', {
            'source_id': p1.id,
            'target_id': p2.id
        })
        
        assert result['status'] == 'success'
        assert 'relation_primitive' in result['result']
    
    def test_scheme_grammar_parser(self):
        """Test basic Scheme grammar parsing"""
        grammar = SchemeGrammar()
        
        # Register test primitive
        primitive = CognitivePrimitiveFactory.from_text("test")
        grammar.register_primitive(primitive)
        
        # Test simple atom parsing
        expression = primitive.id
        parsed = grammar.parse(expression)
        
        assert hasattr(parsed, 'evaluate')
        result = parsed.evaluate({})
        assert result.id == primitive.id
    
    def test_invalid_operations(self):
        """Test error handling for invalid operations"""
        service = CognitiveGrammarService()
        
        # Test unknown operation
        result = service.process_request('unknown_op', {})
        assert result['status'] == 'error'
        assert 'Unknown operation' in result['error']
        
        # Test missing parameters
        result = service.process_request('relate', {})
        assert result['status'] == 'error'


class TestTensorFragments:
    """Test tensor fragment architecture"""
    
    def test_tensor_fragment_creation(self):
        """Test tensor fragment creation from primitives"""
        primitives = [
            CognitivePrimitiveFactory.from_text("test 1"),
            CognitivePrimitiveFactory.from_numerical([1, 2]),
            CognitivePrimitiveFactory.from_relation("A", "rel", "B")
        ]
        
        encoder = HypergraphEncoder()
        fragment = encoder.create_fragment_from_primitives(primitives, "test_fragment")
        
        assert fragment.id == "test_fragment"
        assert len(fragment.primitives) == 3
        assert fragment.tensor_data.size > 0
        assert "nodes" in fragment.hypergraph_structure
        assert "edges" in fragment.hypergraph_structure
        assert "patterns" in fragment.hypergraph_structure
    
    def test_fragment_atomspace_conversion(self):
        """Test bidirectional fragment-AtomSpace conversion"""
        primitives = [CognitivePrimitiveFactory.from_text("test")]
        encoder = HypergraphEncoder()
        
        # Create fragment
        fragment = encoder.create_fragment_from_primitives(primitives)
        
        # Convert to AtomSpace
        atomspace_graph = fragment.to_atomspace_hypergraph()
        assert "fragment_id" in atomspace_graph
        assert "nodes" in atomspace_graph
        assert "edges" in atomspace_graph
        assert "patterns" in atomspace_graph
        
        # Convert back to fragment
        reconstructed = TensorFragment.from_atomspace_hypergraph(atomspace_graph)
        assert reconstructed.id == fragment.id
        assert len(reconstructed.primitives) == len(fragment.primitives)
    
    def test_fragment_merging(self):
        """Test fragment merging operations"""
        p1 = [CognitivePrimitiveFactory.from_text("test 1")]
        p2 = [CognitivePrimitiveFactory.from_text("test 2")]
        
        encoder = HypergraphEncoder()
        fragment1 = encoder.create_fragment_from_primitives(p1, "frag1")
        fragment2 = encoder.create_fragment_from_primitives(p2, "frag2")
        
        # Test weighted average merge
        merged = fragment1.merge_with(fragment2, "weighted_average")
        assert "merged_" in merged.id
        assert len(merged.primitives) == len(fragment1.primitives) + len(fragment2.primitives)
        
        # Test concatenate merge
        merged_concat = fragment1.merge_with(fragment2, "concatenate")
        assert "concat_" in merged_concat.id
    
    def test_hypergraph_encoder(self):
        """Test hypergraph encoder functionality"""
        encoder = HypergraphEncoder()
        
        # Test primitive encoding
        primitive = CognitivePrimitiveFactory.from_text("test primitive")
        pattern = encoder.encode_primitive_to_hypergraph(primitive)
        
        assert "id" in pattern
        assert "type" in pattern
        assert "primitive_signature" in pattern
        assert "structure" in pattern
        
        # Test primitive decoding
        decoded = encoder.decode_hypergraph_to_primitive(pattern)
        assert decoded.id == primitive.id
        assert decoded.signature.modality == primitive.signature.modality


class TestVerification:
    """Test verification and validation components"""
    
    def test_cognitive_verifier(self):
        """Test cognitive primitive verification"""
        verifier = CognitiveVerifier()
        
        # Test valid primitive
        valid_primitive = CognitivePrimitiveFactory.from_text("valid test")
        result = verifier.verify_primitive(valid_primitive)
        
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.metadata, dict)
    
    def test_hypergraph_validator(self):
        """Test hypergraph validation"""
        validator = HypergraphValidator()
        
        # Create test fragment
        primitive = CognitivePrimitiveFactory.from_text("test")
        encoder = HypergraphEncoder()
        fragment = encoder.create_fragment_from_primitives([primitive])
        
        # Validate fragment
        result = validator.validate_fragment(fragment)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.score <= 1.0
    
    def test_visualizer(self):
        """Test cognitive visualizer"""
        visualizer = CognitiveVisualizer()
        
        # Test primitive visualization
        primitive = CognitivePrimitiveFactory.from_text("test visualization")
        viz = visualizer.visualize_primitive(primitive)
        
        assert isinstance(viz, str)
        assert len(viz) > 0
        assert primitive.id in viz
        
        # Test fragment visualization
        encoder = HypergraphEncoder()
        fragment = encoder.create_fragment_from_primitives([primitive])
        frag_viz = visualizer.visualize_fragment(fragment)
        
        assert isinstance(frag_viz, str)
        assert len(frag_viz) > 0
        assert fragment.id in frag_viz
    
    def test_primitive_collection_verification(self):
        """Test verification of primitive collections"""
        verifier = CognitiveVerifier()
        
        primitives = [
            CognitivePrimitiveFactory.from_text("test 1"),
            CognitivePrimitiveFactory.from_text("test 2"),
            CognitivePrimitiveFactory.from_numerical([1, 2, 3])
        ]
        
        result = verifier.verify_primitive_collection(primitives)
        assert isinstance(result.is_valid, bool)
        assert 0.0 <= result.score <= 1.0
        assert "primitive_count" in result.metadata
        assert "collection_consistency" in result.metadata


class TestIntegration:
    """Integration tests for complete cognitive architecture"""
    
    def test_full_cognitive_workflow(self):
        """Test complete cognitive processing workflow"""
        # Create primitives
        primitives = [
            CognitivePrimitiveFactory.from_text("AI processes information"),
            CognitivePrimitiveFactory.from_numerical([0.9, 0.8, 0.7]),
            CognitivePrimitiveFactory.from_relation("AI", "processes", "information")
        ]
        
        # Create fragment
        encoder = HypergraphEncoder()
        fragment = encoder.create_fragment_from_primitives(primitives, "workflow_test")
        
        # Apply Scheme operations
        service = CognitiveGrammarService()
        for primitive in primitives:
            service.register_primitive(primitive)
        
        compose_result = service.process_request('compose', {
            'primitive_ids': [primitives[0].id, primitives[1].id]
        })
        
        # Verify results
        verifier = CognitiveVerifier()
        validator = HypergraphValidator()
        
        fragment_validation = validator.validate_fragment(fragment)
        
        # Assertions
        assert len(primitives) == 3
        assert fragment.id == "workflow_test"
        assert compose_result['status'] == 'success'
        assert isinstance(fragment_validation.score, float)
    
    def test_atomspace_roundtrip(self):
        """Test complete AtomSpace roundtrip conversion"""
        # Create complex structure
        primitives = [
            CognitivePrimitiveFactory.from_text("knowledge base"),
            CognitivePrimitiveFactory.from_relation("user", "queries", "knowledge_base")
        ]
        
        encoder = HypergraphEncoder()
        original_fragment = encoder.create_fragment_from_primitives(primitives)
        
        # Convert to AtomSpace and back
        atomspace_graph = original_fragment.to_atomspace_hypergraph()
        reconstructed_fragment = TensorFragment.from_atomspace_hypergraph(atomspace_graph)
        
        # Verify preservation
        assert reconstructed_fragment.id == original_fragment.id
        assert len(reconstructed_fragment.primitives) == len(original_fragment.primitives)
        assert reconstructed_fragment.tensor_data.shape == original_fragment.tensor_data.shape
    
    def test_cognitive_architecture_scalability(self):
        """Test architecture with larger primitive collections"""
        # Create larger collection
        primitives = []
        for i in range(10):
            primitives.append(CognitivePrimitiveFactory.from_text(f"concept {i}"))
            primitives.append(CognitivePrimitiveFactory.from_numerical([i * 0.1, (i+1) * 0.1]))
        
        # Process with architecture
        encoder = HypergraphEncoder()
        fragment = encoder.create_fragment_from_primitives(primitives, "scalability_test")
        
        verifier = CognitiveVerifier()
        collection_result = verifier.verify_primitive_collection(primitives)
        
        # Verify scalability
        assert len(primitives) == 20
        assert fragment.tensor_data.size > 0
        assert collection_result.score > 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])