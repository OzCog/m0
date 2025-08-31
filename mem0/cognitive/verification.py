"""
Verification & Visualization Module

Implements verification mechanisms for hypergraph consistency and
visualization tools for cognitive primitives and tensor fragments.

This module provides validation, testing, and visualization capabilities
for the cognitive architecture components.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .primitives import CognitivePrimitive, TensorSignature, ModalityType
from .tensors import TensorFragment, HypergraphEncoder
from .schemes import SchemeGrammar, CognitiveOperation


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    score: float
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class Validator(ABC):
    """Abstract base class for validators"""
    
    @abstractmethod
    def validate(self, target: Any) -> ValidationResult:
        """Validate the target object"""
        pass


class CognitiveVerifier:
    """
    Verifier for cognitive primitives and their consistency.
    
    Validates tensor signatures, primitive relationships, and
    cognitive operation coherence.
    """
    
    def __init__(self):
        self.primitive_validators = [
            SignatureValidator(),
            ContentValidator(),
            RelationshipValidator()
        ]
        self.operation_validators = [
            OperationConsistencyValidator(),
            CompositionValidator()
        ]
    
    def verify_primitive(self, primitive: CognitivePrimitive) -> ValidationResult:
        """Verify a single cognitive primitive"""
        errors = []
        warnings = []
        scores = []
        
        for validator in self.primitive_validators:
            result = validator.validate(primitive)
            scores.append(result.score)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        overall_score = np.mean(scores) if scores else 0.0
        is_valid = len(errors) == 0 and overall_score >= 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            errors=errors,
            warnings=warnings,
            metadata={
                "primitive_id": primitive.id,
                "validator_scores": dict(zip([v.__class__.__name__ for v in self.primitive_validators], scores))
            }
        )
    
    def verify_operation(self, operation: CognitiveOperation) -> ValidationResult:
        """Verify a cognitive operation"""
        errors = []
        warnings = []
        scores = []
        
        for validator in self.operation_validators:
            result = validator.validate(operation)
            scores.append(result.score)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        overall_score = np.mean(scores) if scores else 0.0
        is_valid = len(errors) == 0 and overall_score >= 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            errors=errors,
            warnings=warnings,
            metadata={
                "operation": operation.operator,
                "operand_count": len(operation.operands),
                "validator_scores": dict(zip([v.__class__.__name__ for v in self.operation_validators], scores))
            }
        )
    
    def verify_primitive_collection(self, primitives: List[CognitivePrimitive]) -> ValidationResult:
        """Verify a collection of primitives for consistency"""
        errors = []
        warnings = []
        individual_scores = []
        
        # Verify each primitive individually
        for primitive in primitives:
            result = self.verify_primitive(primitive)
            individual_scores.append(result.score)
            if not result.is_valid:
                errors.extend([f"Primitive {primitive.id}: {error}" for error in result.errors])
            warnings.extend([f"Primitive {primitive.id}: {warning}" for warning in result.warnings])
        
        # Check collection-level consistency
        collection_score = self._validate_collection_consistency(primitives)
        
        if collection_score < 0.5:
            warnings.append("Collection has low consistency score")
        
        overall_score = (np.mean(individual_scores) + collection_score) / 2 if individual_scores else collection_score
        is_valid = len(errors) == 0 and overall_score >= 0.6
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            errors=errors,
            warnings=warnings,
            metadata={
                "primitive_count": len(primitives),
                "individual_scores": individual_scores,
                "collection_consistency": collection_score
            }
        )
    
    def _validate_collection_consistency(self, primitives: List[CognitivePrimitive]) -> float:
        """Validate consistency within a collection of primitives"""
        if len(primitives) < 2:
            return 1.0
        
        compatibility_scores = []
        for i, p1 in enumerate(primitives):
            for j, p2 in enumerate(primitives):
                if i < j:
                    compatibility = p1.calculate_compatibility(p2)
                    compatibility_scores.append(compatibility)
        
        return np.mean(compatibility_scores) if compatibility_scores else 0.0


class SignatureValidator(Validator):
    """Validates tensor signatures"""
    
    def validate(self, primitive: CognitivePrimitive) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        signature = primitive.signature
        
        # Check signature constraints
        if signature.depth < 0:
            errors.append("Depth cannot be negative")
            score -= 0.3
        
        if not 0.0 <= signature.context <= 1.0:
            errors.append("Context must be between 0.0 and 1.0")
            score -= 0.3
        
        if not 0.0 <= signature.salience <= 1.0:
            errors.append("Salience must be between 0.0 and 1.0")
            score -= 0.3
        
        if not 0.0 <= signature.autonomy_index <= 1.0:
            errors.append("Autonomy index must be between 0.0 and 1.0")
            score -= 0.3
        
        # Check for reasonable values
        if signature.depth > 10:
            warnings.append("Depth is unusually high (>10)")
            score -= 0.1
        
        if signature.salience < 0.1:
            warnings.append("Salience is very low (<0.1)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "SignatureValidator"}
        )


class ContentValidator(Validator):
    """Validates primitive content"""
    
    def validate(self, primitive: CognitivePrimitive) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        content = primitive.content
        
        # Check content is not None
        if content is None:
            errors.append("Content cannot be None")
            score -= 0.5
        
        # Check content matches modality
        modality = primitive.signature.modality
        
        if modality == ModalityType.NUMERICAL:
            if not isinstance(content, (int, float, list, np.ndarray)):
                warnings.append("Numerical modality expects numeric content")
                score -= 0.2
        
        elif modality == ModalityType.TEXTUAL:
            if not isinstance(content, str):
                warnings.append("Textual modality expects string content")
                score -= 0.2
        
        elif modality == ModalityType.RELATIONAL:
            if not isinstance(content, dict) or "source" not in content or "target" not in content:
                errors.append("Relational modality requires dict with 'source' and 'target'")
                score -= 0.4
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "ContentValidator"}
        )


class RelationshipValidator(Validator):
    """Validates primitive relationships"""
    
    def validate(self, primitive: CognitivePrimitive) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        relationships = primitive.relationships
        
        # Check for self-references
        if primitive.id in relationships:
            warnings.append("Primitive references itself")
            score -= 0.1
        
        # Check for duplicate relationships
        if len(relationships) != len(set(relationships)):
            warnings.append("Duplicate relationships found")
            score -= 0.1
        
        # Check relationship consistency with content
        if primitive.signature.modality == ModalityType.RELATIONAL:
            if isinstance(primitive.content, dict):
                expected_relationships = [primitive.content.get("source"), primitive.content.get("target")]
                if not all(rel in relationships for rel in expected_relationships if rel):
                    warnings.append("Relationships don't match relational content")
                    score -= 0.2
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "RelationshipValidator"}
        )


class OperationConsistencyValidator(Validator):
    """Validates consistency of cognitive operations"""
    
    def validate(self, operation: CognitiveOperation) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        # Check operation has operands
        if not operation.operands:
            errors.append("Operation has no operands")
            score -= 0.5
        
        # Check operand count for specific operations
        if operation.operator == "relate" and len(operation.operands) != 2:
            errors.append("Relate operation requires exactly 2 operands")
            score -= 0.4
        
        if operation.operator == "specialize" and len(operation.operands) != 2:
            errors.append("Specialize operation requires exactly 2 operands")
            score -= 0.4
        
        if operation.operator in ["compose", "abstract"] and len(operation.operands) < 2:
            errors.append(f"{operation.operator} operation requires at least 2 operands")
            score -= 0.4
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "OperationConsistencyValidator"}
        )


class CompositionValidator(Validator):
    """Validates composition operations"""
    
    def validate(self, operation: CognitiveOperation) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        if operation.operator != "compose":
            return ValidationResult(True, 1.0, [], [], {"validator": "CompositionValidator", "skipped": True})
        
        # Check operand compatibility for composition
        # This would need access to the actual primitives, so we'll do basic checks
        
        if len(operation.operands) > 10:
            warnings.append("Composition has many operands (>10), may be complex")
            score -= 0.1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "CompositionValidator"}
        )


class HypergraphValidator:
    """
    Validator for hypergraph structures and consistency.
    
    Validates hypergraph patterns, node connections, and
    AtomSpace representation integrity.
    """
    
    def __init__(self):
        self.fragment_validators = [
            FragmentStructureValidator(),
            HypergraphConsistencyValidator(),
            TensorIntegrityValidator()
        ]
    
    def validate_fragment(self, fragment: TensorFragment) -> ValidationResult:
        """Validate a tensor fragment"""
        errors = []
        warnings = []
        scores = []
        
        for validator in self.fragment_validators:
            result = validator.validate(fragment)
            scores.append(result.score)
            errors.extend(result.errors)
            warnings.extend(result.warnings)
        
        overall_score = np.mean(scores) if scores else 0.0
        is_valid = len(errors) == 0 and overall_score >= 0.7
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            errors=errors,
            warnings=warnings,
            metadata={
                "fragment_id": fragment.id,
                "validator_scores": dict(zip([v.__class__.__name__ for v in self.fragment_validators], scores))
            }
        )
    
    def validate_hypergraph_pattern(self, pattern: Dict[str, Any]) -> ValidationResult:
        """Validate a hypergraph pattern"""
        errors = []
        warnings = []
        score = 1.0
        
        # Check required fields
        required_fields = ["type", "structure"]
        for field in required_fields:
            if field not in pattern:
                errors.append(f"Missing required field: {field}")
                score -= 0.3
        
        # Check pattern type
        valid_types = ["ConceptNode", "InheritanceLink", "SimilarityLink", "PatternNode"]
        if pattern.get("type") not in valid_types:
            warnings.append(f"Unknown pattern type: {pattern.get('type')}")
            score -= 0.1
        
        # Check structure consistency
        structure = pattern.get("structure", {})
        if not isinstance(structure, dict):
            errors.append("Structure must be a dictionary")
            score -= 0.3
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "HypergraphValidator"}
        )
    
    def validate_atomspace_graph(self, atomspace_graph: Dict[str, Any]) -> ValidationResult:
        """Validate complete AtomSpace graph representation"""
        errors = []
        warnings = []
        score = 1.0
        
        # Check required fields
        required_fields = ["nodes", "edges", "patterns"]
        for field in required_fields:
            if field not in atomspace_graph:
                errors.append(f"Missing required field: {field}")
                score -= 0.3
        
        # Validate nodes
        nodes = atomspace_graph.get("nodes", [])
        if not isinstance(nodes, list):
            errors.append("Nodes must be a list")
            score -= 0.3
        else:
            for i, node in enumerate(nodes):
                if not isinstance(node, dict) or "id" not in node:
                    errors.append(f"Node {i} is invalid (must be dict with 'id')")
                    score -= 0.1
        
        # Validate edges
        edges = atomspace_graph.get("edges", [])
        if not isinstance(edges, list):
            errors.append("Edges must be a list")
            score -= 0.3
        else:
            node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
            for i, edge in enumerate(edges):
                if not isinstance(edge, dict):
                    errors.append(f"Edge {i} is invalid (must be dict)")
                    score -= 0.1
                else:
                    source = edge.get("source")
                    target = edge.get("target")
                    if source not in node_ids or target not in node_ids:
                        warnings.append(f"Edge {i} references unknown nodes")
                        score -= 0.05
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={
                "validator": "HypergraphValidator",
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
        )


class FragmentStructureValidator(Validator):
    """Validates tensor fragment structure"""
    
    def validate(self, fragment: TensorFragment) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        # Check tensor data
        if fragment.tensor_data.size == 0:
            errors.append("Tensor data is empty")
            score -= 0.5
        
        # Check hypergraph structure
        required_fields = {"nodes", "edges", "patterns"}
        structure_fields = set(fragment.hypergraph_structure.keys())
        if not required_fields.issubset(structure_fields):
            missing = required_fields - structure_fields
            errors.append(f"Hypergraph structure missing fields: {missing}")
            score -= 0.3
        
        # Check primitive consistency
        if len(fragment.primitives) == 0:
            warnings.append("Fragment has no primitives")
            score -= 0.1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "FragmentStructureValidator"}
        )


class HypergraphConsistencyValidator(Validator):
    """Validates hypergraph consistency"""
    
    def validate(self, fragment: TensorFragment) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        # Check node-edge consistency
        nodes = fragment.hypergraph_structure.get("nodes", [])
        edges = fragment.hypergraph_structure.get("edges", [])
        
        node_ids = {node.get("id") for node in nodes if isinstance(node, dict) and "id" in node}
        
        for edge in edges:
            if isinstance(edge, dict):
                source = edge.get("source")
                target = edge.get("target")
                
                if source and source not in node_ids:
                    warnings.append(f"Edge references unknown source node: {source}")
                    score -= 0.1
                
                if target and target not in node_ids:
                    warnings.append(f"Edge references unknown target node: {target}")
                    score -= 0.1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={"validator": "HypergraphConsistencyValidator"}
        )


class TensorIntegrityValidator(Validator):
    """Validates tensor data integrity"""
    
    def validate(self, fragment: TensorFragment) -> ValidationResult:
        errors = []
        warnings = []
        score = 1.0
        
        tensor_data = fragment.tensor_data
        
        # Check for NaN or infinite values
        if np.isnan(tensor_data).any():
            errors.append("Tensor contains NaN values")
            score -= 0.4
        
        if np.isinf(tensor_data).any():
            errors.append("Tensor contains infinite values")
            score -= 0.4
        
        # Check tensor magnitude
        magnitude = np.linalg.norm(tensor_data)
        if magnitude == 0:
            warnings.append("Tensor has zero magnitude")
            score -= 0.1
        elif magnitude > 1000:
            warnings.append("Tensor has very large magnitude")
            score -= 0.1
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=max(0.0, score),
            errors=errors,
            warnings=warnings,
            metadata={
                "validator": "TensorIntegrityValidator",
                "tensor_shape": tensor_data.shape,
                "tensor_magnitude": magnitude
            }
        )


class CognitiveVisualizer:
    """
    Visualization tools for cognitive primitives and tensor fragments.
    
    Provides text-based and structured visualizations of cognitive
    architecture components for debugging and analysis.
    """
    
    def visualize_primitive(self, primitive: CognitivePrimitive) -> str:
        """Create text visualization of cognitive primitive"""
        lines = [
            f"Cognitive Primitive: {primitive.id}",
            "=" * (len(primitive.id) + 20),
            f"Modality: {primitive.signature.modality.value}",
            f"Depth: {primitive.signature.depth}",
            f"Context: {primitive.signature.context:.3f}",
            f"Salience: {primitive.signature.salience:.3f}",
            f"Autonomy: {primitive.signature.autonomy_index:.3f}",
            "",
            "Content:",
            f"  {self._format_content(primitive.content)}",
            "",
            f"Relationships: {', '.join(primitive.relationships) if primitive.relationships else 'None'}",
            f"Metadata: {primitive.metadata if primitive.metadata else 'None'}"
        ]
        
        return "\n".join(lines)
    
    def visualize_fragment(self, fragment: TensorFragment) -> str:
        """Create text visualization of tensor fragment"""
        lines = [
            f"Tensor Fragment: {fragment.id}",
            "=" * (len(fragment.id) + 17),
            f"Signature: {fragment.signature.modality.value} (depth={fragment.signature.depth})",
            f"Tensor Shape: {fragment.tensor_data.shape}",
            f"Tensor Magnitude: {np.linalg.norm(fragment.tensor_data):.3f}",
            f"Primitives: {len(fragment.primitives)}",
            f"Connections: {len(fragment.connections)}",
            "",
            "Hypergraph Structure:",
            f"  Nodes: {len(fragment.hypergraph_structure.get('nodes', []))}",
            f"  Edges: {len(fragment.hypergraph_structure.get('edges', []))}",
            f"  Patterns: {len(fragment.hypergraph_structure.get('patterns', {}))}",
            ""
        ]
        
        # Add primitive summaries
        if fragment.primitives:
            lines.append("Primitives:")
            for primitive in fragment.primitives[:5]:  # Show first 5
                lines.append(f"  - {primitive.id} ({primitive.signature.modality.value})")
            if len(fragment.primitives) > 5:
                lines.append(f"  ... and {len(fragment.primitives) - 5} more")
        
        # Add connections
        if fragment.connections:
            lines.append("")
            lines.append("Top Connections:")
            sorted_connections = sorted(fragment.connections.items(), key=lambda x: x[1], reverse=True)
            for target_id, strength in sorted_connections[:3]:
                lines.append(f"  - {target_id}: {strength:.3f}")
        
        return "\n".join(lines)
    
    def visualize_hypergraph_summary(self, hypergraph: Dict[str, Any]) -> str:
        """Create summary visualization of hypergraph"""
        lines = [
            "Hypergraph Summary",
            "=" * 18,
            f"Nodes: {len(hypergraph.get('nodes', []))}",
            f"Edges: {len(hypergraph.get('edges', []))}",
            f"Patterns: {len(hypergraph.get('patterns', []))}",
            ""
        ]
        
        # Node type distribution
        nodes = hypergraph.get("nodes", [])
        node_types = {}
        for node in nodes:
            if isinstance(node, dict):
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1
        
        if node_types:
            lines.append("Node Types:")
            for node_type, count in node_types.items():
                lines.append(f"  {node_type}: {count}")
            lines.append("")
        
        # Edge type distribution
        edges = hypergraph.get("edges", [])
        edge_types = {}
        for edge in edges:
            if isinstance(edge, dict):
                edge_type = edge.get("type", "unknown")
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        if edge_types:
            lines.append("Edge Types:")
            for edge_type, count in edge_types.items():
                lines.append(f"  {edge_type}: {count}")
        
        return "\n".join(lines)
    
    def create_validation_report(self, validation_result: ValidationResult) -> str:
        """Create formatted validation report"""
        lines = [
            "Validation Report",
            "=" * 17,
            f"Status: {'VALID' if validation_result.is_valid else 'INVALID'}",
            f"Score: {validation_result.score:.3f}/1.0",
            ""
        ]
        
        if validation_result.errors:
            lines.append("Errors:")
            for error in validation_result.errors:
                lines.append(f"  ❌ {error}")
            lines.append("")
        
        if validation_result.warnings:
            lines.append("Warnings:")
            for warning in validation_result.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")
        
        if validation_result.metadata:
            lines.append("Metadata:")
            for key, value in validation_result.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_content(self, content: Any) -> str:
        """Format content for display"""
        if isinstance(content, str):
            return f'"{content}"' if len(content) < 50 else f'"{content[:47]}..."'
        elif isinstance(content, (int, float)):
            return str(content)
        elif isinstance(content, np.ndarray):
            return f"Array{content.shape} (magnitude: {np.linalg.norm(content):.3f})"
        elif isinstance(content, dict):
            return f"Dict with {len(content)} keys: {list(content.keys())[:3]}"
        elif isinstance(content, list):
            return f"List with {len(content)} items"
        else:
            return str(type(content).__name__)


# Example usage functions for testing and demonstration
def create_test_primitive() -> CognitivePrimitive:
    """Create a test cognitive primitive for demonstration"""
    from .primitives import CognitivePrimitiveFactory
    return CognitivePrimitiveFactory.from_text("This is a test cognitive primitive", salience=0.8, context=0.7)


def create_test_fragment() -> TensorFragment:
    """Create a test tensor fragment for demonstration"""
    from .primitives import CognitivePrimitiveFactory
    
    primitives = [
        CognitivePrimitiveFactory.from_text("Test primitive 1"),
        CognitivePrimitiveFactory.from_numerical([1.0, 2.0, 3.0]),
        CognitivePrimitiveFactory.from_relation("concept1", "relates_to", "concept2")
    ]
    
    encoder = HypergraphEncoder()
    return encoder.create_fragment_from_primitives(primitives, "test_fragment")


def run_verification_demo():
    """Run a demonstration of the verification system"""
    print("=== Cognitive Verification Demo ===\n")
    
    # Create test objects
    primitive = create_test_primitive()
    fragment = create_test_fragment()
    
    # Initialize verifiers
    cognitive_verifier = CognitiveVerifier()
    hypergraph_validator = HypergraphValidator()
    visualizer = CognitiveVisualizer()
    
    # Verify primitive
    print("Verifying Cognitive Primitive:")
    primitive_result = cognitive_verifier.verify_primitive(primitive)
    print(visualizer.create_validation_report(primitive_result))
    print()
    
    # Verify fragment
    print("Verifying Tensor Fragment:")
    fragment_result = hypergraph_validator.validate_fragment(fragment)
    print(visualizer.create_validation_report(fragment_result))
    print()
    
    # Show visualizations
    print("Primitive Visualization:")
    print(visualizer.visualize_primitive(primitive))
    print("\n" + "="*50 + "\n")
    
    print("Fragment Visualization:")
    print(visualizer.visualize_fragment(fragment))
    print()


if __name__ == "__main__":
    run_verification_demo()