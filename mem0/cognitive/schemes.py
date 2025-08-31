"""
Scheme Cognitive Grammar Microservices

Implements Scheme-like grammar structures for cognitive operations and
microservice interfaces for cognitive processing.

This module provides the grammatical framework for composing cognitive primitives
into more complex cognitive operations following Scheme-like functional paradigms.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

from .primitives import CognitivePrimitive, TensorSignature, ModalityType


class CognitiveExpression(ABC):
    """Abstract base class for cognitive expressions in the grammar"""
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate the cognitive expression in the given context"""
        pass
    
    @abstractmethod
    def to_scheme(self) -> str:
        """Convert expression to Scheme-like syntax"""
        pass


@dataclass
class CognitiveAtom(CognitiveExpression):
    """Atomic cognitive expression wrapping a cognitive primitive"""
    primitive: CognitivePrimitive
    
    def evaluate(self, context: Dict[str, Any]) -> CognitivePrimitive:
        """Return the wrapped primitive"""
        return self.primitive
    
    def to_scheme(self) -> str:
        """Convert to Scheme atom representation"""
        return f"(atom {self.primitive.id})"


@dataclass
class CognitiveOperation(CognitiveExpression):
    """Operation that combines cognitive expressions"""
    operator: str
    operands: List[CognitiveExpression]
    
    def evaluate(self, context: Dict[str, Any]) -> Any:
        """Evaluate operation by applying operator to evaluated operands"""
        evaluated_operands = [op.evaluate(context) for op in self.operands]
        
        if self.operator == "compose":
            return self._compose_primitives(evaluated_operands)
        elif self.operator == "relate":
            return self._relate_primitives(evaluated_operands)
        elif self.operator == "abstract":
            return self._abstract_primitives(evaluated_operands)
        elif self.operator == "specialize":
            return self._specialize_primitive(evaluated_operands)
        else:
            raise ValueError(f"Unknown operator: {self.operator}")
    
    def to_scheme(self) -> str:
        """Convert to Scheme operation representation"""
        operand_schemes = " ".join([op.to_scheme() for op in self.operands])
        return f"({self.operator} {operand_schemes})"
    
    def _compose_primitives(self, primitives: List[CognitivePrimitive]) -> CognitivePrimitive:
        """Compose multiple primitives into a new composite primitive"""
        if not primitives:
            raise ValueError("Cannot compose empty list of primitives")
        
        # Calculate composite signature
        avg_context = sum(p.signature.context for p in primitives) / len(primitives)
        avg_salience = sum(p.signature.salience for p in primitives) / len(primitives)
        avg_autonomy = sum(p.signature.autonomy_index for p in primitives) / len(primitives)
        max_depth = max(p.signature.depth for p in primitives) + 1
        
        # Use modality of most salient primitive
        primary_primitive = max(primitives, key=lambda p: p.signature.salience)
        
        composite_signature = TensorSignature(
            modality=primary_primitive.signature.modality,
            depth=max_depth,
            context=avg_context,
            salience=avg_salience,
            autonomy_index=avg_autonomy
        )
        
        return CognitivePrimitive(
            id=f"composite_{hash('_'.join(p.id for p in primitives)) % 10000}",
            signature=composite_signature,
            content={
                "type": "composition",
                "components": [p.id for p in primitives]
            },
            metadata={
                "operation": "compose",
                "component_count": len(primitives)
            },
            relationships=[p.id for p in primitives]
        )
    
    def _relate_primitives(self, primitives: List[CognitivePrimitive]) -> CognitivePrimitive:
        """Create relational primitive connecting given primitives"""
        if len(primitives) != 2:
            raise ValueError("Relate operation requires exactly 2 primitives")
        
        source, target = primitives
        
        # Calculate relational signature
        relation_signature = TensorSignature(
            modality=ModalityType.RELATIONAL,
            depth=max(source.signature.depth, target.signature.depth) + 1,
            context=(source.signature.context + target.signature.context) / 2,
            salience=(source.signature.salience + target.signature.salience) / 2,
            autonomy_index=0.8  # Relations have high autonomy
        )
        
        return CognitivePrimitive(
            id=f"relation_{source.id}_{target.id}",
            signature=relation_signature,
            content={
                "type": "relation",
                "source": source.id,
                "target": target.id,
                "strength": source.calculate_compatibility(target)
            },
            relationships=[source.id, target.id]
        )
    
    def _abstract_primitives(self, primitives: List[CognitivePrimitive]) -> CognitivePrimitive:
        """Create abstract primitive from concrete primitives"""
        if not primitives:
            raise ValueError("Cannot abstract empty list of primitives")
        
        # Calculate abstraction signature
        avg_context = sum(p.signature.context for p in primitives) / len(primitives)
        avg_salience = sum(p.signature.salience for p in primitives) / len(primitives)
        max_depth = max(p.signature.depth for p in primitives) + 1
        
        abstract_signature = TensorSignature(
            modality=ModalityType.SYMBOLIC,  # Abstractions are symbolic
            depth=max_depth,
            context=avg_context * 0.8,  # Abstractions have reduced context specificity
            salience=avg_salience,
            autonomy_index=0.9  # Abstractions have high autonomy
        )
        
        return CognitivePrimitive(
            id=f"abstract_{hash('_'.join(p.id for p in primitives)) % 10000}",
            signature=abstract_signature,
            content={
                "type": "abstraction",
                "instances": [p.id for p in primitives],
                "common_features": self._extract_common_features(primitives)
            },
            metadata={
                "operation": "abstract",
                "instance_count": len(primitives)
            }
        )
    
    def _specialize_primitive(self, primitives: List[CognitivePrimitive]) -> CognitivePrimitive:
        """Specialize an abstract primitive with concrete details"""
        if len(primitives) != 2:
            raise ValueError("Specialize operation requires exactly 2 primitives (abstract + concrete)")
        
        abstract_primitive, concrete_primitive = primitives
        
        specialized_signature = TensorSignature(
            modality=concrete_primitive.signature.modality,
            depth=abstract_primitive.signature.depth,
            context=concrete_primitive.signature.context,
            salience=(abstract_primitive.signature.salience + concrete_primitive.signature.salience) / 2,
            autonomy_index=concrete_primitive.signature.autonomy_index
        )
        
        return CognitivePrimitive(
            id=f"specialized_{abstract_primitive.id}_{concrete_primitive.id}",
            signature=specialized_signature,
            content={
                "type": "specialization",
                "abstract": abstract_primitive.id,
                "concrete": concrete_primitive.id,
                "specialized_content": concrete_primitive.content
            },
            relationships=[abstract_primitive.id, concrete_primitive.id]
        )
    
    def _extract_common_features(self, primitives: List[CognitivePrimitive]) -> Dict[str, Any]:
        """Extract common features from a list of primitives for abstraction"""
        if not primitives:
            return {}
        
        # Find common modalities
        modalities = [p.signature.modality for p in primitives]
        common_modality = modalities[0] if all(m == modalities[0] for m in modalities) else None
        
        # Calculate feature statistics
        contexts = [p.signature.context for p in primitives]
        saliences = [p.signature.salience for p in primitives]
        autonomies = [p.signature.autonomy_index for p in primitives]
        
        return {
            "common_modality": common_modality.value if common_modality else None,
            "context_range": [min(contexts), max(contexts)],
            "salience_range": [min(saliences), max(saliences)],
            "autonomy_range": [min(autonomies), max(autonomies)],
            "primitive_count": len(primitives)
        }


class SchemeGrammar:
    """
    Scheme-like grammar parser and evaluator for cognitive expressions.
    
    Supports basic Scheme syntax for cognitive operations:
    - (atom primitive_id)
    - (compose expr1 expr2 ...)
    - (relate expr1 expr2)
    - (abstract expr1 expr2 ...)
    - (specialize abstract_expr concrete_expr)
    """
    
    def __init__(self):
        self.primitives: Dict[str, CognitivePrimitive] = {}
    
    def register_primitive(self, primitive: CognitivePrimitive):
        """Register a cognitive primitive for use in expressions"""
        self.primitives[primitive.id] = primitive
    
    def parse(self, expression: str) -> CognitiveExpression:
        """Parse a Scheme-like expression into a cognitive expression"""
        tokens = self._tokenize(expression)
        return self._parse_expression(tokens)
    
    def evaluate(self, expression: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """Parse and evaluate a Scheme-like expression"""
        if context is None:
            context = {}
        
        parsed_expr = self.parse(expression)
        return parsed_expr.evaluate(context)
    
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize a Scheme-like expression"""
        # Simple tokenizer - replace with more robust parser if needed
        expression = expression.replace('(', ' ( ').replace(')', ' ) ')
        return [token for token in expression.split() if token]
    
    def _parse_expression(self, tokens: List[str]) -> CognitiveExpression:
        """Parse tokens into cognitive expression"""
        if not tokens:
            raise ValueError("Empty expression")
        
        if tokens[0] != '(':
            # Atomic expression
            primitive_id = tokens[0]
            if primitive_id not in self.primitives:
                raise ValueError(f"Unknown primitive: {primitive_id}")
            return CognitiveAtom(self.primitives[primitive_id])
        
        # Complex expression
        tokens = tokens[1:]  # Skip opening paren
        
        if not tokens:
            raise ValueError("Empty parenthesized expression")
        
        operator = tokens[0]
        tokens = tokens[1:]
        
        operands = []
        paren_count = 0
        current_expr = []
        
        for token in tokens:
            if token == ')' and paren_count == 0:
                if current_expr:
                    operands.append(self._parse_expression(current_expr))
                break
            elif token == '(':
                paren_count += 1
                current_expr.append(token)
            elif token == ')':
                paren_count -= 1
                current_expr.append(token)
            else:
                if paren_count == 0:
                    # Simple atom
                    if current_expr:
                        operands.append(self._parse_expression(current_expr))
                        current_expr = []
                    operands.append(self._parse_expression([token]))
                else:
                    current_expr.append(token)
        
        return CognitiveOperation(operator, operands)


class CognitiveGrammarService:
    """
    Microservice interface for cognitive grammar operations.
    
    Provides REST-like interface for cognitive primitive composition,
    relation extraction, and abstraction operations.
    """
    
    def __init__(self):
        self.grammar = SchemeGrammar()
        self.operation_handlers: Dict[str, Callable] = {
            "compose": self._handle_compose,
            "relate": self._handle_relate,
            "abstract": self._handle_abstract,
            "specialize": self._handle_specialize,
            "evaluate": self._handle_evaluate
        }
    
    def register_primitive(self, primitive: CognitivePrimitive):
        """Register a cognitive primitive with the service"""
        self.grammar.register_primitive(primitive)
    
    def process_request(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a cognitive grammar operation request"""
        if operation not in self.operation_handlers:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": list(self.operation_handlers.keys())
            }
        
        try:
            result = self.operation_handlers[operation](payload)
            return {
                "status": "success",
                "operation": operation,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "operation": operation,
                "error": str(e)
            }
    
    def _handle_compose(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle composition operation"""
        primitive_ids = payload.get("primitive_ids", [])
        primitives = [self.grammar.primitives[pid] for pid in primitive_ids if pid in self.grammar.primitives]
        
        if not primitives:
            raise ValueError("No valid primitives provided for composition")
        
        operation = CognitiveOperation("compose", [CognitiveAtom(p) for p in primitives])
        result = operation.evaluate({})
        
        return {
            "composed_primitive": result.to_atomspace_pattern(),
            "scheme_expression": operation.to_scheme()
        }
    
    def _handle_relate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle relation operation"""
        source_id = payload.get("source_id")
        target_id = payload.get("target_id")
        
        if not source_id or not target_id:
            raise ValueError("Both source_id and target_id required for relation")
        
        if source_id not in self.grammar.primitives or target_id not in self.grammar.primitives:
            raise ValueError("Unknown primitive IDs")
        
        source = self.grammar.primitives[source_id]
        target = self.grammar.primitives[target_id]
        
        operation = CognitiveOperation("relate", [CognitiveAtom(source), CognitiveAtom(target)])
        result = operation.evaluate({})
        
        return {
            "relation_primitive": result.to_atomspace_pattern(),
            "scheme_expression": operation.to_scheme()
        }
    
    def _handle_abstract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle abstraction operation"""
        primitive_ids = payload.get("primitive_ids", [])
        primitives = [self.grammar.primitives[pid] for pid in primitive_ids if pid in self.grammar.primitives]
        
        if len(primitives) < 2:
            raise ValueError("At least 2 primitives required for abstraction")
        
        operation = CognitiveOperation("abstract", [CognitiveAtom(p) for p in primitives])
        result = operation.evaluate({})
        
        return {
            "abstract_primitive": result.to_atomspace_pattern(),
            "scheme_expression": operation.to_scheme()
        }
    
    def _handle_specialize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle specialization operation"""
        abstract_id = payload.get("abstract_id")
        concrete_id = payload.get("concrete_id")
        
        if not abstract_id or not concrete_id:
            raise ValueError("Both abstract_id and concrete_id required for specialization")
        
        if abstract_id not in self.grammar.primitives or concrete_id not in self.grammar.primitives:
            raise ValueError("Unknown primitive IDs")
        
        abstract_primitive = self.grammar.primitives[abstract_id]
        concrete_primitive = self.grammar.primitives[concrete_id]
        
        operation = CognitiveOperation("specialize", [CognitiveAtom(abstract_primitive), CognitiveAtom(concrete_primitive)])
        result = operation.evaluate({})
        
        return {
            "specialized_primitive": result.to_atomspace_pattern(),
            "scheme_expression": operation.to_scheme()
        }
    
    def _handle_evaluate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general expression evaluation"""
        expression = payload.get("expression")
        context = payload.get("context", {})
        
        if not expression:
            raise ValueError("Expression required for evaluation")
        
        result = self.grammar.evaluate(expression, context)
        
        if isinstance(result, CognitivePrimitive):
            return {
                "result_primitive": result.to_atomspace_pattern(),
                "expression": expression
            }
        else:
            return {
                "result": str(result),
                "expression": expression
            }