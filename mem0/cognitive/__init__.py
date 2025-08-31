"""
Cognitive Primitives & Foundational Hypergraph Encoding

This module implements Phase 1 of the cognitive architecture:
- Scheme Cognitive Grammar Microservices
- Tensor Fragment Architecture  
- Verification & Visualization

Tensor signature: [modality, depth, context, salience, autonomy_index]
"""

from .primitives import CognitivePrimitive, TensorSignature, ModalityType, CognitivePrimitiveFactory
from .schemes import SchemeGrammar, CognitiveGrammarService
from .tensors import TensorFragment, HypergraphEncoder
from .verification import CognitiveVerifier, HypergraphValidator

__all__ = [
    "CognitivePrimitive",
    "TensorSignature",
    "ModalityType",
    "CognitivePrimitiveFactory",
    "SchemeGrammar",
    "CognitiveGrammarService",
    "TensorFragment",
    "HypergraphEncoder",
    "CognitiveVerifier",
    "HypergraphValidator",
]