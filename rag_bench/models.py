"""
RAG Benchmarking Data Contracts and Models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_poisoned: bool = False
    adversarial_payload: Optional[str] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class Query:
    query_id: str
    text: str
    ground_truth_doc_id: str
    target_topic: str


@dataclass
class RetrievalItem:
    doc_id: str
    score: float
    is_poisoned: bool
    text: str


@dataclass
class RetrievalResult:
    query_id: str
    items: List[RetrievalItem]
    poison_retrieved_in_top_k: bool
    ground_truth_rank: Optional[int]


@dataclass
class RobustnessMetrics:
    hit_rate_at_k: float
    mrr_at_k: float
    attack_success_rate: float
    mean_poison_rank: Optional[float]
    top_k: int
