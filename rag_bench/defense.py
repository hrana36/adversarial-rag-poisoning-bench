"""
Robust Defensive Filtering for Retrieval-Augmented Generation
Detects and quarantines poisoned or Trojan documents from retrieved Top-K candidate pools.
"""
import re
from typing import List, Tuple, Dict, Any
from rag_bench.models import Document


class RobustRetrievalFilter:
    """
    Multi-stage pre-generation filtering pipeline for sanitizing dense retrieval results.
    """

    INJECTION_INDICATORS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s+override",
        r"reveal\s+master\s+credentials",
        r"bypass\s+zero-trust",
        r"direct\s+all\s+payments\s+to",
        r"\[topic\s+classifier:",
    ]

    def __init__(self, token_repetition_threshold: float = 0.40):
        self.repetition_threshold = token_repetition_threshold
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_INDICATORS]

    def check_heuristic_injection(self, text: str) -> bool:
        """Flags explicit prompt injection tokens and system override patterns."""
        for pattern in self.patterns:
            if pattern.search(text):
                return True
        return False

    def check_lexical_anomaly(self, text: str) -> bool:
        """
        Detects uncharacteristic token repetition or artificial anchor stuffing.
        """
        words = [w.lower() for w in text.split() if len(w) > 3]
        if not words:
            return False
        unique_ratio = len(set(words)) / len(words)
        # If words are repeated excessively (e.g. repeated query anchors), unique_ratio drops
        return unique_ratio < self.repetition_threshold

    def is_adversarial(self, doc: Document) -> Tuple[bool, str]:
        """
        Evaluates whether a candidate document displays adversarial poisoning indicators.
        """
        if self.check_heuristic_injection(doc.text):
            return True, "Triggered prompt injection heuristic"

        if self.check_lexical_anomaly(doc.text):
            return True, "Triggered abnormal lexical repetition / anchor stuffing"

        return False, "Clean"

    def filter_candidates(
        self,
        candidates: List[Tuple[Document, float]],
        desired_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Filters out flagged adversarial documents from the candidate pool and returns sanitized Top-K.
        """
        sanitized = []
        for doc, score in candidates:
            is_adv, reason = self.is_adversarial(doc)
            if not is_adv:
                sanitized.append((doc, score))
            if len(sanitized) >= desired_k:
                break
        return sanitized
