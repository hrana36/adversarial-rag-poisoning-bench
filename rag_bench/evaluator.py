"""
RAG Robustness Benchmark Evaluator
Computes HitRate@K, MRR@K, Attack Success Rate (ASR), and Retrieval Degradation.
"""
from typing import List, Dict, Any, Tuple, Optional
from rag_bench.models import Document, Query, RobustnessMetrics
from rag_bench.retrieval import DenseVectorRetriever
from rag_bench.defense import RobustRetrievalFilter


class RAGRobustnessEvaluator:
    """
    Evaluates retrieval robustness under clean, poisoned, and defended scenarios.
    """

    def __init__(self, retriever: DenseVectorRetriever, defense: Optional[RobustRetrievalFilter] = None):
        self.retriever = retriever
        self.defense = defense

    def evaluate_queries(
        self,
        queries: List[Query],
        top_k: int = 5,
        apply_defense: bool = False,
    ) -> RobustnessMetrics:
        """
        Runs evaluation over all test queries.
        """
        if not queries:
            raise ValueError("Query set cannot be empty.")

        hits = 0
        reciprocal_ranks = []
        poison_retrieved_count = 0
        poison_ranks = []

        for q in queries:
            # Retrieve candidates (fetch slightly more if defense is applied to allow backfilling)
            fetch_k = top_k * 3 if apply_defense and self.defense else top_k
            candidates = self.retriever.retrieve(q.text, top_k=fetch_k)

            if apply_defense and self.defense:
                candidates = self.defense.filter_candidates(candidates, desired_k=top_k)
            else:
                candidates = candidates[:top_k]

            # 1. Check ground truth hit and rank
            gt_rank: Optional[int] = None
            for idx, (doc, _) in enumerate(candidates):
                if doc.doc_id == q.ground_truth_doc_id:
                    gt_rank = idx + 1
                    break

            if gt_rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / gt_rank)
            else:
                reciprocal_ranks.append(0.0)

            # 2. Check if poisoned document hijacked Top-K
            p_rank: Optional[int] = None
            for idx, (doc, _) in enumerate(candidates):
                if doc.is_poisoned:
                    p_rank = idx + 1
                    break

            if p_rank is not None:
                poison_retrieved_count += 1
                poison_ranks.append(p_rank)

        hit_rate = hits / len(queries)
        mrr = sum(reciprocal_ranks) / len(queries)
        asr = poison_retrieved_count / len(queries)
        mean_p_rank = sum(poison_ranks) / len(poison_ranks) if poison_ranks else None

        return RobustnessMetrics(
            hit_rate_at_k=round(hit_rate, 4),
            mrr_at_k=round(mrr, 4),
            attack_success_rate=round(asr, 4),
            mean_poison_rank=round(mean_p_rank, 2) if mean_p_rank else None,
            top_k=top_k,
        )
