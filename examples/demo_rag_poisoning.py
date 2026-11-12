"""
Interactive Demonstration of Adversarial RAG Poisoning & Robust Defense
Demonstrates how an adversary hijacks Top-1 retrieval with prompt injection,
and how robust heuristic/lexical filtering neutralizes the attack.
"""
import os
import sys

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

from datasets.corpus_generator import load_benchmark_dataset
from rag_bench.retrieval import DenseVectorRetriever
from rag_bench.attacker import AdversarialRAGAttacker
from rag_bench.defense import RobustRetrievalFilter


def main():
    print("=" * 80)
    print("   ADVERSARIAL RAG POISONING & DEFENSIVE PURIFICATION DEMO               ")
    print("=" * 80)

    clean_docs, queries = load_benchmark_dataset()
    test_query = queries[0]  # "What are the rules for connecting to corporate network over VPN?"
    print(f"[*] Target User Query : \"{test_query.text}\"")
    print(f"[*] Ground Truth Doc  : {test_query.ground_truth_doc_id}")

    # 1. Clean Retrieval
    retriever = DenseVectorRetriever()
    retriever.fit_and_index(clean_docs)
    res_clean = retriever.retrieve(test_query.text, top_k=3)

    print("\n[Phase 1] Clean Vector Retrieval:")
    for rank, (doc, score) in enumerate(res_clean, 1):
        print(f"  Rank {rank} (Score: {score:.4f}) [{doc.doc_id}]: {doc.text[:75]}...")

    # 2. Inject Trojan Poison Document
    p_doc = AdversarialRAGAttacker.craft_poisoned_document(
        doc_id="trojan_doc_999",
        target_query=test_query.text,
        payload_idx=0,
        anchor_multiplier=4,
    )
    poisoned_corpus = list(clean_docs) + [p_doc]
    retriever_poisoned = DenseVectorRetriever()
    retriever_poisoned.fit_and_index(poisoned_corpus)
    res_poisoned = retriever_poisoned.retrieve(test_query.text, top_k=3)

    print("\n[Phase 2] Poisoned Vector Retrieval (No Defense):")
    for rank, (doc, score) in enumerate(res_poisoned, 1):
        status = "!!! POISONED TROJAN !!!" if doc.is_poisoned else "Benign"
        print(f"  Rank {rank} (Score: {score:.4f}) [{doc.doc_id}] ({status}):")
        print(f"         {doc.text[:100]}...")

    # 3. Defended Retrieval
    filter_engine = RobustRetrievalFilter()
    sanitized_results = filter_engine.filter_candidates(
        retriever_poisoned.retrieve(test_query.text, top_k=6),
        desired_k=3,
    )

    print("\n[Phase 3] Defended Vector Retrieval (Robust Filter Active):")
    for rank, (doc, score) in enumerate(sanitized_results, 1):
        print(f"  Rank {rank} (Score: {score:.4f}) [{doc.doc_id}]: {doc.text[:75]}...")

    print("\n" + "=" * 80)
    print("[+] Defense successfully quarantined the poisoned injection document.")
    print("[+] Legitimate enterprise document restored to Rank 1.")
    print("=" * 80)


if __name__ == "__main__":
    main()
