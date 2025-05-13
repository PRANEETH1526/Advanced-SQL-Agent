from typing import List
import math

def dcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes the Discounted Cumulative Gain (DCG) at rank k.

    DCG rewards relevant documents appearing earlier in the result list, using a logarithmic discount.

    Args:
        retrieved_ids (List[str]): List of document IDs retrieved by the system.
        relevant_ids (List[str]): List of ground-truth relevant document IDs.
        k (int): Rank at which to compute DCG.

    Returns:
        float: The DCG score at rank k.
    """
    return sum(1 / math.log2(i + 2) for i, doc_id in enumerate(retrieved_ids[:k]) if doc_id in relevant_ids)


def idcg_at_k(relevant_ids: List[str], k: int) -> float:
    """
    Computes the Ideal Discounted Cumulative Gain (IDCG) at rank k.

    IDCG represents the best possible DCG score if all relevant documents were ranked at the top.

    Args:
        relevant_ids (List[str]): List of ground-truth relevant document IDs.
        k (int): Rank at which to compute IDCG.

    Returns:
        float: The IDCG score at rank k.
    """
    ideal_hits = min(len(relevant_ids), k)
    return sum(1 / math.log2(i + 2) for i in range(ideal_hits))


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes the Normalized Discounted Cumulative Gain (nDCG) at rank k.

    nDCG measures the ranking quality of retrieved documents relative to an ideal ranking.

    Args:
        retrieved_ids (List[str]): List of document IDs retrieved by the system.
        relevant_ids (List[str]): List of ground-truth relevant document IDs.
        k (int): Rank at which to compute nDCG.

    Returns:
        float: The normalized DCG score between 0 and 1.
    """
    dcg = dcg_at_k(retrieved_ids, relevant_ids, k)
    idcg = idcg_at_k(relevant_ids, k)
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Precision at rank k.

    Precision@k measures the fraction of the top-k retrieved documents that are relevant.

    Args:
        retrieved_ids (List[str]): List of document IDs retrieved by the system.
        relevant_ids (List[str]): List of ground-truth relevant document IDs.
        k (int): Rank at which to compute precision.

    Returns:
        float: Precision score between 0 and 1.
    """
    return len([doc_id for doc_id in retrieved_ids[:k] if doc_id in relevant_ids]) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Recall at rank k.

    Recall@k measures the fraction of all relevant documents that were retrieved in the top-k.

    Args:
        retrieved_ids (List[str]): List of document IDs retrieved by the system.
        relevant_ids (List[str]): List of ground-truth relevant document IDs.
        k (int): Rank at which to compute recall.

    Returns:
        float: Recall score between 0 and 1.
    """
    return len([doc_id for doc_id in retrieved_ids[:k] if doc_id in relevant_ids]) / len(relevant_ids) if len(relevant_ids) > 0 else 0.0


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Computes Mean Reciprocal Rank (MRR).

    MRR is the average reciprocal of the rank of the first relevant document retrieved.
    It evaluates how early the first relevant document appears.

    Args:
        retrieved_ids (List[str]): List of document IDs retrieved by the system.
        relevant_ids (List[str]): List of ground-truth relevant document IDs.

    Returns:
        float: MRR score between 0 and 1.
    """
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Computes Average Precision at rank k (AP@k).

    AP@k evaluates how many relevant documents are retrieved and how well they are ranked.
    It averages the precision at each point a relevant document is found.

    Args:
        retrieved_ids (List[str]): List of document IDs retrieved by the system.
        relevant_ids (List[str]): List of ground-truth relevant document IDs.
        k (int): Rank at which to compute the metric.

    Returns:
        float: Average precision score at rank k.
    """
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant_ids) if relevant_ids else 0.0