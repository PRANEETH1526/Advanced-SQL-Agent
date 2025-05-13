import os
import sys
from pathlib import Path
from typing import List, Dict

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eval.query_gen import load_queries, QueryRecord
from eval.scoring import ndcg_at_k, precision_at_k, recall_at_k, mean_reciprocal_rank, average_precision_at_k
from agents.vectorstore import get_collection, search_vectorstore  

# Parameters
K = 5
SEARCH_TYPE = "hybrid"  # Options: "dense", "sparse", or "hybrid"
REMOVE_STOPWORDS = True  # Set to True to remove stop words from the query

DATABASE = "intellidesign.db"  # The name of the collection to use
COLLECTION = "all"  # The name of the collection to use


def evaluate_milvus_retrieval(query_records: List[QueryRecord], k: int = K, search_type: str = SEARCH_TYPE) -> Dict[str, float]:
    """
    Evaluate the performance of Milvus retrieval using nDCG@k and Precision@k.

    Args:
        query_records (List[QueryRecord]): List of query records with generated pseudo queries and gold document IDs.
        k (int): The number of top results to consider for each query.
        search_type (str): The type of vector search to use ("dense", "sparse", or "hybrid").

    Returns:
        Dict[str, float]: A dictionary containing average nDCG@k, Precision@k, and total query count.
    """
    collection = get_collection(DATABASE, COLLECTION)
    total_ndcg = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    total_map = 0.0
    total_queries = 0

    for record in query_records:
        gold_doc_id = record.id
        for pseudo_query in record.queries:
            results = search_vectorstore(
                col=collection,
                query=pseudo_query,
                k=k,
                search_type=search_type,
                merge=False,  # Disable merging to preserve chunk-level granularity
                remove_stopwords=REMOVE_STOPWORDS
            )

            print(f"Results for query '{pseudo_query}': {[doc["score"] for doc in results]}")
            retrieved_ids = [doc["doc_id"] for doc in results]
            print(f"Query: {pseudo_query}, Retrieved IDs: {retrieved_ids}, Gold ID: {gold_doc_id}")
            relevant_ids = [gold_doc_id]  # Assumes one relevant doc per query

            total_ndcg += ndcg_at_k(retrieved_ids, relevant_ids, k)
            total_precision += precision_at_k(retrieved_ids, relevant_ids, k)
            total_recall += recall_at_k(retrieved_ids, relevant_ids, k)
            total_mrr += mean_reciprocal_rank(retrieved_ids, relevant_ids)
            total_map += average_precision_at_k(retrieved_ids, relevant_ids, k)
            total_queries += 1

    return {
        "nDCG@k": total_ndcg / total_queries if total_queries else 0.0,
        "Precision@k": total_precision / total_queries if total_queries else 0.0,
        "Recall@k": total_recall / total_queries if total_queries else 0.0,
        "MRR": total_mrr / total_queries if total_queries else 0.0,
        "MAP@k": total_map / total_queries if total_queries else 0.0,
        "Total Queries": total_queries
    } 

def main(iterations: int = 50):
    """
    Main function to load queries and evaluate Milvus retrieval.
    """
    query_records = load_queries("queries.jsonl")
    # randomly sample a subset of queries
    import random
    random.seed(38)
    query_records = random.sample(query_records, iterations)
    metrics = evaluate_milvus_retrieval(query_records)
    
    print("\n🔍 Milvus Retrieval Evaluation")
    with open("retrieval_eval_results.txt", "a") as f:
        f.write(f"🔍 Milvus Retrieval Evaluation for {SEARCH_TYPE} retrieval with Remove Stopwords = {REMOVE_STOPWORDS} and k = {K}\n and sample = {iterations}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
        # write to file
        with open("retrieval_eval_results.txt", "a") as f:
            f.write(f"{metric}: {value:.4f}\n" if isinstance(value, float) else f"{metric}: {value}\n")


if __name__ == "__main__":
    main()