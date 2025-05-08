from agents.llm import embeddings as dense_embedder
#from agents.llm import sparse_embeddings as sparse_embedder
from typing import Dict, Any, List
from pymilvus import MilvusClient, DataType, Collection, AnnSearchRequest, WeightedRanker, connections

DATABASE_URI = "./intellidesign.db"
COLLECTION_NAME = "sql_agent" # default collection name

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def get_collection(database_uri=DATABASE_URI, collection_name=COLLECTION_NAME) -> Collection:
    """
    Connect to the Milvus server and get the collection.
    Returns:
        Collection: The Milvus collection.
    """
    connections.disconnect("default")
    connections.connect(
        uri=database_uri,
        alias="default",
    ) 
    return Collection(collection_name)


def create_collection():
    """
    Create a new collection in Milvus with the specified schema and index parameters.
    This function checks if the collection already exists, and if not, it creates a new one. 
    """
    client = MilvusClient(
        uri=DATABASE_URI
    )

    if client.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return
    
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=4000, enable_analyzer=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1536)

    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        index_name="vector_index",
        index_type="IVF_FLAT",
        metric_type="IP",
        params={"nlist": 128}
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    ) 

def insert_data(collection: Collection, information: str):
    """
    Insert data into the Milvus collection.
    Args:
        collection (Collection): The Milvus collection.
        information (str): The information to insert.
    """
    data = [
        {
            "text": information,
            "vector": dense_embedder.embed_query(information),
        }
    ]
    collection.insert(data)
    print(f"Inserted data into collection '{COLLECTION_NAME}'.")

def delete_data(collection: Collection, doc_id: int):
    """
    Delete data from the Milvus collection.
    Args:
        collection (Collection): The Milvus collection.
        doc_id (int): The document ID to delete.
    """
    collection.delete(f"id == {doc_id}")
    print(f"Deleted document with ID {doc_id} from collection '{COLLECTION_NAME}'.")


def dense_search(col: Collection, query: str, limit=10, expr=None) -> List[Dict]:
    """
    Perform a dense search in the Milvus collection.
    Args:
        col (Collection): The Milvus collection to search.
        query (str): The query string.
        limit (int): The number of results to return.
        expr (str): Optional expression for filtering results.
    Returns:
        List[Dict]: List of dictionaries containing the search results.
    """
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [dense_embedder.embed_query(query)],
        anns_field="vector",
        limit=limit,
        param=search_params,
        output_fields=["text", "doc_id", "source", "title", "url"],
        expr=expr
    )[0]
    return res

def sparse_search(col: Collection, query: str, limit=10, expr=None, remove_stopwords=False) -> List[Dict]:
    """
    Perform a sparse search in the Milvus collection.
    Args:
        col (Collection): The Milvus collection to search.
        query (str): The query string.
        limit (int): The number of results to return.
        expr (str): Optional expression for filtering results.
        remove_stopwords (bool): Whether to remove stopwords from the query.
    Returns:
        List[Dict]: List of dictionaries containing the search results.
    """
    if remove_stopwords:
        tokens = word_tokenize(query)
        query = ' '.join([word for word in tokens if word.lower() not in stop_words])
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [sparse_embedder.encode_queries([query])["sparse"]],
        anns_field="sparse",
        limit=limit,
        param=search_params,
        output_fields=["text", "doc_id", "source", "title", "url"],
        expr=expr
    )[0]
    return res
    
def hybrid_search(
    col: Collection,
    query: str,
    sparse_weight=0.7,
    dense_weight=1.0,
    limit=10,
    expr=None,
    remove_stopwords=True
) -> List[Dict]:
    """
    Perform a hybrid search in the Milvus collection.
    Args:
        col (Collection): The Milvus collection to search.
        query (str): The query string.
        sparse_weight (float): Weight for sparse search results.
        dense_weight (float): Weight for dense search results.
        limit (int): The number of results to return.
        expr (str): Optional expression for filtering results.
        remove_stopwords (bool): Whether to remove stopwords from the query.
    Returns:
        List[Dict]: List of dictionaries containing the search results.
    """
    dense_search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10}
    }
    dense_req = AnnSearchRequest(
        data=[dense_embedder.embed_query(query)],
        anns_field="vector",
        param=dense_search_params,
        limit=limit,
        expr=expr
    )

    sparse_search_params = {
        "metric_type": "IP",
        "params": {}
    }

    if remove_stopwords:
        tokens = word_tokenize(query)
        query = ' '.join([word for word in tokens if word.lower() not in stop_words])

    sparse_req = AnnSearchRequest(
        data=[sparse_embedder.encode_queries([query])["sparse"]],
        anns_field="sparse",
        param=sparse_search_params,
        limit=limit,
        expr=expr
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req],
        rerank=rerank,
        limit=limit,
        output_fields=["text", "doc_id", "source", "title", "url"],
    )[0]
    return res

def merge_chunks_in_results(docs):
    """
    Merges chunks of documents with the same title into a single document.
    Args:
        docs (List[Document]): List of documents to merge.
    Returns:
        List[Document]: List of merged documents.
    """
    merged_docs = {}
    best_scores = {}

    for doc in docs:
        score = doc.distance
        entity = doc.fields
        title = entity.get("title")
        if title in merged_docs:
            merged_docs[title]["text"] += entity.get("text")
            best_scores[title] = max(best_scores[title], score)
        else:
            merged_docs[title] = entity 
            best_scores[title] = score
        
    for title in merged_docs:
        merged_docs[title]["score"] = best_scores[title]
    
    return list(merged_docs.values())

def process_chunks(docs):
    """
    Processes chunks of documents to ensure they are in the correct format.
    Args:
        docs (List[Document]): List of documents to process.
    Returns:
        List[Document]: List of processed documents.
    """
    processed_docs = set()
    for doc in docs:
        score = doc.distance
        entity = doc.fields
        title = entity.get("title")
        processed_docs.add({
            "text": entity.get("text"),
            "score": score,
            "source": entity.get("source"),
            "url": entity.get("url"),
            "title": title,
            "doc_id": entity.get("doc_id"),
        })
        
    return list(processed_docs)

def normalize_scores(results: List[Dict], epsilon: float = 0.3) -> List[Dict]:
    """
    Normalize the scores of the documents in the search results, ensuring no score is zero.
    Args:
        results (List[Dict]): List of dictionaries containing the search results.
        epsilon (float): Minimum score after normalization to avoid 0 relevance.
    Returns:
        List[Dict]: List of dictionaries with normalized scores.
    """
    if not results:
        return results

    scores = [doc['score'] for doc in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        for doc in results:
            doc['score'] = 1.0
        return results

    for doc in results:
        norm_score = (doc['score'] - min_score) / (max_score - min_score)
        doc['score'] = norm_score * (1 - epsilon) + epsilon  # Scales to [ε, 1]

    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def json_to_milvus_filter(filters):
    """
    Convert a list of filter conditions into a Milvus filter expression.
    :param filters: list - A list of dictionaries representing filter conditions.
    :return: str - Milvus filter expression.
    """
    def parse_condition(field, operator, value):
        if isinstance(value, str):
            value = f'"{value}"'  # Ensure strings are quoted
        elif isinstance(value, list):
            formatted_items = []
            for x in value:
                if isinstance(x, str):
                    formatted_items.append(f'"{x}"')
                else:
                    formatted_items.append(str(x))
            value = f'[{", ".join(formatted_items)}]'
        
        if operator == "==":
            return f"{field} == {value}"
        elif operator == "!=":
            return f"{field} != {value}"
        elif operator in [">", "<", ">=", "<="]:
            return f"{field} {operator} {value}"
        elif operator == "IN":
            return f"{field} in {value}"
        elif operator == "LIKE":
            return f"{field} LIKE {value}"
        elif operator == "JSON_CONTAINS":
            return f"json_contains({field}, {value})"
        elif operator == "JSON_CONTAINS_ALL":
            return f"json_contains_all({field}, {value})"
        elif operator == "JSON_CONTAINS_ANY":
            return f"json_contains_any({field}, {value})"
        elif operator == "ARRAY_CONTAINS":
            return f"ARRAY_CONTAINS({field}, {value})"
        elif operator == "ARRAY_CONTAINS_ALL":
            return f"ARRAY_CONTAINS_ALL({field}, {value})"
        elif operator == "ARRAY_CONTAINS_ANY":
            return f"ARRAY_CONTAINS_ANY({field}, {value})"
        elif operator == "ARRAY_LENGTH":
            return f"ARRAY_LENGTH({field}) {value}"  # Expecting condition in value (e.g., "< 10")
        else:
            raise ValueError(f"Unsupported operator: {operator}")
    
    def parse_filters(filters):
        expressions = []
        for condition in filters:
            field = condition["field"]
            operator = condition["operator"]
            value = condition["value"]
            expressions.append(parse_condition(field, operator, value))
        return " AND ".join(expressions)  # Default logical operator is AND
    
    return parse_filters(filters)

def search_vectorstore(
    col: Collection, 
    query: str, 
    k: int = 5, 
    filters: Dict[str, Any] = None, 
    search_type="dense", 
    merge=True,
    remove_stopwords=False) -> List[Dict]:
    """
    Search the vector store for relevant documents based on the query.
    Args:
        collection (Collection): The Milvus collection to search.
        query (str): The query string.
        k (int): The number of results to return.
        filters (Dict[str, Any]): Optional filters for the search.
        search_type (str): Type of search to perform ("dense", "sparse", "hybrid").
        merge (bool): Whether to merge duplicate documents.
    Returns:
        List[Dict]: List of dictionaries containing the search results.
    """
    if filters:
        expr = json_to_milvus_filter(filters)
    else:
        expr = None

    search_results = []

    if search_type == "dense":
        search_results = dense_search(col, query, k, expr)
    elif search_type == "sparse":
        search_results = sparse_search(col, query, k, expr, remove_stopwords=remove_stopwords)
    else:
        search_results = hybrid_search(col, query, limit=k, expr=expr, remove_stopwords=remove_stopwords)
    
    if merge:
        search_results = merge_chunks_in_results(search_results)
    else:
        search_results = process_chunks(search_results)
    
    search_results = normalize_scores(search_results)

    return search_results

if __name__ == "__main__":
    # Example usage
    collection = get_collection()
    
    print(results)
    delete_data(collection, 1)