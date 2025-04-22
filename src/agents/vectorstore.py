from pymilvus import DataType, MilvusClient, connections, Collection
from agents.llm import embeddings

DATABASE_URI = "./sql_agent.db"
COLLECTION_NAME = "sql_agent"



def create_collection():
    """
    Create a new collection in Milvus with the specified schema and index parameters.
    This function checks if the collection already exists, and if not, it creates a new one. 
    """
    client = MilvusClient(
        uri=DATABASE_URI,
    )

    if client.has_collection(collection_name=COLLECTION_NAME):
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


def get_collection() -> Collection:
    """
    Connect to the Milvus server and get the collection.
    Returns:
        Collection: The Milvus collection.
    """
    connections.connect(
        uri=DATABASE_URI,
        alias="default",
    ) 
    return Collection(COLLECTION_NAME)


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
            "vector": embeddings.embed_query(information),
        }
    ]
    collection.insert(data)
    print(f"Inserted data into collection '{COLLECTION_NAME}'.")

def search_data(collection: Collection, query: str, top_k: int = 1):
    """
    Search for data in the Milvus collection.
    Args:
        collection (Collection): The Milvus collection.
        query (str): The query to search for.
        top_k (int): The number of top results to return.
    Returns:
        str: The text of the most relevant result.
    """
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": 10},
    }
    query_vector = embeddings.embed_query(query)
    result = collection.search(
        data=[query_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["text"],
        expr=None,
    )[0]
    return result[0].entity.get("text") if result else None


if __name__ == "__main__":
    # Get the collection
    collection = get_collection()
    # Search for data in the collection
    result = search_data(collection, "component categories purchased in the last year based on total spend")
    print(f"Search result: {result}")
