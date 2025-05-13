import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from agents.llm import llm 

# ------------------ Models ------------------

class Query(BaseModel):
    """
    Represents a generated query.
    """
    query_list: List[str] = Field(..., description="List of queries generated from the original text.")

class QueryRecord(BaseModel):
    """
    Represents a record of the original text and its generated queries.
    """
    id: str
    text: str
    queries: List[str] = Field(..., description="List of generated queries.")

# ------------------ Prompt ------------------

query_generation_system_prompt = """
**System Prompt**
**Role**: You are a **Query Generation Assistant**. Your task is to:
1. Generate a list of 5 pseudo queries based on the original text provided.
2. Ensure that the generated queries are relevant and meaningful. Make sure they are not too similar to each other.
3. The queries should be diverse and cover different aspects of the original text.
4. Include queries that are both broad and specific. 
5. Include queries that don't contain specific keywords from the original text but are still relevant.
6. Include queries that are phrased as questions, statements, or commands.
7. Include queries that are phrased in different styles (e.g., formal, informal, technical, etc.).
8. Include queries that range in length from short to long.
---
"""

query_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_generation_system_prompt),
        ("human", "Generate 3 pseudo queries based on the following text: {text}"),
    ]
)


query_generation_llm = llm.with_structured_output(Query)

# ------------------ Functions ------------------



def generate_queries(id: str, text: str, save_path: str = "queries.jsonl") -> Query:
    """
    Generates pseudo queries based on the provided text and writes them to a JSONL file.

    Args:
        id (str): Unique identifier for the text.
        text (str): The original text to generate queries from.
        save_path (str): Path to the file where the queries will be stored.
    
    Returns:
        Query: The generated queries.
    """
    prompt = query_generation_prompt.format(text=text)
    response = query_generation_llm.invoke(prompt)

    record = QueryRecord(
        id=str(id),
        text=text,
        queries=response.query_list
    )

    # Append to file
    with open(save_path, "a", encoding="utf-8") as f:
        f.write(record.json() + "\n")

    return record

def load_queries(save_path: str = "queries.jsonl") -> List[QueryRecord]:
    """
    Loads all saved queries and their original texts from the JSONL file.

    Args:
        save_path (str): Path to the JSONL file.

    Returns:
        List[QueryRecord]: A list of QueryRecord objects.
    """
    if not Path(save_path).exists():
        return []

    records = []
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            record = QueryRecord(**data)
            records.append(record)
    return records

