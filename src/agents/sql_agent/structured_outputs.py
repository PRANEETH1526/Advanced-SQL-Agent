from pydantic import BaseModel, Field
from typing_extensions import Literal

class Subtasks(BaseModel):
    subtasks: list[str] = Field(
        description="The list of subtasks to be transformed into queries"
    )

class ContextInformation(BaseModel):
    tables: str = Field(
        description="A brief description of the table and the COMMENT section of the schema if there is one"
    )
    relationships: str = Field(
        description="A list of key relationships between the tables including the join conditions"
    )
    relevant_fields: str = Field(
        description="A list of relevant fields in the tables that are useful for the query including their data types"
    )


class Query(BaseModel):
    query: str = Field(description="The SQL query to be executed")
    reasoning: str = Field(description="The reasoning for the query")

class SufficientTables(BaseModel):
    sufficiency_decision: bool = Field(
        description="True if the provided information is sufficient for the query, False otherwise"
    )
    reason: str = Field(description="The reasoning for the answer")

class TransformUserQuestion(BaseModel):
    question: str = Field(description="The transformed user question")

class QueryClassification(BaseModel):
    classification: Literal["GeneralQuestion", "SQLRequest"] = Field(
        description="The classification of the user question"
    )