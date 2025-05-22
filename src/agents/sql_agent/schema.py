from pydantic import BaseModel, Field, model_validator
from typing_extensions import Literal
from typing import List

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


class SQL(BaseModel):
    SQL: str = Field(description="The generated SQL query to be generated")
    reasoning: str = Field(description="How the SQL was generated")

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

class QuerySelection(BaseModel):
    ids: list[int] = Field(
        description="The list of ids of the selected queries that are relevant to the user question"
    )
    reasoning: str = Field(
        description="A brief reasoning for the selected queries"
    )

class ChartType(BaseModel):
    """Schema for chart type."""
    type: Literal["line", "bar", "none"] = Field(
        description="The type of chart to be created. Can be 'line' or 'bar'. If 'none', no chart will be created."
    )