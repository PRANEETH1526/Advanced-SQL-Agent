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

class SelectedQueries(BaseModel):
    selected_ids: list[int] = Field(
        description="The list of ids of the selected relevant queries"
    )
    reasoning: str = Field(
        description="The reasoning for the selected queries"
    )

class LineGraphInput(BaseModel):
    """Schema for creating a simple line graph."""

    x: List[float] = Field(..., description="X‑axis values (must match length of y)")
    y: List[float] = Field(..., description="Y‑axis values (must match length of x)")
    title: str = Field("Line Graph", description="Figure title")
    x_label: str = Field("X", description="Label for the X‑axis")
    y_label: str = Field("Y", description="Label for the Y‑axis")

    @model_validator(mode="before")
    def check_lengths(cls, values):
        x, y = values.get("x"), values.get("y")
        if x is not None and y is not None and len(x) != len(y):
            raise ValueError("x and y must have the same length")
        return values

class BarGraphInput(LineGraphInput):
    """Schema for creating a simple bar graph."""

    categories: List[str] = Field(..., description="Category labels (x‑axis)")
    values: List[float] = Field(..., description="Bar heights (must match length of categories)")
    title: str = Field("Bar Graph", description="Figure title")
    x_label: str = Field("Category", description="Label for the X‑axis")
    y_label: str = Field("Value", description="Label for the Y‑axis")

    @model_validator(mode="before")
    def check_lengths(cls, values):
        categories, values = values.get("categories"), values.get("values")
        if categories is not None and values is not None and len(categories) != len(values):
            raise ValueError("categories and values must have the same length")
        return values
