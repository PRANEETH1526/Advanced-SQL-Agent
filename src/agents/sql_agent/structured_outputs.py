from pydantic import BaseModel, Field

class Subtasks(BaseModel):
    subtasks: list[str] = Field(
        description="The list of subtasks to be transformed into queries"
    )

class Query(BaseModel):
    query: str = Field(description="The SQL query to be executed")
    reasoning: str = Field(description="The reasoning for the query")

class SufficientTables(BaseModel):
    sufficient: bool = Field(
        description="Whether the information from the tables are sufficient to answer the user question"
    )
    reason: str = Field(description="The reason for the answer")

class TransformUserQuestion(BaseModel):
    question: str = Field(description="The transformed user question")

class SimpleQuery(BaseModel):
    simple: bool = Field(description="Whether the query is simple or not")