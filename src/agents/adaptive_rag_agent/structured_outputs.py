from pydantic import BaseModel, Field
from typing_extensions import Literal

class RouteQuery(BaseModel):
    """
    Route query to most relevant data sources
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., description="The data source to route the query to. Vectorstore stores company data, while websearch queries the web."
    )


class GradeDocument(BaseModel):
    """
    Grade the relevance of a document to a question
    """

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="The relevance of the document to the question, either yes or no",
    )


class GradeHallucinations(BaseModel):
    """
    Grade the hallucinations in a document
    """

    binary_score: Literal["yes", "no"] = Field(
        ..., description="Whether the document contains hallucinations"
    )


class GradeAnswer(BaseModel):
    """
    Grade the answer to a question
    """

    binary_score: Literal["yes", "no"] = Field(
        ..., description="Whether the answer addresses the question, either yes or no"
    )
