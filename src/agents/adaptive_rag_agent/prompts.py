import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.llm import llm

from agents.adaptive_rag_agent.structured_outputs import (
    GradeAnswer,
    GradeDocument,
    GradeHallucinations,
    RouteQuery,
)

structured_llm_router = llm.with_structured_output(RouteQuery)

structured_llm_retrieval_grader = llm.with_structured_output(GradeDocument)

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)


# Prompt
router_system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents that are relevant to specific company information such as workflows, processes, policies, products, etc.
If the question can be answered using your general knowledge, use web-search."""
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)

retrieval_grader_system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question and context."""
retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_system),
        (
            "human",
            "Retrieved Document: {document}\n\nContext: {context}\n\nQuestion: {question}",
        ),
    ]
)

hallucination_grader_system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

answer_grader_system = """You are a grader assessing whether an answer addresses / resolves a question and context \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system),
        (
            "human",
            "Context: \n\n {context} \n\n User question: \n\n {question} \n\n LLM generation: {generation}",
        ),
    ]
)

rewriter_system = """You are a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rewriter_system),
        (
            "human",
            "Context: {context}\n\nQuestion: {question}\n\nFormulate an improved question.",
        ),
    ]
)


question_router = router_prompt | structured_llm_router
retrieval_grader = retrieval_grader_prompt | structured_llm_retrieval_grader
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
answer_grader = answer_prompt | structured_llm_answer_grader
question_rewriter = re_write_prompt | llm | StrOutputParser()

rag_system = """
    You are a helpful assistant that can answer questions based on the provided context."""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_system),
        ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer:"),
    ]
)

rag_chain = rag_prompt | llm | StrOutputParser()