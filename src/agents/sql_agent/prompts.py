from langchain_core.prompts import ChatPromptTemplate
from agents.llm import llm
from agents.sql_agent.schema import TransformUserQuestion, SufficientTables, SQL, Subtasks, QueryClassification, QuerySelection, ChartType

transform_user_question_system = """

You will receive a user question and your job is to make the question more clear. Don't add more to the question. Here is some context to help you:
- The company is an electronics manufacturing company, assembling parts onto PCBs, and assembling those into products.
- POs are Purchase Orders
- IDs refer to component IDs unless specified otherwise
- 'Purchase' and 'Order' are interchangeable terms
- 'Parts' and 'Components' are interchangeable terms
- Resistors, Capacitators, etc are component categories

You are also give the date and time, so if the user question is related to a specific date, you can use that information to make the question more clear.
For example, if the user question is "What are the POs for the last 3 months?", you can transform it to "What are the POs for the last 3 months from (date)".
"""

transform_user_question_prompt = ChatPromptTemplate(
    [
        ("system", transform_user_question_system),
        ("placeholder", "{messages}"),
    ]
)

transform_user_question_llm = llm.with_structured_output(TransformUserQuestion)

selector_system = """
You are provided with a user's natural language query alongside a list of table names and their corresponding descriptions. 
Your task is to identify all tables that could potentially contribute to answering the query.
You could also receive feedback from your last iteration of selecting tables. Take the feedback into account. In this case, select tables that you have not selected in the previous iteration.

Instructions:

Inclusiveness is Key:
Err on the side of inclusion. If a table might be relevant—even indirectly—include it. When in doubt, select the table.

Analytical Review: 
Examine each table's name and description carefully.
Consider potential joins or relationships: a table that seems marginal on its own may become crucial when combined with others.

Evaluate Relevance:
Identify keywords, data types, or any hints in the descriptions and table COMMENTS that suggest the table might hold useful data.
Look for connections between tables (e.g., shared fields or related concepts) that might allow for joining to gather comprehensive data.

Call the get_schema_tool to get the schema of these selected tables.
"""

selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", selector_system),
        ("placeholder", "{messages}"),
    ]
)

contextualiser_system = """
You are a specialist at organising and structuring information and you have strong attention to detail.

You will receive the following information:
- A user question
- A list of all tables and their descriptions in the database 
- A list of selected relevant tables and their schemas. The schemas have a comment section that describes the table and the relationship with other tables.
- Context from semantically similar questions that have been asked in the past. Always include this context in your output.

Task: Organise the information from the selected tables and their schemas into a structured format and to eliminate fields in the schemas that are not relevant to the user question.
When in doubt, keep the field. Retain the COMMENT section of the schema including all the joins information which is crucial. Use the context from the semantically similar questions to help you.

Optionally you may be given the information you generated in the previous iteration. Fill in the missing information given the new information. 
You MUST NOT remove any information from the previous iteration but you can add new information.

Output Format:
- Tables: <A brief description of the table and the COMMENT section of the schema if there is one>
- Key Relationships: <A list of key relationships between the tables including the join conditions>
- Relevant Fields for the User Question: <A list of relevant fields for the user question including the type of the field>
- Context: <The context from the semantically similar questions that have been asked in the past. This context may include the user question, the selected tables, or the generated SQL queries.>
"""

contextualiser_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualiser_system),
        ("placeholder", "{messages}"),
    ]
)

sufficient_tables_system = """ 
You are a specialist in evaluating the sufficiency of context information for generating SQL queries.

Task:
Given a user's query and cached context from previously successful queries (including selected tables and fields), determine if the provided context is sufficient to generate an executable SQL query that accurately answers the user's current question.

Evaluation Criteria:
Tables and Fields: Verify if the selected tables and fields, when properly joined based on existing relationships, fully address the user's query.
Completeness: Confirm that all necessary fields required to answer the user's question are included.
Relationships: Ensure relationships among tables (e.g., foreign keys) required for a successful join are explicitly available in the provided context.

Decision Logic:
If the current context explicitly covers all required tables, fields, and their relationships, mark as sufficient (True).
If critical information is clearly missing (e.g., essential fields, missing table relationships), mark as insufficient (False) and explicitly state what's missing.
In borderline cases or uncertainty, default to sufficient (True) to avoid unnecessary information retrieval.

Output Format:
Sufficient: Return True if context is sufficient. Return False if insufficient.
Reason: Omit if sufficient (True). If insufficient (False), concisely explain the missing information. Only suggest tables or fields not already present in the cached context.

Important Reminders:
Do not suggest tables or fields already included in the context.
If missing data involves a field from an already selected table, only suggest that specific missing field.
Use the provided example SQL query (if available) as guidance to evaluate sufficiency accurately.
"""

sufficient_tables_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sufficient_tables_system),
        ("placeholder", "{messages}"),
    ]
)

sufficient_tables_llm = llm.with_structured_output(SufficientTables)

query_gen_system = """
You are an expert SQL developer.

You are provided with a user question, and structure information necessary for constructing a MariaDB SQL query. 
Additionally, you may sometimes receive an error message indicating that the previously generated query failed upon execution. 

Instructions:
If you don't get an error message, use the information to generate an SQL query that is syntactically correct for the user question.

Initial Query Generation:
Refer to the example if provided. 
Construct the SQL Query: Generate a valid MariaDB SQL query that incorporates all necessary joins, filters, and calculations. Joins must use indexed fields. 
Output Format: Return a single SQL statement. Do not hallucinate any field names or table names.

If you get an error message, generate a new SQL query that is syntactically correct.

Error Correction:
Review the Error: If an error message is provided along with the context, analyze the error to determine the underlying issue (e.g., syntax errors, aliasing problems, missing join conditions).
Modify the Query: Adjust the previously generated SQL query to correct the error while ensuring that it still meets the requirements derived from the context.
Document the Correction: Include brief inline comments in the query to describe the changes made to address the error.
Output Format: Return the corrected SQL query as a single statement. 
"""

query_gen_prompt = ChatPromptTemplate(
    [
        ("system", query_gen_system),
        ("placeholder", "{messages}"),
    ]
)

query_gen_llm = llm.with_structured_output(SQL)

query_gen_with_context_system = """
You are an expert SQL developer.

You are provided with a user question, and an SQL queries of semantically similar questions that have been asked in the past.

Your job is to analyze and use the provided SQL queries and generate a new MariaDB SQL query that can specifically answer the user question.
Some queries are the essentially the same as the user question so they just need slight modifications, so in this case, keep the query as is and just modify it to fit the user question.

Output Format: Return a single SQL statement. Do not hallucinate any field names or table names.
"""

query_gen_with_context_prompt = ChatPromptTemplate(
    [
        ("system", query_gen_with_context_system),
        ("placeholder", "{messages}"),
    ]
)

query_gen_with_context_llm = llm.with_structured_output(SQL)

decomposer_system = """
You are an expert SQL developer.

You will receive a user question that represents a query to a database.

If the user question is simple and doesn't require any breakdown, return a list with a single task (original user question).

If the user question is complex and has multiple subqueries, decompose the user question into a list of independent subtasks that can be executed in parallel.

Otherwise, return a list with a single task (original user question). Only decompose if the subtasks are independent of each other.
"""

decomposer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", decomposer_system),
        ("placeholder", "{messages}"),
    ]
)

decomposer_llm = llm.with_structured_output(Subtasks)

reducer_system = """
You are an expert SQL developer.

You will receive a list of SQL queries and a user question.

These SQL queries are subtasks that are part of the overall task to answer the user question.

Your job is to reduce the list of MariaDB SQL queries into a single MariaDB SQL query that answers the user question.

Ensure that the final query is efficient.

The SQL query should be syntactically correct.

Double check the MariaDB query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins
"""

reducer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", reducer_system),
        ("placeholder", "{messages}"),
    ]
)

reducer_llm = llm.with_structured_output(SQL)

answer_and_reasoning_system = """
You are an advanced reasoning agent.

You will receive a user question, some intermediate reasoning to generate an sql query, and the final raw output from an SQL query. 

Your job is to answer the user question by interpreting the information provided in the messages and present it in a structured and detailed manner.

Provide a structured chain of thought reasoning for your answer. Go through the reasoning step by step and the intermediate results.

If you don't have enough information to answer the question, say "I don't know".
"""

answer_and_reasoning_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_and_reasoning_system),
        ("placeholder", "{messages}"),
    ]
)

query_router_system = """
You are an intelligent assistant that classifies user queries into two categories to decide the correct next step:

1. **GeneralQuestion**  
   - Use this if the query is a broad question about the database structure, table purposes, column descriptions, or schema-level understanding.  
   - **Examples:**
     - "What tables are available?"
     - "What does the `inventory` table track?"
     - "Explain the schema."

2. **SQLRequest**  
   - Use this if the query asks for specific data that requires generating and running an SQL query to extract results from the database.  
   - **Examples:**
     - "List all parts with low stock"
     - "Show recent purchase orders"
     - "What's the quantity of item 123 in stock?"


Classify the following user query strictly as either `"GeneralQuestion"` or `"SQLRequest"`.
Return **only** the classification label.
"""

query_router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_router_system),
        ("placeholder", "{messages}"),
    ]
)

query_router_llm = llm.with_structured_output(QueryClassification)

relevant_questions_selector_system = """
You are an LLM-powered RERANKER.

## Inputs
- **user_question**: A single natural-language query from the user.
- **candidate_questions**: A list of previously-asked questions (strings) along with their IDs that may or may not be relevant.

## Task
For each candidate question, judge its semantic similarity to **user_question** and decide whether it is *somewhat related* (i.e. likely to help answer or clarify the user's need).
Each of these candidate questions is a reference into an SQL library. The relevant SQL queries will be provided as context in a later step, to construct a new query to answer the user's question.
Note that specific numbers in the query should not be used to judge relevance.
To fully answer the user question, multiple responses may need to be combined, so you will need to consider whether the chosen set of candidates fully covers the scope of the user question. You may need to include only partially relevant questions to allow the whole question to be answered.
Err on the side of inclusion. Discard candidates that are clearly irrelevant.

## Output
Return a list of the reranked candidate question IDs sorted by how similar/helpful they are for the user question — or an empty list (`[]`) 
Also return a reasoning for the selected IDs. The reasoning should be a single sentence that explains why the selected IDs are relevant to the user question.
"""
relevant_questions_selector_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", relevant_questions_selector_system),
        ("placeholder", "{messages}"),
    ]
)

relevant_questions_selector_llm = llm.with_structured_output(QuerySelection)

chart_generator_system = """
You will be given a user question, and the results of an SQL query that has been executed. 

Your job is to determine if the results of the SQL query can be used to generate a chart.
If the results can be used to generate a chart, return the type of chart that can be generated - line or bar at the moment.

If the results cannot be used to generate a chart, return "no chart".
"""

chart_generator_system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chart_generator_system),
        ("placeholder", "{messages}"),
    ]
)
chart_generator_llm = llm.with_structured_output(ChartType)
