from langchain_core.prompts import ChatPromptTemplate
from agents.llm import llm, mini_llm
from agents.sql_agent.structured_outputs import TransformUserQuestion, SufficientTables, Query, Subtasks, SimpleQuery

transform_user_question_system = """

You will receive a user question and your job is to make the question more specific and clear.
Try to make the question as simple as possible.

"""

transform_user_question_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", transform_user_question_system),
        ("placeholder", "{messages}"),
    ]
)

transform_user_question_llm = mini_llm.with_structured_output(TransformUserQuestion)

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

Task: Organise the information from the selected tables and their schemas into a structured format and to eliminate fields in the schemas that are not relevant to the user question.
When in doubt, keep the field. Retain the COMMENT section of the schema including all the joins information which is crucial. 

Optionally you may be given the information you generated in the previous iteration. Fill in the missing information given the new information. 
You MUST NOT remove any information from the previous iteration but you can add new information.

Output Format:
- Tables: <A brief description of the table and the COMMENT section of the schema if there is one>
- Key Relationships: <A list of key relationships between the tables including the join conditions>
- Relevant Fields for the User Question: <A list of relevant fields for the user question including the type of the field>
"""

contextualiser_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualiser_system),
        ("placeholder", "{messages}"),
    ]
)

sufficient_tables_system = """ 
You are a specialist at evaluating the sufficiency of information for generating SQL queries.

Task:
You are given organized information about a user query, along with a list of schema details and table relationships selected by a Selector LLM. Your task is to determine if the currently selected tables and their fields are sufficient to generate an executable SQL query that answers the user’s question.

Instructions:

1. Sufficiency Evaluation:
Use chain-of-thought reasoning to assess whether, by properly joining the selected tables, there is enough information to answer the query.
If any critical information is missing (e.g., required fields, table relationships, or key attributes), you may mark the selection as insufficient.
However, before concluding insufficiency, verify carefully that the required information is not already covered by the selected tables or their known relationships.
If uncertain, default to "sufficient" to avoid overestimating missing elements.

2. Output Format:
Sufficient: Return True if the current selection is sufficient. Return False if it is not.
Reason: If True, omit this field. If False, provide a clear explanation of what's missing. Only suggest tables or fields that are not already present in the selected list.

Additional Notes:
Do not suggest a table or field that is already included in the selection.
If the missing data is a field within an already selected table, suggest only the missing field(s), not the entire table.
Reference the example SQL query (if provided) to guide your reasoning.
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

query_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_gen_system),
        ("placeholder", "{messages}"),
    ]
)

query_gen_llm = llm.with_structured_output(Query)

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

reducer_llm = llm.with_structured_output(Query)

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
Your job is to determine if the user question is simple and doesn't require any breakdown or if there is enough information already to answer the user question

If upon reflection you believe that the user question is simple (i.e. only one table is needed to answer the question, no joins are needed, etc.)
return "True".
If you believe that the user question is complex (i.e. multiple tables are needed, joins required), return "False".
If you are unsure, return "False".

Simple query example: Get the number of entries in the components table
Complex query example: Get the number of components broken down by category, with percentage (components table joined with categories table)
"""

query_router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", query_router_system),
        ("placeholder", "{messages}"),
    ]
)

query_router_llm = llm.with_structured_output(SimpleQuery)