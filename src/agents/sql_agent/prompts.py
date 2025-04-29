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
Task: You receive schema and relationships information about the tables selected by the Selector LLM along with the original user query. 
Your job is to determine whether the fields and tables provided in the selected tables are sufficient to generate an SQL query that can be executed to answer the user's query.
If they are not sufficient, you must provide a clear explanation of what is missing and suggest additional tables.

Instructions:

Sufficiency Evaluation:
Use chain of thought reasoning to determine that if appropriately joining the provided tables, we have sufficient information to answer the user's query. Refer to the example sql query provided in the context if available.
If this is the case, state that the selection is sufficient. 
If any critical tables and their information or relevant fields from the current tables are missing, mark the decision as insufficient. Default to "sufficient" if you are unsure.

Output Format:
Sufficient: True if there is enough information to generate an SQL query. Return False if there is not enough information.
Reason: If the tables/information is sufficient, return nothing. If the tables/information provided is insufficient, suggest ALL the tables/fields that could be relevant that hasnt been included.
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

You are provided with a user question, a list of table names, schema summaries, join conditions, and any additional notes necessary for constructing a MariaDB SQL query. 
Additionally, you may sometimes receive an error message indicating that the previously generated query failed upon execution. 

Instructions:

If you don't get an error message, use chain of thought reasoning to analyse the information in depth and generate an SQL query that is syntactically correct.

Initial Query Generation:
Analyze the Context: Use the provided context to understand the tables involved, their schemas, join conditions, and any filtering or grouping requirements needed to answer the user query.
Construct the SQL Query: Generate a valid MariaDB SQL query that incorporates all necessary joins, filters, and calculations. Joins must use indexed fields. Ensure that the query is efficient and optimized for performance.
Output Format: Return a single SQL statement. Do not hallucinate any field names or table names. Use the field names and table names provided in the context.

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

Your answer should be in the same language as the user question.

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