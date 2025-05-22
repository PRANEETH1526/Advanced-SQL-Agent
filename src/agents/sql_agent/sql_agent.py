import operator
import re
from typing import Annotated, Any, TypedDict
import matplotlib.pyplot as plt
import io  
import base64

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from agents.llm import llm, advanced_llm
from agents.vectorstore import get_collection, insert_data, dense_search, retrieve_contexts
from agents.sql_agent.prompts import (
    transform_user_question_prompt,
    transform_user_question_llm,
    selector_prompt,
    contextualiser_prompt,
    sufficient_tables_prompt,
    sufficient_tables_llm,
    decomposer_prompt,
    decomposer_llm, 
    query_gen_prompt,
    query_gen_llm,
    query_gen_with_context_prompt,
    query_gen_with_context_llm,
    reducer_prompt,
    reducer_llm,
    answer_and_reasoning_prompt,
    query_router_prompt,
    query_router_llm,
    relevant_questions_selector_prompt,
    relevant_questions_selector_llm,
    chart_generator_system_prompt,
    chart_generator_llm
)


from datetime import datetime
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage

DATABASE_URI = "mariadb+pymysql://userconnect@10.1.93.4/cms" 

def download_db():
    db = SQLDatabase.from_uri(DATABASE_URI)
    return db

db = download_db()
collection = get_collection(database_uri="intellidesign.db", collection_name="sql_context")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


class State(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    sufficient_info: bool
    information: str
    query: str
    max_retries: int = 1
    chart_type: str
    answer: str


class QueryTask(TypedDict):
    question: str
    task: str
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    information: str


# Nodes
def first_tool_call(state: State) -> State:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def info_sql_database_tool_call(state: State) -> State:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "info_sql_database_tool", "args": {}, "id": "tool_abcd123"}
                ],
            ),
        ]
    }


def handle_tool_error(state: State) -> State:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, State]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


@tool
def db_query_tool(query: str) -> str:
    """
    Query the database with a MariaDB SQL query
    """
    result = db.run_no_throw(query)
    if result.startswith("Error:"):
        return (
            "Error: Query failed. Please rewrite your query and try again."
            + f"\n\nDetails:{result}"
        )
    if len(result) > 2000:
        return result[:2000] + "..."
    return result


@tool
def info_sql_database_tool() -> str:
    """
    Get the information about the tables in the database
    """
    result = db.run_no_throw(
        "SELECT TABLE_NAME, TABLE_COMMENT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'cms' AND LENGTH(TABLE_COMMENT) > 100;"
    )
    full_response = f"Tables and Descriptions:\n\n{result}"
    return full_response


@tool
def get_schema_tool(table_name: str) -> str:
    """
    get the schema of the table
    """
    schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    result = schema_tool.invoke(table_name)
    pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
    result = re.sub(pattern, "", result)
    result = f"Selected Table {table_name}\n\nSchema: {result}"
    return result

@tool("line_graph_tool") 
def line_graph_tool(x: List[float | int], y: List[float | int], title: str, x_label: str, y_label: str) -> str:  # noqa: D401
    """Generate a PNG line graph encoded in base64.

    Params: 
        x (List[float | int]): X‑axis values (must match length of y)
        y (List[float | int]): Y‑axis values (must match length of x)
        title (str): Figure title
        x_label (str): Label for the X‑axis
        y_label (str): Label for the Y‑axis
    
    Returns a base64‑encoded PNG string so the agent can embed or stream the image directly.
    """
    # Plot
    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Encode to base64 (no disk I/O)
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    fig.savefig("charts/line_graph.png", format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    b64_png = base64.b64encode(buffer.read()).decode()
    markdown = f"![{title}](data:image/png;base64,{b64_png})"
    return markdown

@tool("bar_graph")
def bar_graph_tool(categories: List[str], values: List[float | int], title: str, x_label: str, y_label: str) -> str:  # noqa: D401
    """Generate a PNG bar chart encoded in base64.

    Params:
        categories (List[str]): Category labels (x‑axis)
        values (List[float | int]): Y‑axis values (must match length of categories)
        title (str): Figure title
        x_label (str): Label for the X‑axis
        y_label (str): Label for the Y‑axis

    Returns a base64‑encoded PNG string so the agent can embed or stream the image directly.
    """
    # Plot
    fig, ax = plt.subplots()
    ax.bar(categories, values)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    # Encode to base64 (no disk I/O)
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    fig.savefig("charts/bar_graph.png", format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    b64_png = base64.b64encode(buffer.read()).decode()
    markdown = f"![{title}](data:image/png;base64,{b64_png})"
    return markdown

react_agent = create_react_agent(llm, tools=[list_tables_tool, get_schema_tool])
graph_agent = create_react_agent(llm, tools=[line_graph_tool, bar_graph_tool])

def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

def transform_user_question(state: State) -> State:
    try:
        system_prompt = read_file("/server/intelliweb/ai_prompt_files/sql_agent_transform_user_question.txt")
        system_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        )
    except FileNotFoundError:
        # not found, use the default prompt
        system_prompt = transform_user_question_prompt
    prompt = system_prompt.format(messages=state["messages"])
    response = transform_user_question_llm.invoke(f"Current Date: {datetime.now()}\n\n{prompt}")
    return {
        "question": response.question,
        "messages": [AIMessage(content=response.question)],
        "max_retries": 1
    }

def get_context(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    context = retrieve_contexts(
            collection,
            state["question"],
            limit=8,
        )
    if not context:
        return {
            "messages": [AIMessage(content="No context found")],
            "information": "",
        }
    relevant_questions = []
    id_to_context = {}
    for doc in context:
        id = doc["id"]
        query = doc["query"]
        context = doc["context"]
        relevant_questions.append(f"ID {id}: {query}")
        id_to_context[id] = context 
    prompt_input = [HumanMessage(f"User Question: {user_question}\n\nCandidate Questions:\n" + "\n".join(relevant_questions))]
    try:
        system_prompt = read_file("/server/intelliweb/ai_prompt_files/sql_agent_get_context.txt")
        system_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        )
    except FileNotFoundError:
        # not found, use the default prompt
        system_prompt = relevant_questions_selector_prompt
    question_selection = system_prompt.format(messages=prompt_input)
    response = relevant_questions_selector_llm.invoke(question_selection) 
    information = ""
    for id in response.ids:
        doc = id_to_context[id]
        information += f"{doc}\n\n" 
    return {"information": information, "messages": [AIMessage(content=f"Retrieved Context: {information}")]}

def selector(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    messages = [user_question]
    last_message = state["messages"][-1]
    max_retries = state["max_retries"]
    sufficient_info = state.get("sufficient_info", True)
    if not sufficient_info:
        max_retries -= 1
    messages.append(last_message)
    prompt = selector_prompt.format(messages=messages)
    model_get_schema = llm.bind_tools([get_schema_tool], tool_choice="required")
    response = model_get_schema.invoke(prompt)
    return {"messages": [response], "max_retries": max_retries}

def contextualiser(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    unique_tool_calls = set([
        message.content for message in state["messages"] if isinstance(message, ToolMessage)
    ])
    db_info = [
        AIMessage(content=message)
        for message in unique_tool_calls
    ]
    messages = [user_question] + db_info
    prompt = contextualiser_prompt.format(messages=messages)
    response = llm.invoke(prompt)
    information = response.content

    return {"messages": [response], "information": information} 


def sufficient_tables(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information]
    prompt = sufficient_tables_prompt.format(messages=messages)
    # add timestamp to the prompt
    response = sufficient_tables_llm.invoke(f"Timestamp: {datetime.now()}\n\n{prompt}")
    full_response = (
        f"Sufficient Tables: {response.sufficiency_decision}\n\nReason: {response.reason}"
    )
    return {
        "messages": [AIMessage(content=full_response)],
        "sufficient_info": response.sufficiency_decision,
    }


def decomposer(state: State) -> State:
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information]
    prompt = decomposer_prompt.format(messages=messages)
    response = decomposer_llm.invoke(prompt)
    full_response = f"Subtasks: {response.subtasks}"
    return {
        "query_tasks": response.subtasks,
        "messages": [AIMessage(content=full_response)],
    }


def query_gen(state: QueryTask) -> State:
    last_message = state["messages"][-1]
    user_question = HumanMessage(content=state["question"])
    messages = [user_question, AIMessage(content=state["information"])]

    if last_message.content.startswith("Error:"):
        messages.append(last_message)

    prompt = query_gen_prompt.format(messages=messages)
    response = query_gen_llm.invoke(prompt)

    full_response = f"Query: {response.SQL}\n\nReasoning: {response.reasoning}"
    return {
        "messages": [AIMessage(content=full_response)],
        "query": response.SQL,
    }

def query_gen_with_context(state: QueryTask) -> State:
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information]
    try:
        system_prompt = read_file("/server/intelliweb/ai_prompt_files/sql_agent_query_gen_with_context.txt")
        system_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt,
                ),
                ("placeholder", "{messages}"),
            ]
        )
    except FileNotFoundError:
        # not found, use the default prompt
        system_prompt = query_gen_with_context_prompt
    prompt = system_prompt.format(messages=messages)
    response = query_gen_with_context_llm.invoke(prompt)
    full_response = f"Query: {response.SQL}\n\nReasoning: {response.reasoning}"
    return {
        "messages": [AIMessage(content=full_response)],
        "query": response.SQL,
    }


def reducer(state: State) -> State:
    subtasks = AIMessage(content="\n".join(state["query_tasks"]))
    sql_queries = AIMessage(content="\n".join(state["sql_queries"]))
    user_question = HumanMessage(content=state["question"])
    information = AIMessage(content=state["information"])
    messages = [user_question, information, subtasks, sql_queries]
    prompt = reducer_prompt.format(messages=messages)
    response = reducer_llm.invoke(prompt)
    full_response = f"Query: {response.SQL}\n\nReasoning: {response.reasoning}"
    state["query"] = response.SQL
    return {"query": response.SQL, "messages": [AIMessage(content=full_response)]}


def get_query_execution(state: State) -> State:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "db_query_tool",
                        "args": {"query": state["query"]},
                        "id": "query_execution",
                    },
                ],
            )
        ]
    }

def chart_generation(state: State) -> State:
    data = state["messages"][-1]
    user_question = HumanMessage(content=state["question"])
    query = state["query"]
    prompt = chart_generator_system_prompt.format(messages=[user_question, data])
    chart_type = chart_generator_llm.invoke(prompt).type
    response = AIMessage(content="Could not generate chart")
    if chart_type == "line":
        line_model = llm.bind_tools([line_graph_tool], tool_choice="required")
        response = line_model.invoke(f"Generate a line graph based on the following SQL Query and Data:\n\nQuery {query}\n\nData {data}")
    elif chart_type == "bar":
        bar_model = llm.bind_tools([bar_graph_tool], tool_choice="required")
        response = bar_model.invoke(f"Generate a bar graph based on the following SQL Query and Data:\n\nQuery {query}\n\nData {data}")
    return {"messages": [response], "chart_type": chart_type}


def final_answer(state: State) -> State:
    return {
        "messages": [AIMessage(content=state["query"])],
    }


# Edges

def sufficient_context(state: State) -> State:
    information = state["information"]
    if information:
        return "query_gen_with_context"
    else:
        return "info_sql_database_tool_call"
    
def map_query_task(state: State):
    messages = state["messages"]
    return [
        Send(
            "query_gen",
            {
                "task": task,
                "messages": messages,
                "question": state["question"],
                "information": state["information"],
            },
        )
        for task in state["query_tasks"]
    ]

def query_router(state: State) -> State:
    """
    Maps to query gen straight away if the query is simple, or it can already be answered with the information provided.

    Maps to contextualiser if the query is complex and needs to be broken down.
    """
    user_question = state["messages"][0]
    messages = [user_question]
    prompt = query_router_prompt.format(messages=messages)
    response = query_router_llm.invoke(prompt)
    if response.classification == "SQLRequest":
        return "transform_user_question"
    elif response.classification == "GeneralQuestion":
        return "react_agent"

def continue_sufficient_tables(state: State) -> State:
    max_retries = state["max_retries"]
    sufficient_tables = state["sufficient_info"]
    if sufficient_tables or max_retries == 0:
        return "query_gen"
    else:
        return "selector"


def continue_query_gen(state: State) -> State:
    messages = state["messages"]
    last_message = messages[-1]
    max_retries = state["max_retries"]
    if max_retries < 0:
        return "failed"
    if last_message.content.startswith("Error:"):
        task = "Fix the error in the query and rewrite the query."
        state["query"] = ""
        return "query_gen"
    else:
        return "correct_query"
    
def generate_chart(state: State) -> State:
    chart_type = state["chart_type"]
    if chart_type == "line":
        return "line_graph_tool"
    elif chart_type == "bar":
        return "bar_graph_tool"
    else:
        return "None"
    
    # Build the workflow
workflow = StateGraph(State)
workflow.add_node("react_agent", react_agent)
workflow.add_node("transform_user_question", transform_user_question)
workflow.add_node("get_context", get_context)
workflow.add_node("info_sql_database_tool_call", info_sql_database_tool_call)
workflow.add_node(
    "info_sql_database_tool", create_tool_node_with_fallback([info_sql_database_tool])
)
workflow.add_node("tables_selector", selector)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))
workflow.add_node("contextualiser", contextualiser)
workflow.add_node("sufficient_tables", sufficient_tables)
workflow.add_node("decomposer", decomposer)
workflow.add_node("query_gen", query_gen)
workflow.add_node("query_gen_with_context", query_gen_with_context)
workflow.add_node("reducer", reducer)
workflow.add_node("get_query_execution", get_query_execution)
workflow.add_node("query_execute", create_tool_node_with_fallback([db_query_tool]))
workflow.add_node("line_graph_tool", create_tool_node_with_fallback([line_graph_tool]))
workflow.add_node("bar_graph_tool", create_tool_node_with_fallback([bar_graph_tool]))
workflow.add_node("chart_generation", chart_generation)
workflow.add_node("final_answer", final_answer)

workflow.add_conditional_edges(
    START,
    query_router,
    {
      "react_agent": "react_agent",
      "transform_user_question": "transform_user_question",
    },
)
workflow.add_edge("transform_user_question", "get_context")

workflow.add_conditional_edges(
    "get_context",
    sufficient_context,
    {
        "info_sql_database_tool_call": END,
        "query_gen_with_context": "query_gen_with_context",
    },
)

workflow.add_edge("info_sql_database_tool_call", "info_sql_database_tool")
workflow.add_edge("info_sql_database_tool", "tables_selector")
workflow.add_edge("tables_selector", "get_schema_tool")
workflow.add_edge("get_schema_tool", "contextualiser")
workflow.add_edge("contextualiser", "sufficient_tables")
workflow.add_conditional_edges(
    "sufficient_tables",
    continue_sufficient_tables,
    {
        "selector": "tables_selector",
        "query_gen": "query_gen",
    },
)
#workflow.add_conditional_edges("decomposer", map_query_task, ["query_gen"])
workflow.add_edge("query_gen_with_context", "get_query_execution")
workflow.add_edge("query_gen", "get_query_execution")
#workflow.add_edge("reducer", "get_query_execution")
workflow.add_edge("get_query_execution", "query_execute")
workflow.add_conditional_edges(
    "query_execute",
    continue_query_gen,
    {
        "query_gen": "query_gen",
        "correct_query": "chart_generation",
        "failed": END,
    },
)
workflow.add_conditional_edges(
    "chart_generation",
    generate_chart,
    {
        "line_graph_tool": "line_graph_tool",
        "bar_graph_tool": "bar_graph_tool",
        "None": END,
    },
)
workflow.add_edge("line_graph_tool", "final_answer")
workflow.add_edge("bar_graph_tool", "final_answer")
workflow.add_edge("react_agent", END)
workflow.add_edge("final_answer", END)

sql_agent = workflow.compile()

