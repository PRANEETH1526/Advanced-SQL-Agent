import json
import logging
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.concurrency import run_in_threadpool
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.pregel import Pregel
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient

from service.agents import DEFAULT_AGENT, get_agent, get_all_agent_info
from service.memory import initialize_database
from service.settings import settings
from service.schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    FeedbackResponse,
    StreamInput,
    UserInput,
    InformationUpdateInput,
    ContextRequest,
    ContextResponse,
    StateHistoryResponse,
    InsertContextInput,
    InsertContextResponse,
    RetrieveContextInput,
    RetrieveContextResponse,
    DeleteContextInput,
    DeleteContextResponse,
)
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
    stream_output
)

from agents.vectorstore import get_collection, insert_data, delete_data, insert_context, retrieve_contexts

import asyncio

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def verify_bearer(
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET api key.", auto_error=False)),
    ],
) -> None:
    if not settings.AUTH_SECRET:
        return
    auth_secret = settings.AUTH_SECRET
    if not http_auth or http_auth.credentials != auth_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer based on settings.
    """
    try:
        async with initialize_database() as saver:
            await saver.setup()
            agents = get_all_agent_info()
            for a in agents:
                agent = get_agent(a.key)
                agent.checkpointer = saver
            yield
        print("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan)
router = APIRouter(dependencies=[Depends(verify_bearer)])

"""
@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )
"""


async def _handle_input(user_input: UserInput, agent: Pregel) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.
    """
    run_id = uuid4()
    thread_id = user_input.thread_id or str(uuid4())

    configurable = {"thread_id": thread_id, "model": None}

    if user_input.agent_config:
        if overlap := configurable.keys() & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id, thread_id


@router.post("/{agent_id}/invoke")
@router.post("/invoke")
async def invoke(user_input: UserInput, agent_id: str = DEFAULT_AGENT) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: Pregel = get_agent(agent_id)
    kwargs, run_id, thread_id = await _handle_input(user_input, agent)
    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        output.thread_id = str(thread_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput, agent_id: str = DEFAULT_AGENT
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: Pregel = get_agent(agent_id)
    kwargs, run_id, thread_id = await _handle_input(user_input, agent)

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"]
        ):
            if not isinstance(stream_event, tuple):
                continue
            stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    new_messages.extend((node, m) for m in update_messages)

            if stream_mode == "custom":
                new_messages = [event]

            for node, message in new_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                    chat_message.thread_id = str(thread_id)
                    chat_message.node = node
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
@router.post("/stream", response_class=StreamingResponse, responses=_sse_response_example())
async def stream(user_input: StreamInput, agent_id: str = DEFAULT_AGENT) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    return StreamingResponse(
        message_generator(user_input, agent_id),
        media_type="text/event-stream",
    )

@router.post(
    "/{agent_id}/update_information",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
)
async def update_information(
    payload: InformationUpdateInput,
    agent_id: str = DEFAULT_AGENT,
) -> StreamingResponse:
    """
    Fork off, overwrite 'information', then
    stream the replay + continuation via SSE.
    """
    agent: Pregel = get_agent(agent_id)
    config = {"configurable": {"thread_id": payload.thread_id}}
    loop = asyncio.get_event_loop()
    # offload the sync call to a threadpool
    try:
        history = await loop.run_in_executor(
            None,                          # default threadpool
            lambda: list(agent.get_state_history(config)),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get state history: {e}",
        )
    fork_cfg = next(
    (h.config for h in history 
     if h.next and h.next[0] == "sufficient_tables"),
        None
    )
    
    try:
        new_cfg = await agent.aupdate_state(fork_cfg, {"information": payload.information})
        thread_id = new_cfg.get("configurable", {}).get("thread_id")
        run_id = new_cfg.get("run_id")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fork/update state: {e}",
        )

    return StreamingResponse(stream_output(agent, None, new_cfg, run_id, thread_id), media_type="text/event-stream")


@router.post("/{agent_id}/save_information", response_model=ContextResponse)
async def save_information(
    req: ContextRequest,
    agent_id: str = DEFAULT_AGENT,
) -> ContextResponse:
    """
    Save the current 'information' field from the latest state for this thread.
    """
    agent: Pregel = get_agent(agent_id)
    config = RunnableConfig(configurable={"thread_id": req.thread_id})
    state = await run_in_threadpool(agent.get_state, config)
    info = state.values.get("information")
    question = state.values.get("question")
    collection = get_collection("./intellidesign.db", "sql_agent")
    if not collection:
        raise HTTPException(status_code=500, detail="Collection not found")
    insert_data(collection, f"Example Context for question: {question}\n\n{info}")
    return ContextResponse(information=info)


@router.post("/{agent_id}/delete_information", response_model=ContextResponse)
async def delete_information(
    req: ContextRequest,
    agent_id: str = DEFAULT_AGENT,
) -> ContextResponse:   
    """
    Delete the current 'information' field from the latest state for this thread.
    """
    agent: Pregel = get_agent(agent_id)
    config = RunnableConfig(configurable={"thread_id": req.thread_id})
    state = await run_in_threadpool(agent.get_state, config)
    info_id = state.values.get("information_id")
    collection = get_collection("./intellidesign.db", "sql_agent")
    if not collection:
        raise HTTPException(status_code=500, detail="Collection not found")
    # Delete the information from the collection
    delete_data(collection, info_id)
    return ContextResponse(information=f"Deleted information with ID: {info_id}")


@router.post("/{agent_id}/get_information", response_model=ContextResponse)
async def get_information(
    req: ContextRequest,
    agent_id: str = DEFAULT_AGENT,
) -> ContextResponse:
    """
    Get the current 'information' field from the latest state for this thread.
    """
    agent: Pregel = get_agent(agent_id)
    config = RunnableConfig(configurable={"thread_id": req.thread_id})
    state = await run_in_threadpool(agent.get_state, config)
    info = state.values.get("information")
    return ContextResponse(information=info)

# insert context 
@router.post("/insert_context", response_model=InsertContextResponse)
async def insert_sql_context(
    payload: InsertContextInput,
) -> InsertContextResponse:
    """
    Insert context into the database.
    """
    collection = get_collection("./intellidesign.db", "sql_context")
    if not collection:
        raise HTTPException(status_code=500, detail="Collection not found")
    id = insert_context(
        collection,
        payload.query,
        payload.context,
    )
    return InsertContextResponse(id=id)

@router.get("/retrieve_context", response_model=RetrieveContextResponse)
async def retrieve_sql_context(
    payload: RetrieveContextInput,
) -> RetrieveContextResponse:
    """
    Retrieve context from the database.
    """
    collection = get_collection("./intellidesign.db", "sql_context")
    if not collection:
        raise HTTPException(status_code=500, detail="Collection not found")
    context = retrieve_contexts(
        collection,
        payload.query,
    )
    return RetrieveContextResponse(context=context)

@router.delete("/delete_context", response_model=DeleteContextResponse)
async def delete_sql_context(
    payload: DeleteContextInput,
) -> DeleteContextResponse:
    """
    Delete context from the database.
    """
    collection = get_collection("./intellidesign.db", "sql_context")
    if not collection:
        raise HTTPException(status_code=500, detail="Collection not found")
    delete_data(
        collection,
        payload.id,
    )
    return DeleteContextResponse(id=payload.id)

@router.post("/{agent_id}/get_state_history", response_model=StateHistoryResponse)
async def get_state_history(
    req: ContextRequest,
    agent_id: str = DEFAULT_AGENT,
) -> StateHistoryResponse:
    """
    Get the state history for a given thread ID.
    """
    agent: Pregel = get_agent(agent_id)
    config = {"configurable": {"thread_id": req.thread_id}}
    loop = asyncio.get_event_loop()
    # offload the sync call to a threadpool
    history = await loop.run_in_executor(
        None,                          # default threadpool
        lambda: list(agent.get_state_history(config)),
    )
    return StateHistoryResponse(history=history) 


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


@router.post("/history")
def history(input: ChatHistoryInput) -> ChatHistory:
    """
    Get chat history.
    """
    # TODO: Hard-coding DEFAULT_AGENT here is wonky
    agent: Pregel = get_agent(DEFAULT_AGENT)
    try:
        state_snapshot = agent.get_state(
            config=RunnableConfig(
                configurable={
                    "thread_id": input.thread_id,
                }
            )
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(router)