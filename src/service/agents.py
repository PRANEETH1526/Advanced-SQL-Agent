from dataclasses import dataclass

from langgraph.pregel import Pregel

from agents.sql_agent.sql_agent import sql_agent
from agents.adaptive_rag_agent.adaptive_rag_agent import adaptive_rag_agent

from service.schema import AgentInfo

DEFAULT_AGENT = "sql-agent"

@dataclass
class Agent:
    description: str
    graph: Pregel


agents: dict[str, Agent] = {
    "sql-agent": Agent(
        description="An assistant that dynamically generates and executes SQL queries on the Intellidesign database", graph=sql_agent
    ),
    "rag-agent": Agent(
        description="An assistant that retrieves information on Intellidesign data", graph=adaptive_rag_agent
    ),
}

def get_agent(agent_id: str) -> Pregel:
    return agents[agent_id].graph

def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]