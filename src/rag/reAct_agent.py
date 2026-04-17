"""
ReAct agent setup for document retrieval and question answering.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

import src.config.settings as rag_settings
from src.llms.openai import llm
from src.rag.retriever_setup import get_retriever_tool


def _build_agent():
    tools = [
        get_retriever_tool(
            search_type=rag_settings.SEARCH_TYPE,
            k=rag_settings.RETRIEVER_K,
        )
    ]

    # Simple prompt - tool calling agents don't need {tools} or {tool_names}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided tools to answer questions about the uploaded document. Always use the retriever tool to look up information before answering."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Force a tool call so the model doesn't answer from general knowledge
    # without retrieving document context.
    llm_required_tool = llm.bind_tools(tools, tool_choice="required")
    agent = create_tool_calling_agent(llm_required_tool, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True,
        return_intermediate_steps=True
    )
    return tools, agent, executor


try:
    tools, react_agent, agent_executor = _build_agent()
except Exception as e:
    print(f"[WARN] Agent bootstrap failed: {e}. Starting without retriever agent.")
    tools, react_agent, agent_executor = [], None, None


def rebuild_agent():
    """Rebuild the agent with the latest vectorstore after a document upload."""
    global tools, react_agent, agent_executor
    try:
        tools, react_agent, agent_executor = _build_agent()
        print("Agent rebuilt with updated vectorstore")
    except Exception as e:
        print(f"[WARN] Agent rebuild failed: {e}")
