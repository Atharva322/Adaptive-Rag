"""
ReAct agent setup for document retrieval and question answering.
"""

import os
from src.config.settings import Config

"""
ReAct agent setup for document retrieval and question answering.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from src.llms.openai import llm
from src.rag.retriever_setup import get_retriever


def _build_agent():
    tools = [get_retriever()]

    # Simple prompt - tool calling agents don't need {tools} or {tool_names}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided tools to answer questions about the uploaded document. Always use the retriever tool to look up information before answering."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=3,
        verbose=True,
        return_intermediate_steps=True
    )
    return tools, agent, executor


tools, react_agent, agent_executor = _build_agent()


def rebuild_agent():
    """Rebuild the agent with the latest vectorstore after a document upload."""
    global tools, react_agent, agent_executor
    tools, react_agent, agent_executor = _build_agent()
    print("Agent rebuilt with updated vectorstore")
