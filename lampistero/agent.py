from langgraph.graph import END, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from lampistero.models import AgentState, Parameters

from lampistero.retrieval import rag_retriever

from lampistero.llm_interactions import (
    generate_query_from_history,
    rewrite,
    generate_answer,
    generate_answer_with_tools,
    generate_answer_cag,
    should_continue,
    continue_to_retrieval,
)
import logging

from lampistero.tools import (
    get_retriever_tool,
    get_question_retriever_tool,
    search_by_date,
    search_by_entity,
)

# Set up logger
logger = logging.getLogger(__name__)


def create_graph() -> CompiledStateGraph:
    # Define a new graph
    workflow = StateGraph(AgentState)

    logger.debug("Creating agent workflow graph")

    # Query and context topics selection
    workflow.add_node("question_from_history", generate_query_from_history)
    # Context retrievers
    workflow.add_node("rag_retriever", rag_retriever)
    # RAG node
    workflow.add_node("rag", generate_answer)
    # Re-writing the question
    workflow.add_node("rewrite", rewrite)

    # Set up the graph edges
    workflow.add_edge(START, "question_from_history")
    workflow.add_edge("question_from_history", "rag_retriever")
    workflow.add_edge("rag_retriever", "rag")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "rag",
        # Assess agent decision
        path=should_continue,
        path_map=["rewrite", END],
    )
    workflow.add_conditional_edges(
        "rewrite",
        path=continue_to_retrieval,
        path_map=["rag_retriever"],  # type: ignore
    )

    # Compile
    logger.debug("Compiling workflow graph")
    graph = workflow.compile()

    return graph


def create_graph_with_tools(parameters: Parameters) -> CompiledStateGraph:
    # Define a new graph
    workflow = StateGraph(AgentState)

    logger.debug("Creating agent workflow graph")

    # Query and context topics selection
    workflow.add_node("question_from_history", generate_query_from_history)

    # Answer generation
    workflow.add_node("rag", generate_answer_with_tools)

    # Context retrievers
    tool_node = ToolNode(
        tools=[
            get_retriever_tool(parameters),
            get_question_retriever_tool(parameters),
            search_by_date,
            search_by_entity,
        ]
    )
    workflow.add_node("tools", tool_node)

    # Set up the graph edges
    workflow.set_entry_point("question_from_history")
    workflow.add_edge("question_from_history", "rag")
    workflow.add_conditional_edges(
        "rag",
        tools_condition,
    )
    workflow.add_edge("tools", "rag")

    # Compile
    logger.debug("Compiling workflow graph")
    graph = workflow.compile()

    return graph


def create_graph_cag() -> CompiledStateGraph:
    # Define a new graph
    workflow = StateGraph(AgentState)

    logger.debug("Creating agent workflow graph")

    # Query and context topics selection
    workflow.add_node("question_from_history", generate_query_from_history)
    # RAG node
    workflow.add_node("cag", generate_answer_cag)

    # Set up the graph edges
    workflow.add_edge(START, "question_from_history")
    workflow.add_edge("question_from_history", "cag")
    workflow.add_edge("cag", END)

    # Compile
    logger.debug("Compiling workflow graph")
    graph = workflow.compile()

    return graph
