import logging
import pytest
from lampistero.agent import create_graph, create_graph_with_tools
from lampistero.models import AgentState, Parameters


logger = logging.getLogger()

def test_create_graph():
    graph = create_graph()
    assert graph is not None
    assert callable(graph.invoke)

def test_create_graph_with_tools():
    parameters = Parameters()
    graph = create_graph_with_tools(parameters)
    assert graph is not None
    assert callable(graph.invoke)
