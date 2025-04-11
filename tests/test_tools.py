import pytest
from lampistero.tools.tools import get_retriever_tool
from lampistero.models import Parameters

def test_get_retriever_tool():
    parameters = Parameters()
    tool = get_retriever_tool(parameters)
    assert tool is not None
    assert tool.name == "RAG"