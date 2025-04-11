# Lampistero Graph

Graph building functionality for Lampistero using LangGraph.

## Installation

You can install the package using pip:

```bash
pip install -e /path/to/lampistero-graph
```

Or with uv:

```bash
uv pip install -e /path/to/lampistero-graph
```

## Usage

```python
from lampistero_graph import GraphBuilder, Parameters
from lampistero_graph.models import AgentState, LLMModels

# Create a graph with your custom node functions
graph = GraphBuilder.create_graph(
    rag_retriever=your_retriever_function,
    generate_query_from_history=your_query_generator,
    rewrite=your_rewrite_function,
    generate_answer=your_answer_generator,
    should_continue=your_continue_checker,
    continue_to_retrieval=your_retrieval_checker
)

# Or create a graph with tools
parameters = Parameters(
    model=LLMModels.GEMINI,
    temperature=0.1,
    max_tokens=4096
)

graph_with_tools = GraphBuilder.create_graph_with_tools(
    generate_query_from_history=your_query_generator,
    generate_answer_with_tools=your_answer_generator_with_tools,
    get_retriever_tool=your_retriever_tool_getter,
    parameters=parameters
)
```