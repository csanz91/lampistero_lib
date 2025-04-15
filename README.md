# Lampistero

Lampistero is a Python library for building graph-based retrieval-augmented generation (RAG) pipelines, leveraging modern LLMs and vector databases. It provides tools for document retrieval, reranking, and LLM-based answer generation, with flexible configuration and support for hybrid search.

## Features
- Graph-based orchestration for RAG workflows
- Integration with LangChain, Qdrant, OpenAI, Google Gemini, DeepSeek, and more
- Customizable retriever and reranker tools
- Support for hybrid dense/sparse retrieval
- Pydantic-based configuration models
- Utilities for retry logic and database access

## Installation

Install via pip (requires Python 3.12+):

```bash
pip install lampistero
```

Or for development:

```bash
git clone <repo-url>
cd lampistero
pip install -e .[dev]
```

## Usage Example

```python
from lampistero import create_graph, Parameters

params = Parameters()
graph = create_graph(parameters=params)
result = graph.invoke({"question": "What is retrieval-augmented generation?"})
print(result)
```

## Configuration

You can customize retrieval, reranking, and LLM models via the `Parameters` class. See `lampistero.models.Parameters` for all options.

## Testing

Run tests with:

```bash
pytest
```

## License

See the LICENSE file for details.
