from lampistero.utils.retry import retry
from lampistero.reranker.reranker import rerank_with_jina, rerank_api
from lampistero.llm_interactions import (
    generate_query_from_history,
    generate_answer,
    generate_answer_with_tools,
    rewrite,
    should_continue,
    continue_to_retrieval,
)
from lampistero.retrieval import (
    retrieve_documents,
    rag_retriever,
)
from lampistero.database import (
    get_document_from_question,
    get_contiguous_documents,
)

import logging

logger = logging.getLogger(__name__)

# Re-export the functions for backward compatibility
__all__ = [
    "retry",
    "rerank_with_jina",
    "rerank_api",
    "generate_query_from_history",
    "generate_answer",
    "generate_answer_with_tools",
    "rewrite",
    "should_continue",
    "continue_to_retrieval",
    "retrieve_documents",
    "rag_retriever",
    "get_document_from_question",
    "get_contiguous_documents",
]
