import logging
import os

from langchain_core.documents.base import Document
from lampistero.models import Parameters
from lampistero.reranker.reranker import rerank_api
from lampistero.database import get_document_from_question, get_contiguous_documents

from lampistero.llm_models import (
    collection_name,
    collection_name_questions,
    vectorstore,
    vectorstore_questions,
)

logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str, parameters: Parameters, filter=None
) -> list[Document]:
    """
    Retrieve relevant documents for a given query.

    Args:
        query: Query text for retrieval
        parameters: Retrieval parameters
        filter: Optional filter to apply to retrieval

    Returns:
        List of relevant Document objects
    """

    documents_questions = []
    if parameters.enable_questions_retrieval:
        retriever_questions = vectorstore_questions.as_retriever(
            search_type=parameters.questions_retriever_params.search_type,
            search_kwargs=parameters.questions_retriever_params.search_kwargs,
        )

        documents_questions: list[Document] = retriever_questions.invoke(
            query, filter=filter
        )
        documents_questions = [
            get_document_from_question(collection_name_questions, doc.metadata["_id"])
            for doc in documents_questions
        ]

    retriever = vectorstore.as_retriever(
        search_type=parameters.retriever_params.search_type,
        search_kwargs=parameters.retriever_params.search_kwargs,
    )

    documents: list[Document] = retriever.invoke(query, filter=filter)
    documents_contents = [doc.page_content for doc in documents]
    for doc in documents_questions:
        if doc.page_content not in documents_contents:
            documents.append(doc)

    if parameters.enable_reranking:
        reranked_documents = rerank_api(query, documents, top_k=parameters.rerank_top_k)
    else:
        reranked_documents = documents

    if not parameters.enable_augmentation:
        return reranked_documents

    augmented_docs = []
    for doc in reranked_documents:
        if (
            "disable_augmentation" in doc.metadata
            and doc.metadata["disable_augmentation"]
        ):
            augmented_docs.append(doc)
            continue

        # Skip documents with specific topics
        if (
            doc.metadata["topic"] in ["anuarios", "words", "blog"]
            or "question" in doc.metadata
        ):
            augmented_docs.append(doc)
            continue

        contiguous_documents = get_contiguous_documents(
            collection_name, doc.metadata["_id"]
        )

        # Join the contiguous documents into a single document
        augmented_docs.append(
            Document(
                page_content="\n".join(
                    [doc.page_content for doc in contiguous_documents]
                ),
                metadata=doc.metadata,
            )
        )

    return augmented_docs


def rag_retriever(state):
    """
    Entry point for RAG retrieval.

    Args:
        state: Current application state with question and parameters

    Returns:
        Updated state with retrieved documents
    """
    question = state["question"]
    documents = retrieve_documents(query=question, parameters=state["parameters"])

    logger.info(f"Retrieved {len(documents)} documents.")
    return {"documents": documents}


def get_cached_context() -> str:
    try:
        data_path = os.environ["LAMPISTERO_DATA_PATH"]
    except KeyError:
        raise ValueError(
            "Environment variable 'LAMPISTERO_DATA_PATH' is not set. "
            "Please set it to the path where your data is located."
        )

    context_path = os.path.join(data_path, "context_caching.txt")

    with open(context_path, "r") as file:
        context = file.read()

    return context
