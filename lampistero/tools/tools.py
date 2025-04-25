import logging

from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool

from lampistero.models import Parameters, DateModel
from lampistero.llm_models import (
    vectorstore,
    vectorstore_questions,
    collection_name,
)
from lampistero.database import (
    get_documents_by_date,
    get_documents_by_entity,
)


logger = logging.getLogger(__name__)


def get_retriever_tool(parameters: Parameters):
    retriever = vectorstore.as_retriever(
        search_type=parameters.retriever_params.search_type,
        search_kwargs=parameters.retriever_params.search_kwargs,
    )

    rag_tool = create_retriever_tool(
        retriever=retriever,
        name="RAG",
        description="A tool to retrieve documents to be able to response the user query.",
    )

    return rag_tool


@tool
def search_by_date(dates: list[DateModel]) -> str:
    """Search documents by a list of specific dates."""
    if not dates:
        return "No dates provided for search."

    logger.info(f"Searching documents by dates: {dates}")
    retrieved_docs_map = {}
    try:
        for date_filter in dates:
            date_docs = get_documents_by_date(
                collection_name=collection_name,
                year=date_filter.year,
                month=date_filter.month,
                day=date_filter.day,
            )
            logger.info(f"Found {len(date_docs)} documents for date {date_filter}.")
            for doc in date_docs:
                if doc.metadata["_id"] not in retrieved_docs_map:
                    retrieved_docs_map[doc.metadata["_id"]] = doc
    except ValueError as e:
        logger.warning(f"Could not retrieve documents for dates: {e}")
        # Continue processing other dates if possible, but log the error
    except Exception as e:
        logger.error(f"Unexpected error retrieving documents for dates: {e}")
        # Depending on desired behavior, might return partial results or empty string

    if not retrieved_docs_map:
        return "No documents found for the specified dates."

    return "\n\n".join([doc.page_content for doc in retrieved_docs_map.values()])


@tool
def search_by_entity(entities: list[str]) -> str:
    """Search documents by a list of specific entity values."""
    if not entities:
        return "No entities provided for search."

    logger.info(f"Searching documents by entities: {entities}")
    retrieved_docs_map = {}
    try:
        for entity_value in entities:
            entity_docs = get_documents_by_entity(
                collection_name=collection_name, entity_value=entity_value
            )
            logger.info(
                f"Found {len(entity_docs)} documents for entity '{entity_value}'."
            )
            for doc in entity_docs:
                if doc.metadata["_id"] not in retrieved_docs_map:
                    retrieved_docs_map[doc.metadata["_id"]] = doc
    except ValueError as e:
        logger.warning(f"Could not retrieve documents for entities: {e}")
    except Exception as e:
        logger.error(f"Unexpected error retrieving documents for entities: {e}")

    if not retrieved_docs_map:
        return "No documents found for the specified entities."

    return "\n\n".join([doc.page_content for doc in retrieved_docs_map.values()])


def get_question_retriever_tool(parameters: Parameters):
    retriever = vectorstore_questions.as_retriever(
        search_type=parameters.retriever_params.search_type,
        search_kwargs=parameters.retriever_params.search_kwargs,
    )

    tool = create_retriever_tool(
        retriever=retriever,
        name="Question Similarity Search",
        description="Search documents based on similarity to existing questions.",
    )

    return tool
