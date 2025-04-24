import logging
import os
import json
import time
from typing import Optional

from langchain_core.documents.base import Document
from lampistero.models import Parameters, DateModel
from lampistero.reranker.reranker import rerank_api
from lampistero.database import (
    get_document_from_question,
    get_contiguous_documents,
    get_group_documents,
    get_documents_by_date,
    get_documents_by_entity,
)

from lampistero.llm_models import (
    collection_name,
    collection_name_questions,
    vectorstore,
    vectorstore_2,
    vectorstore_questions,
)

logger = logging.getLogger(__name__)


def _extend_documents_with_groups(documents: list[Document]) -> list[Document]:
    """Helper function to extend documents with their groups."""
    extended_documents_map = {}
    for doc in documents:
        # Use document ID as key to avoid duplicates from group expansion
        if doc.metadata["_id"] not in extended_documents_map:
            extended_documents_map[doc.metadata["_id"]] = doc

        group_id: str = doc.metadata.get("group_id", "")
        if group_id:
            group_documents = get_group_documents(collection_name, group_id)
            for group_doc in group_documents:
                if group_doc.metadata["_id"] not in extended_documents_map:
                    extended_documents_map[group_doc.metadata["_id"]] = group_doc
    return list(extended_documents_map.values())


def _augment_documents(documents: list[Document]) -> list[Document]:
    """Helper function to augment documents with contiguous context."""
    augmented_docs = []
    processed_ids = (
        set()
    )  # Keep track of processed document IDs to avoid redundant augmentation

    for doc in documents:
        doc_id = doc.metadata["_id"]
        if doc_id in processed_ids:
            continue

        if (
            "disable_augmentation" in doc.metadata
            and doc.metadata["disable_augmentation"]
        ):
            augmented_docs.append(doc)
            processed_ids.add(doc_id)
            continue

        try:
            contiguous_documents = get_contiguous_documents(collection_name, doc_id)
            # Add all IDs from contiguous documents to processed_ids
            for cont_doc in contiguous_documents:
                processed_ids.add(cont_doc.metadata["_id"])

            # Join the contiguous documents into a single document
            augmented_docs.append(
                Document(
                    page_content="\n".join(
                        [cd.page_content for cd in contiguous_documents]
                    ),
                    metadata=doc.metadata,  # Use metadata from the original trigger doc
                )
            )
        except ValueError as e:
            logger.warning(
                f"Could not get contiguous documents for doc ID {doc_id}: {e}"
            )
            # Append the original document if augmentation fails
            augmented_docs.append(doc)
            processed_ids.add(doc_id)

    return augmented_docs


def retrieve_documents(
    query: str,
    parameters: Parameters,
    filter=None,
    dates: Optional[list[DateModel]] = None,
    entities: Optional[list[str]] = None,
    decomposed_questions: Optional[list[str]] = None,
) -> list[Document]:
    """
    Retrieve relevant documents for a given query, optionally filtering by dates and entities,
    and log statistics to a JSON file.

    Args:
        query: Query text for retrieval
        parameters: Retrieval parameters
        filter: Optional filter to apply to vector retrieval
        dates: Optional list of DateModel objects to filter by.
        entities: Optional list of entity strings to filter by.

    Returns:
        list of relevant Document objects, processed according to parameters.
    """
    stats: dict[str, str | int | list[str] | float] = { # Allow float for time
        "query": query,
        "decomposed_questions": decomposed_questions or [],
    }  # Initialize stats dict
    retrieved_docs_map = {}  # Use a dict with doc_id as key to handle overlaps
    initial_vector_count = 0

    # 1. Vector Search Retrieval (Main first, then Questions)
    start_time = time.time() # Start timer for main retrieval

    # Main retrieval
    retriever = vectorstore.as_retriever(
        search_type=parameters.retriever_params.search_type,
        search_kwargs=parameters.retriever_params.search_kwargs,
    )

    queries = [query]
    if decomposed_questions:
        queries.extend(decomposed_questions)

    for q in queries:
        try:
            retrieved_docs: list[Document] = retriever.invoke(q, filter=filter)
            logger.debug(
                f"Retrieved {len(retrieved_docs)} documents from main vector store for query '{q}'."
            )
            initial_vector_count += len(retrieved_docs)
            for doc in retrieved_docs:
                if doc.metadata["_id"] not in retrieved_docs_map:
                    retrieved_docs_map[doc.metadata["_id"]] = doc
        except Exception as e:
            logger.error(
                f"Error retrieving documents from main vector store: {e}, query: {q}"
            )
    end_time = time.time() # End timer for main retrieval
    logger.info(
        f"Found {len(retrieved_docs_map)} documents after main retrieval."
    )  # Log after main
    stats["main_retrieval_count"] = len(retrieved_docs_map)  # Store stat
    stats["main_retrieval_time_s"] = end_time - start_time # Store time

    # Vectorstore 2 Retrieval
    start_time = time.time() # Start timer for vs2 retrieval
    count_before_vs2 = len(retrieved_docs_map)
    retriever_vs2 = vectorstore_2.as_retriever(
        search_type=parameters.retriever_params.search_type,  # Assuming same params for now
        search_kwargs=parameters.retriever_params.search_kwargs,
    )
    for q in queries:  # Use the same queries list (original + decomposed)
        try:
            retrieved_docs_vs2: list[Document] = retriever_vs2.invoke(q, filter=filter)
            logger.debug(
                f"Retrieved {len(retrieved_docs_vs2)} documents from vectorstore_2 for query '{q}'."
            )
            # initial_vector_count += len(retrieved_docs_vs2) # Decide if this should count towards initial
            for doc in retrieved_docs_vs2:
                if doc.metadata["_id"] not in retrieved_docs_map:
                    retrieved_docs_map[doc.metadata["_id"]] = doc
        except Exception as e:
            logger.error(
                f"Error retrieving documents from vectorstore_2: {e}, query: {q}"
            )
    end_time = time.time() # End timer for vs2 retrieval
    unique_from_vs2 = len(retrieved_docs_map) - count_before_vs2
    logger.info(
        f"Found {unique_from_vs2} new unique documents from vectorstore_2 retrieval. Total unique: {len(retrieved_docs_map)}"
    )
    stats["vectorstore_2_added"] = unique_from_vs2  # Store stat
    stats["vectorstore_2_time_s"] = end_time - start_time # Store time

    # Questions Vector Store Retrieval
    start_time = time.time() # Start timer for questions retrieval
    count_after_main = len(retrieved_docs_map)  # Store count after main AND vs2
    documents_questions = []
    if parameters.enable_questions_retrieval:
        retriever_questions = vectorstore_questions.as_retriever(
            search_type=parameters.questions_retriever_params.search_type,
            search_kwargs=parameters.questions_retriever_params.search_kwargs,
        )
        try:
            retrieved_question_docs: list[Document] = retriever_questions.invoke(
                query, filter=filter
            )
            documents_questions = [
                get_document_from_question(
                    collection_name_questions, doc.metadata["_id"]
                )
                for doc in retrieved_question_docs
            ]
            logger.debug(
                f"Retrieved {len(documents_questions)} documents from questions vector store."
            )
            initial_vector_count += len(documents_questions)  # Add to initial count
            for doc in documents_questions:
                if doc.metadata["_id"] not in retrieved_docs_map:
                    retrieved_docs_map[doc.metadata["_id"]] = doc
        except Exception as e:
            logger.error(f"Error retrieving documents from questions vector store: {e}")
    else:
        logger.info("Skipped questions retrieval (disabled).")

    end_time = time.time() # End timer for questions retrieval
    # Log unique documents added by questions retrieval
    unique_from_questions = len(retrieved_docs_map) - count_after_main
    logger.info(
        f"Found {unique_from_questions} new unique documents from questions retrieval."
    )
    stats["questions_added"] = unique_from_questions  # Store stat
    stats["questions_retrieval_time_s"] = end_time - start_time # Store time

    # 2. Date-based Retrieval
    start_time = time.time() # Start timer for date retrieval
    if dates:
        MAX_DATE_DOCS = 20
        count_before_dates = len(retrieved_docs_map)
        for date_filter in dates:
            try:
                date_docs = get_documents_by_date(
                    collection_name=collection_name,
                    year=date_filter.year,
                    month=date_filter.month,
                    day=date_filter.day,
                )
                if len(date_docs) > MAX_DATE_DOCS:
                    logger.info(
                        f"Found {len(date_docs)} documents for date {date_filter}. Discarding them..."
                    )
                    continue

                for doc in date_docs:
                    if doc.metadata["_id"] not in retrieved_docs_map:
                        retrieved_docs_map[doc.metadata["_id"]] = doc
            except ValueError as e:
                logger.warning(
                    f"Could not retrieve documents for date {date_filter}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error retrieving documents for date {date_filter}: {e}"
                )
        unique_from_dates = (
            len(retrieved_docs_map) - count_before_dates
        )  # Calculate unique
        logger.info(
            f"Found {unique_from_dates} new documents from date search. Total unique: {len(retrieved_docs_map)}"
        )
        stats["date_added"] = unique_from_dates  # Store stat
    else:
        logger.info("Skipped date retrieval (no dates provided).")
        stats["date_added"] = 0 # Explicitly set to 0 if skipped
    end_time = time.time() # End timer for date retrieval
    stats["date_retrieval_time_s"] = end_time - start_time # Store time

    # 3. Entity-based Retrieval
    start_time = time.time() # Start timer for entity retrieval
    if entities:
        MAX_ENTITY_DOCS = 20
        count_before_entities = len(retrieved_docs_map)
        for entity_value in entities:
            try:
                entity_docs = get_documents_by_entity(
                    collection_name=collection_name, entity_value=entity_value
                )

                if len(entity_docs) > MAX_ENTITY_DOCS:
                    logger.info(
                        f"Found {len(entity_docs)} documents for entity '{entity_value}'. Discarting them..."
                    )
                    continue

                for doc in entity_docs:
                    if doc.metadata["_id"] not in retrieved_docs_map:
                        retrieved_docs_map[doc.metadata["_id"]] = doc
            except ValueError as e:
                logger.warning(
                    f"Could not retrieve documents for entity '{entity_value}': {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error retrieving documents for entity '{entity_value}': {e}"
                )
        unique_from_entities = (
            len(retrieved_docs_map) - count_before_entities
        )  # Calculate unique
        logger.info(
            f"Found {unique_from_entities} new documents from entity search. Total unique: {len(retrieved_docs_map)}"
        )
        stats["entity_added"] = unique_from_entities  # Store stat
    else:
        logger.info("Skipped entity retrieval (no entities provided).")
        stats["entity_added"] = 0 # Explicitly set to 0 if skipped
    end_time = time.time() # End timer for entity retrieval
    stats["entity_retrieval_time_s"] = end_time - start_time # Store time

    # Initial combined list of unique documents
    combined_documents = list(retrieved_docs_map.values())

    if not combined_documents:
        logger.warning("No documents found after initial retrieval stages.")
        return []
    logger.info(f"Total unique documents before extension: {len(combined_documents)}")
    stats["total_unique_before_extension"] = len(combined_documents)  # Store stat

    # 4. Extend with Group Documents (Applied to the combined list)
    start_time = time.time() # Start timer for group extension
    extended_documents = _extend_documents_with_groups(combined_documents)
    end_time = time.time() # End timer for group extension
    logger.info(f"Total documents after group extension: {len(extended_documents)}")
    stats["total_after_extension"] = len(extended_documents)  # Store stat
    stats["group_extension_time_s"] = end_time - start_time # Store time

    # 5. Reranking (Applied after extension)
    start_time = time.time() # Start timer for reranking
    if (
        parameters.enable_reranking and len(extended_documents) > 1
    ):  # Reranking needs > 1 doc
        try:
            processed_documents = rerank_api(
                query, extended_documents, top_k=parameters.rerank_top_k
            )
            logger.info(
                f"Reranked documents, retaining top {len(processed_documents)}."
            )
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Proceeding without reranking.")
            processed_documents = extended_documents  # Fallback to extended list
            logger.info("Skipped reranking due to error.")
    else:
        processed_documents = (
            extended_documents  # Skip reranking if disabled or not enough docs
        )
        if len(extended_documents) <= 1:
            logger.info("Skipped reranking (<= 1 document).")
        else:
            logger.info("Skipped reranking (disabled).")
    end_time = time.time() # End timer for reranking
    stats["total_after_rerank"] = len(
        processed_documents
    )  # Store stat (will be same as extension if skipped)
    stats["rerank_time_s"] = end_time - start_time # Store time

    # 6. Augmentation (Applied after reranking or extension)
    start_time = time.time() # Start timer for augmentation
    if parameters.enable_augmentation:
        final_documents = _augment_documents(processed_documents)
        logger.info(f"Total documents after augmentation: {len(final_documents)}")
    else:
        final_documents = processed_documents  # Skip augmentation if disabled
        logger.info("Skipped augmentation (disabled).")
    end_time = time.time() # End timer for augmentation
    stats["final_count"] = len(final_documents)  # Store final stat
    stats["augmentation_time_s"] = end_time - start_time # Store time

    logger.info(f"Returning {len(final_documents)} final documents.")

    # --- Log stats to JSON file ---
    try:
        data_path = os.environ["LAMPISTERO_DATA_PATH"]
        stats_file_path = os.path.join(data_path, "retrieval_stats.json")
        all_stats = []
        if os.path.exists(stats_file_path):
            try:
                with open(stats_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:  # Check if file is not empty
                        all_stats = json.loads(content)
                        if not isinstance(all_stats, list):  # Ensure it's a list
                            logger.warning(
                                f"Stats file {stats_file_path} does not contain a list. Reinitializing."
                            )
                            all_stats = []
                    else:
                        all_stats = []  # File is empty, start new list
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Could not decode JSON from stats file {stats_file_path}: {e}. Starting new list."
                )
                all_stats = []
            except FileNotFoundError:
                logger.info(
                    f"Stats file {stats_file_path} not found. Creating new file."
                )
                all_stats = []

        all_stats.append(stats)

        with open(stats_file_path, "w", encoding="utf-8") as f:
            json.dump(all_stats, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully appended stats to {stats_file_path}")
    except KeyError:
        logger.error(
            "LAMPISTERO_DATA_PATH environment variable not set. Cannot save stats."
        )
    except Exception as e:
        logger.error(f"Failed to write stats to file: {e}")
    # --- End log stats ---

    return final_documents


def rag_retriever(state):
    """
    Entry point for RAG retrieval.

    Args:
        state: Current application state with question and parameters

    Returns:
        Updated state with retrieved documents
    """
    question = state["question"]
    dates = state["retrieval_dates"]
    entities = state["retrieval_entities"]
    decomposed_questions = state["decomposed_questions"]

    logger.info(f"Retrieving documents for question: {question}")
    logger.info(f"Entities: {entities}")
    logger.info(f"Dates: {dates}")
    logger.info(f"Decomposed questions: {decomposed_questions}")

    documents = retrieve_documents(
        query=question,
        parameters=state["parameters"],
        entities=entities,
        dates=dates,
        decomposed_questions=decomposed_questions,
    )

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
