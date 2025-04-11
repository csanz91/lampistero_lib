import os
import sqlite3
import logging
from langchain_core.documents.base import Document
from lampistero.models import RetrievalAugmentedMode

logger = logging.getLogger(__name__)


def get_database_path() -> str:
    try:
        data_path = os.environ["LAMPISTERO_DATA_PATH"]
    except KeyError:
        raise ValueError(
            "Environment variable 'LAMPISTERO_DATA_PATH' is not set. "
            "Please set it to the path where your data is located."
        )

    db_path = os.path.join(data_path, "documents.db")

    return db_path


def get_contiguous_documents(
    collection_name: str,
    row_id: int,
    n: int = 1,
    mode: RetrievalAugmentedMode = RetrievalAugmentedMode.BOTH,
) -> list[Document]:
    """
    Retrieve contiguous documents surrounding the specified document ID.

    Args:
        collection_name: Name of the collection/table in the database
        row_id: ID of the document to use as reference point
        n: Number of documents to retrieve on each side
        mode: Direction for retrieval (PREV, NEXT, BOTH, or NONE)
        db_path: Path to the SQLite database

    Returns:
        List of Document objects
    """

    db_path = get_database_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Retrieve the document with the given row_id
    cursor.execute(
        f"SELECT * FROM {collection_name} WHERE document_id = ?",
        (row_id,),
    )
    doc = cursor.fetchone()
    if not doc:
        logger.error(f"Document with ID {row_id} not found.")
        raise ValueError(f"Document with ID {row_id} not found.")

    doc = dict(doc)
    topic = doc["topic"]
    chunk_index = doc["chunk_index"]

    chunk_index_min = chunk_index
    chunk_index_max = chunk_index
    if mode == RetrievalAugmentedMode.PREV or mode == RetrievalAugmentedMode.BOTH:
        chunk_index_min = max(chunk_index - n, 0)
    if mode == RetrievalAugmentedMode.NEXT or mode == RetrievalAugmentedMode.BOTH:
        chunk_index_max = chunk_index + n

    cursor.execute(
        f"SELECT * FROM {collection_name} WHERE chunk_index >= ? AND chunk_index <= ? AND topic = ?",
        (chunk_index_min, chunk_index_max, topic),
    )
    docs = cursor.fetchall()
    conn.close()
    if not docs:
        logger.error("No documents found in the specified range.")
        raise ValueError("No documents found in the specified range.")

    docs = [dict(doc) for doc in docs]
    current_docs = [
        Document(
            id=doc["document_id"],
            page_content=doc["content"],
            metadata={
                "file_name": doc.get("file_name", ""),
                "page": doc.get("page", ""),
                "topic": topic,
                "_id": doc["document_id"],
            },
        )
        for doc in docs
    ]

    return current_docs


def get_document_from_question(
    collection_name: str,
    row_id: int,
) -> Document:
    """Retrieve a document from the questions collection by ID."""

    db_path = get_database_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Retrieve the document with the given row_id
    cursor.execute(
        f"SELECT * FROM {collection_name} WHERE document_id = ?",
        (row_id,),
    )
    doc = cursor.fetchone()
    if not doc:
        logger.error(f"Document with ID {row_id} not found.")
        raise ValueError(f"Document with ID {row_id} not found.")

    doc = dict(doc)
    topic = doc["topic"]
    question = doc["question"]
    document_content = doc["content"]

    selected_document = Document(
        id=doc["document_id"],
        page_content=document_content,
        metadata={
            "question": question,
            "topic": topic,
            "_id": doc["document_id"],
        },
    )

    conn.close()
    return selected_document
