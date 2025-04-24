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
            "disable_augmentation": True,
        },
    )

    conn.close()
    return selected_document


def get_group_documents(
    collection_name: str,
    group_id: str,
) -> list[Document]:
    """
    Retrieve all documents belonging to a specific group ID from a collection.

    Args:
        collection_name: Name of the collection/table in the database.
        group_id: The group ID to filter documents by.

    Returns:
        List of Document objects belonging to the specified group ID.
    """
    db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(
            f"SELECT * FROM {collection_name} WHERE group_id = ?",
            (group_id,),
        )
        docs = cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Error querying table {collection_name}: {e}")
        # Reraise or handle appropriately, maybe check if 'group_id' column exists
        raise ValueError(f"Could not query group_id in table {collection_name}. Does the column exist?") from e
    finally:
        conn.close()

    if not docs:
        logger.warning(f"No documents found for group_id {group_id} in collection {collection_name}.")
        return [] # Return empty list if no documents found

    documents = []
    for row in docs:
        doc_dict = dict(row)

        document = Document(
            id=doc_dict["document_id"],
            page_content=doc_dict["content"],
            metadata={"_id": doc_dict["document_id"]},
        )
        documents.append(document)

    return documents


def get_documents_by_date(
    collection_name: str,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
) -> list[Document]:
    """
    Retrieve documents from a collection based on date components.

    Filters documents by querying an associated '_dates' table based on year,
    month, and day. Only non-None date components are used for filtering.

    Args:
        collection_name: Name of the main collection/table in the database.
        year: The year to filter by (optional).
        month: The month to filter by (optional).
        day: The day to filter by (optional).

    Returns:
        List of Document objects matching the date criteria.

    Raises:
        ValueError: If the date table or main table cannot be queried,
                    or if no documents are found for the given criteria.
    """
    db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    date_table_name = f"{collection_name}_dates"
    where_clauses = []
    params = []

    if year is not None:
        where_clauses.append("year = ?")
        params.append(year)
    if month is not None:
        where_clauses.append("month = ?")
        params.append(month)
    if day is not None:
        where_clauses.append("day = ?")
        params.append(day)

    if not where_clauses:
        logger.warning("No date components provided for filtering.")
        conn.close()
        return [] # Return empty list if no date filters are specified

    where_sql = " AND ".join(where_clauses)
    query_dates = f"SELECT document_id FROM {date_table_name} WHERE {where_sql}"

    try:
        cursor.execute(query_dates, tuple(params))
        date_rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Error querying date table {date_table_name}: {e}")
        conn.close()
        raise ValueError(f"Could not query date table {date_table_name}. Does it exist with year, month, day columns?") from e

    if not date_rows:
        logger.warning(f"No document IDs found in {date_table_name} for the given date criteria.")
        conn.close()
        return []

    document_ids = [row["document_id"] for row in date_rows]
    placeholders = ",".join("?" * len(document_ids))
    query_docs = f"SELECT * FROM {collection_name} WHERE document_id IN ({placeholders})"

    try:
        cursor.execute(query_docs, tuple(document_ids))
        doc_rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Error querying main table {collection_name}: {e}")
        conn.close()
        raise ValueError(f"Could not query main table {collection_name}.") from e
    finally:
        conn.close()

    if not doc_rows:
        # This case might be less common if IDs were found, but good to handle
        logger.warning(f"No documents found in {collection_name} for the retrieved document IDs.")
        return []

    documents = []
    for row in doc_rows:
        doc_dict = dict(row)

        document = Document(
            id=doc_dict["document_id"],
            page_content=doc_dict.get("content", ""),
            metadata={"_id": doc_dict["document_id"]},
        )
        documents.append(document)

    print(f"Found {len(documents)} documents matching the date criteria.")
    return documents


def get_documents_by_entity(
    collection_name: str,
    entity_value: str,
) -> list[Document]:
    """
    Retrieve documents from a collection based on an entity value.

    Filters documents by querying an associated '_entities' table for a
    case-insensitive match of the entity_value.

    Args:
        collection_name: Name of the main collection/table in the database.
        entity_value: The entity value to search for (case-insensitive).

    Returns:
        List of Document objects associated with the entity value.

    Raises:
        ValueError: If the entity table or main table cannot be queried,
                    or if no documents are found for the given entity.
    """
    db_path = get_database_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    entity_table_name = f"{collection_name}_entities"
    query_entities = f"SELECT document_id FROM {entity_table_name} WHERE LOWER(entity_value) = LOWER(?)"

    try:
        cursor.execute(query_entities, (entity_value,))
        entity_rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Error querying entity table {entity_table_name}: {e}")
        conn.close()
        raise ValueError(f"Could not query entity table {entity_table_name}. Does it exist with an 'entity_value' column?") from e

    if not entity_rows:
        logger.warning(f"No document IDs found in {entity_table_name} for entity_value '{entity_value}'.")
        conn.close()
        return []

    document_ids = [row["document_id"] for row in entity_rows]
    placeholders = ",".join("?" * len(document_ids))
    query_docs = f"SELECT * FROM {collection_name} WHERE document_id IN ({placeholders})"

    try:
        cursor.execute(query_docs, tuple(document_ids))
        doc_rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.error(f"Error querying main table {collection_name}: {e}")
        conn.close()
        raise ValueError(f"Could not query main table {collection_name}.") from e
    finally:
        conn.close()

    if not doc_rows:
        logger.warning(f"No documents found in {collection_name} for the retrieved document IDs based on entity '{entity_value}'.")
        return []

    documents = []
    for row in doc_rows:
        doc_dict = dict(row)

        document = Document(
            id=doc_dict["document_id"],
            page_content=doc_dict.get("content", ""),
            metadata={"_id": doc_dict["document_id"]},
        )
        documents.append(document)

    return documents
