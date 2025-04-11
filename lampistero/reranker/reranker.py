import logging
import requests
from typing import List
from docker_secrets import get_docker_secrets
from langchain_core.documents.base import Document


logger = logging.getLogger(__name__)


def rerank_with_jina(query: str, documents: List[Document]) -> List[Document]:
    """Rerank documents using Jina.ai ColBERT API."""
    try:
        url = "https://api.jina.ai/v1/rerank"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_docker_secrets('JINA_API')}",
        }
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": 5,
            "documents": [doc.page_content for doc in documents],
        }

        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            raise Exception(f"Jina reranking failed: {response.text}")

        reranked = response.json()
        scored_docs = []
        results = reranked["results"]

        for result in results:
            text = result["document"]["text"]
            relevance_score = result["relevance_score"]
            scored_docs.append((text, relevance_score))

        # Sort by score and return documents
        results = [
            Document(doc)
            for doc, _ in sorted(scored_docs, key=lambda x: x[1], reverse=True)
        ]
        logger.info(f"Reranked {len(results)} documents.")
        return results
    except Exception as e:
        logger.error(f"Jina reranking failed: {str(e)}")
        raise


def rerank_api(query: str, documents: List[Document], top_k=5) -> List[Document]:
    """Rerank documents using local reranker API."""
    response = requests.post(
        "http://csm-server.lan:9070/rerank",
        json={
            "documents": [doc.page_content for doc in documents],
            "documents_ids": [doc.metadata["_id"] for doc in documents],
            "query": query,
            "top_k": top_k,
        },
    )
    if response.status_code != 200:
        raise Exception(f"Rerank failed: {response.text}")

    reranked_documents_ids = response.json().get("documents_ids", [])

    original_docs_indexed = {doc.metadata["_id"]: doc for doc in documents}
    return [original_docs_indexed[doc_id] for doc_id in reranked_documents_ids]