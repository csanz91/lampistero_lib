from langchain.tools.retriever import create_retriever_tool
from lampistero.models import Parameters
from lampistero.llm_models import vectorstore

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
