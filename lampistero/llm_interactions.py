import logging
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

from typing import Literal
from langgraph.types import Send

from lampistero.models import AgentState
from lampistero.utils.retry import retry
from lampistero.llm_models import get_llm_model
from lampistero.tools import get_retriever_tool


logger = logging.getLogger(__name__)


@retry()
def generate_query_from_history(state: AgentState):
    """Generate a query from the chat history."""

    state["documents"] = []

    # If there are no messages history, return the state
    chat_history = state["chat_history"]
    if not chat_history:
        return state

    class ContextualizeQueryResponse(BaseModel):
        """Query extracted from chat history."""

        question: str = Field(description="Question to be asked")

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "The question should be in Spanish.",
    )

    # LLM
    llm = get_llm_model(
        model=state["parameters"].llm_chat_history_model,
        temperature=state["parameters"].llm_chat_history_temperature,
        max_tokens=512,
    ).with_structured_output(ContextualizeQueryResponse)

    # langchain ChatModel will be automatically traced
    chat_history = "\n".join(msg.pretty_repr() for msg in chat_history)

    ai_msg: ContextualizeQueryResponse = llm.invoke(
        [
            {"role": "system", "content": contextualize_q_system_prompt},
            {"role": "system", "content": chat_history},
            {"role": "user", "content": state["question"]},
        ],
    )  # type: ignore

    if not ai_msg:
        logger.error("No answer was generated.")
        raise ValueError("No answer was generated.")

    return {
        "question": ai_msg.question,
        "documents": state["documents"],
    }


@retry()
def generate_answer(state: AgentState):
    # LangChain retriever will be automatically traced
    docs = "\n".join(doc.page_content for doc in state["documents"])

    retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages(
        messages=[
            {
                "role": "system",
                "content": """You are an expert Q&A system that is trusted around the world.
Always answer the query using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
3. Answer always in Spanish
4. Try to understand and think through the context to answer the question.
5. If you don't know the answer, just say that you don't know, but think hard before that.
6. Give a binary score 'yes' or 'no' score to indicate whether you were able to respond to the question.
<context>
{docs}
</context>""",
            },
            {"role": "human", "content": "{question}"},
        ]
    )

    # Data model
    class RagResponse(BaseModel):
        """Binary score for relevance check."""

        response: str = Field(description="Response to the question")
        binary_score: str = Field(
            description="The source documents contained enough information to generate the response precisely {'yes'|'no'}"
        )

    llm = get_llm_model(
        temperature=state["parameters"].llm_answer_temperature,
        model=state["parameters"].llm_answer_model,
    ).with_structured_output(RagResponse)

    # LLM
    chain = retrieval_qa_chat_prompt | llm

    # langchain ChatModel will be automatically traced
    ai_msg: RagResponse = chain.invoke(
        {
            "docs": docs,
            "question": state["question"],
        },
    )  # type: ignore

    if not ai_msg:
        logger.error("No answer was generated.")
        raise ValueError("No answer was generated.")

    num_retrievals = state.get("num_retrievals", 0) + 1
    return {
        "answer": ai_msg.response,
        "context_is_relevant": ai_msg.binary_score.lower() == "yes",
        "num_retrievals": num_retrievals,
    }


@retry()
def rewrite(state: AgentState):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    class RewriteResponse(BaseModel):
        """New question geneated and most relevand key words."""

        new_question: str = Field(description="Formulated and improved question")
        list_relevant_key_words: list[str] = Field(
            description="List of the top 3 relevant key words associated with the question"
        )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            {
                "role": "user",
                "content": """Look at the input and try to reason about the underlying semantic intent / meaning.

Here is the initial question:
------- 
{question}
-------

Formulate an improved question and also provide a list of the top 3 relevant key words associated with the question. Always answer in Spanish.
""",
            }
        ]
    )

    llm = get_llm_model(
        temperature=state["parameters"].llm_rewrite_temperature,
        max_tokens=256,
        model=state["parameters"].llm_rewrite_model,
    ).with_structured_output(RewriteResponse)

    # Create a chain
    chain = rewrite_prompt | llm

    # Invoke the chain
    ai_msg: RewriteResponse = chain.invoke(
        {
            "question": state["question"],
        },
    )  # type: ignore

    if not ai_msg:
        logger.error("No answer was generated.")
        raise ValueError("No answer was generated.")

    return {
        "retrieval_questions": [ai_msg.new_question]
        + [", ".join(ai_msg.list_relevant_key_words)],
    }


def should_continue(state: AgentState) -> Literal["rewrite", "__end__"]:
    """Return the next node to execute."""

    if (
        not state["context_is_relevant"]
        and state["num_retrievals"] < state["parameters"].max_retrievals
    ):
        logger.info("Decision: Need to rewrite query.")
        return "rewrite"

    if not state["context_is_relevant"]:
        logger.warning(
            f"No relevant context found after {state['num_retrievals']} retrievals for the question: {state['question']}"
        )
    logger.info("Decision: Ending conversation.")
    return "__end__"


def continue_to_retrieval(state: AgentState):
    logger.debug("Continuing to retrieval.")
    return [
        Send(
            "rag_retriever",
            {
                "retrieval_question": s,
                "question": state["question"],
                "num_retrievals": state["num_retrievals"],
                "parameters": state["parameters"],
            },
        )
        for s in state["retrieval_questions"]
    ]


# def route_tools(state: AgentState) -> Literal["tools", "__end__"]:
#     """
#     Use in the conditional_edge to route to the ToolNode if the last message
#     has tool calls. Otherwise, route to the end.
#     """

#     if messages := state.get("messages", []):
#         ai_message = messages[-1]
#     else:
#         raise ValueError(f"No messages found in input state to tool_edge: {state}")
#     if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
#         return "tools"
#     return "__end__"


# def rag_tool(state: AgentState):
#     if messages := state.get("messages", []):
#         message = messages[-1]
#     else:
#         raise ValueError("No message found in input")
#     outputs = []
#     for tool_call in message.tool_calls:
#         tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
#         outputs.append(
#             ToolMessage(
#                 content=json.dumps(tool_result),
#                 name=tool_call["name"],
#                 tool_call_id=tool_call["id"],
#             )
#         )
#     return {"messages": outputs}


@retry()
def generate_answer_with_tools(state: AgentState):
    llm = get_llm_model(
        temperature=state["parameters"].llm_answer_temperature,
        model=state["parameters"].llm_answer_model,
    )
    model_with_tools = llm.bind_tools(
        [get_retriever_tool(state["parameters"])],
    )

    # Tool calling
    prompt = f"""
    You are an expert Q&A system that is trusted around the world. This are the rules to follow:

    1. To answer the user question you can only use the content provided from the RAG tool. You can call it as many times as needed to get the information you need.
    2. Always answer in spanish
    3. If you don't know the answer, just say that you don't know, but think hard before that.
    4. You can ask the user for more information if you need it.
    
    User question: {state["question"]}
    """
    documents = []
    if state["messages"]:
        documents = [
            msg.content for msg in state["messages"] if type(msg) is ToolMessage
        ]
        text = "\n\n".join(documents)  # type: ignore
        if text:
            prompt += f"Retrieved data: {text}"

    ai_msg = model_with_tools.invoke(prompt)

    return {"messages": [ai_msg], "answer": ai_msg.content, "documents": []}
