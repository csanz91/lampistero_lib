import logging
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

from typing import Literal
from langgraph.types import Send

from lampistero.models import AgentState, DateModel
from lampistero.utils.retry import retry
from lampistero.llm_models import get_llm_model, LLMModels
from lampistero.tools import (
    get_retriever_tool,
    get_question_retriever_tool,
    search_by_date,
    search_by_entity,
)

from lampistero.retrieval.document_retrieval import get_cached_context


logger = logging.getLogger(__name__)


@retry()
def generate_query_from_history(state: AgentState):
    """Generate a query from the chat history."""

    state["documents"] = []

    # Fix the question so the town name is properly recognized
    state["question"] = state["question"].replace("escucha", "Escucha")

    # # If there are no messages history, return the state
    chat_history = state["chat_history"]
    # if not chat_history:
    #     return state

    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, 
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed.
The question should be in Spanish.

**Output format:**
question: user question, fix any spelling mistakes and make it more clear. For example: 'en que ano se fundo la empresa de escucha' -> '¿En qué año se fundó la empresa ubicada en Escucha?'
dates: list of dates related to the query. Each date can have a year, month and day components. For example: 'en el año 1978 se fundo la empresa'. Year: 1978, month: None, day: None
entities: list of entities related to the query. Each entity can be a person, organization, location, etc. For example: 'la empresa Lampistero se fundo en Escucha'. Entities: ['Lampistero', 'Escucha']
decomposed_questions: Break down the original query into a list of simpler sub-queries. Each sub-query should focus on a different aspect of the original question.

**Contexto:**
El contexto de las preguntas estara relacionado con la historia y cultura de un pueblo minero de España llamado Escucha. Valdeconejos es un pueblo cercano a Escucha. El contexto puede incluir eventos históricos, personajes importantes, tradiciones, leyendas, etc. El contexto no debe ser mencionado en la respuesta."""

    # Data model
    class LLMResponse(BaseModel):
        """Binary score for relevance check."""

        question: str = Field(description="Improved question")
        dates: list[DateModel] = Field(
            description="List of dates related to the query", default_factory=list
        )
        entities: list[str] = Field(
            description="List of entities related to the query", default_factory=list
        )
        decomposed_questions: list[str] = Field(
            description="List of decomposed queries", default_factory=list
        )

    # LLM
    llm = get_llm_model(
        model=state["parameters"].llm_chat_history_model,
        temperature=state["parameters"].llm_chat_history_temperature,
        max_tokens=512,
    ).with_structured_output(LLMResponse)

    # langchain ChatModel will be automatically traced
    chat_history = "\n".join(msg.pretty_repr() for msg in chat_history)

    user_prompt = f"""
Historial de Chat:
{chat_history}
---
Pregunta:
{state["question"]}"""

    ai_msg: LLMResponse = llm.invoke(
        [
            {"role": "system", "content": contextualize_q_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )  # type: ignore

    if not ai_msg:
        logger.error("No answer was generated.")
        raise ValueError("No answer was generated.")

    return {
        "question": ai_msg.question,
        "retrieval_dates": ai_msg.dates,
        "retrieval_entities": ai_msg.entities,
        "documents": state["documents"],
        "decomposed_questions": ai_msg.decomposed_questions,
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
7. Be conversational but factual in your responses
8. Take time to present the response in an elaborated way
9. Use bullet points if needed to present the answer
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
        [
            get_retriever_tool(state["parameters"]),
            get_question_retriever_tool(state["parameters"]),
            search_by_date,
            search_by_entity,
        ],
    )

    # Tool calling
    prompt = f"""
    You are an expert Q&A system that is trusted around the world. These are the rules to follow:

    1. To answer the user question you must use the content provided from the tools. You can call it as many times as needed to get the information you need.
    2. Always answer in spanish
    3. If after trying to understand the documents you don't know how to answer the question, just say that you don't know, but think hard before that.
    
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


@retry()
def generate_answer_cag(state: AgentState):
    # LangChain retriever will be automatically traced

    context = get_cached_context()

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
6. Be conversational but factual in your responses
7. Take time to present the response in an elaborated way
<context>
{context}
</context>""",
            },
            {"role": "human", "content": "{question}"},
        ]
    )

    assert state["parameters"].llm_answer_model in [
        LLMModels.GEMINI_2_0_FLASH,
        LLMModels.GEMINI_2_5_PRO,
        LLMModels.GEMINI_2_0_FLASH_THINKING,
    ], "Only GEMINI and GEMINI_2_5_PRO models are supported for CAG"

    llm = get_llm_model(
        temperature=state["parameters"].llm_answer_temperature,
        model=state["parameters"].llm_answer_model,
    )

    # LLM
    chain = retrieval_qa_chat_prompt | llm

    # langchain ChatModel will be automatically traced
    ai_msg = chain.invoke(
        {
            "context": context,
            "question": state["question"],
        },
    )  # type: ignore

    if not ai_msg:
        logger.error("No answer was generated.")
        raise ValueError("No answer was generated.")

    return {
        "answer": ai_msg.content,
    }
