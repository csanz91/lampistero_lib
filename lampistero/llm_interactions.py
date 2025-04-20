import logging
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

from typing import Literal
from langgraph.types import Send

from lampistero.models import AgentState
from lampistero.utils.retry import retry
from lampistero.llm_models import get_llm_model, LLMModels
from lampistero.tools import get_retriever_tool
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

    class ContextualizeQueryResponse(BaseModel):
        """Query extracted from chat history."""

        question: str = Field(description="Question to be asked")

    contextualize_q_system_prompt = """**Rol:** Eres un asistente de IA especializado en la reescritura y expansión de consultas para sistemas avanzados de búsqueda vectorial.

**Objetivo:** Reescribir la "Pregunta" proporcionada para maximizar su efectividad en la búsqueda vectorial. Esto implica:
1.  **Contextualización:** Incorporar inteligentemente información y contexto relevantes del "Historial de Chat".
2.  **Expansión y Enriquecimiento:** Ampliar la consulta con sinónimos relevantes, conceptos relacionados y detalles aclaratorios para capturar el significado semántico completo.
3.  **Preservación Semántica:** Asegurar que la intención y el significado central de la consulta *original* del usuario se conserven con precisión y sean el núcleo de la consulta reescrita.
4.  **Optimización para Búsqueda Vectorial:** Estructurar la consulta reescrita para resaltar entidades clave, conceptos y relaciones, haciéndola ideal para la coincidencia de similitud semántica.
5.  **Idioma de Salida:** La consulta final reescrita **DEBE ESTAR EN ESPAÑOL**.

**Entradas:**
1.  `Pregunta`: La última pregunta o declaración del usuario.
2.  `Historial de Chat`: Una lista cronológica de preguntas anteriores del usuario y respuestas del asistente (si las hay). Puede estar vacío (puede contener texto en cualquier idioma).

**Instrucciones:**

1.  **Analizar Consulta Actual:** Identifica el tema central, las entidades, la acción y la intención de la `Pregunta`.
2.  **Analizar Historial de Chat (Si está disponible):**
    *   Examina el `Historial de Chat` en busca de contexto directamente relevante para la `Pregunta`. Busca:
        *   **Resolución de Pronombres/Ambigüedad:** Identifica a qué se refieren los pronombres (él, ella, eso, ellos, etc.) o términos ambiguos en la consulta actual basándote en el historial.
        *   **Temas/Entidades Establecidos:** Reconoce temas, nombres o conceptos recurrentes discutidos anteriormente que proporcionan contexto.
        *   **Suposiciones Implícitas:** Comprende cualquier suposición subyacente o contexto construido durante la conversación.
        *   **Cambio de Enfoque:** Observa si la consulta actual se desvía significativamente del tema del historial reciente.
    *   **Priorizar Historial Reciente:** Da más peso a los turnos recientes de la conversación, pero considera el contexto fundamental establecido anteriormente si es relevante.
    *   **Descartar Historial Irrelevante:** *No* incluyas información del historial que no esté relacionada con la intención específica de la consulta *actual*.
3.  **Sintetizar y Reescribir (Paso Intermedio):**
    *   Combina la intención central de la `Pregunta` con el contexto *relevante* identificado del `Historial de Chat`.
    *   Reemplaza pronombres o términos ambiguos con las entidades o conceptos específicos a los que se refieren (basado en el historial).
    *   Reformula la consulta de forma natural, integrando el contexto sin problemas.
4.  **Expandir y Enriquecer (Paso Intermedio):**
    *   Añade sinónimos relevantes, formulaciones alternativas o conceptos relacionados que amplíen el alcance semántico sin cambiar el significado central (p. ej., si la consulta es sobre "entrenamiento de perros", añade términos como "comportamiento canino", "obediencia de cachorros").
    *   Si la consulta es vaga, añade especificidad derivada del contexto o interpretaciones de sentido común de la intención probable (p. ej., "Tell me about it" después de hablar de un modelo de coche específico se convierte en "Háblame de las características y especificaciones del [Nombre del Modelo de Coche]").
5.  **Formular y Refinar para Búsqueda Vectorial (Salida Final en Español):**
    *   Asegúrate de que la consulta final sea clara, descriptiva y centrada en los elementos semánticos clave.
    *   Evita relleno conversacional a menos que sea esencial para el significado.
    *   **Traduce y formula la consulta final reescrita completa y coherentemente en ESPAÑOL.** La salida debe ser una única cadena de consulta en español.

**Qué Evitar:**

*   Simplemente concatenar la consulta actual y el historial de chat.
*   Incluir detalles históricos irrelevantes.
*   Cambiar el tema o la intención fundamental de la `Pregunta`.
*   Hacer la consulta excesivamente larga con información redundante.
*   Incluir cuestiones o informacion a la pregunta no requeridos por el usuario
*   Producir cualquier salida que no sea la cadena de consulta reescrita **en español**.

**Formato de Salida:**
Produce *únicamente* la cadena de consulta final reescrita **en español**.

**Contexto:**
El contexto de las preguntas estara relacionado con la historia y cultura de un pueblo minero de España llamado Escucha."""


    # LLM
    llm = get_llm_model(
        model=state["parameters"].llm_chat_history_model,
        temperature=state["parameters"].llm_chat_history_temperature,
        max_tokens=512,
    )

    # langchain ChatModel will be automatically traced
    chat_history = "\n".join(msg.pretty_repr() for msg in chat_history)

    user_prompt = f"""
Historial de Chat:
{chat_history}
---
Pregunta:
{state["question"]}"""

    ai_msg = llm.invoke(
        [
            {"role": "system", "content": contextualize_q_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )  # type: ignore

    if not ai_msg:
        logger.error("No answer was generated.")
        raise ValueError("No answer was generated.")

    return {
        "question": ai_msg.content,
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
7. Be conversational but factual in your responses
8. Take time to present the response in an elaborated way
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
    You are an expert Q&A system that is trusted around the world. These are the rules to follow:

    1. To answer the user question you must use the content provided from the RAG tool. You can call it as many times as needed to get the information you need.
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
        LLMModels.GEMINI,
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
