from enum import Enum
from typing import Optional, Any, Annotated
from pydantic import BaseModel, Field
from operator import add
from langchain_core.messages import BaseMessage, ToolMessage
import logging

from typing_extensions import TypedDict

from langchain_core.documents.base import Document

# Set up logger
logger = logging.getLogger(__name__)


class ModelCapability:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    HIGH_CONTEXT = "high_context"


class LLMModels(str, Enum):
    GEMINI_2_0_FLASH = "google_genai:gemini-2.0-flash-001"
    GEMINI_2_0_FLASH_THINKING = "google_genai:gemini-2.0-flash-thinking-exp-01-21"
    GEMINI_2_5_PRO = "google_genai:gemini-2.5-pro-exp-03-25"
    GEMINI_2_5_FLASH = "google_genai:gemini-2.5-flash-preview-04-17"
    GEMINI_FLASH_2_0_LITE = "google_genai:gemini-2.0-flash-lite"
    GEMINI_PRO = "google/gemini-2.0-pro-exp-02-05:free"
    DEEPSEEK = "deepseek:deepseek-chat"
    DEEPSEEK_REASONER = "deepseek/deepseek-r1:free"
    GPT_4_1_MINI = "openai:gpt-4.1-mini"
    GPT_4_1 = "openai:gpt-4.1"
    GPT_4O = "openai:gpt-4o"
    O4_MINI = "openai:o4-mini"
    O3 = "openai:o3"
    O3_MINI = "openai:o3-mini"
    AGENT = "lampistero-agent"
    CAG = "lampistero-cag"
    GROK_3_MINI = "x-ai/grok-3-mini-beta"
    MAI_DS_R1 = "microsoft/MAI-DS-R1"


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    stop: str | list[str] | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: dict[str, int]


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: str | list[str] | None = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: dict[str, int]


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Optional[str] = "float"
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict[str, int]


def add_documents(existing_docs, new_docs):
    """Custom function to add documents.
    Check if the document exists in the list and if not, add it.
    """
    for new_doc in new_docs:
        if new_doc not in existing_docs:
            existing_docs.append(new_doc)
    return existing_docs


class RetrieverParams(BaseModel):
    search_kwargs: dict = Field(default={}, description="Search parameters")
    search_type: str = "similarity"


class Parameters(BaseModel):
    enable_reranking: bool = Field(
        default=True, description="Whether to rerank the documents or not"
    )
    rerank_top_k: int = Field(default=5, description="Number of documents to rerank")
    llm_answer_model: LLMModels = Field(
        default=LLMModels.GEMINI_2_0_FLASH, description="The model to use for LLM answers"
    )
    llm_answer_temperature: float = Field(
        default=1.0, description="Temperature for LLM answers"
    )
    llm_chat_history_model: LLMModels = Field(
        default=LLMModels.GEMINI_FLASH_2_0_LITE,
        description="The model to use for LLM chat history",
    )
    llm_chat_history_temperature: float = Field(
        default=0.1, description="Temperature for LLM chat history"
    )
    llm_rewrite_model: LLMModels = Field(
        default=LLMModels.GEMINI_2_0_FLASH, description="The model to use for LLM rewrites"
    )
    llm_rewrite_temperature: float = Field(
        default=1.0, description="Temperature for LLM rewrites"
    )
    enable_augmentation: bool = Field(
        default=True, description="Whether to enable augmentation or not"
    )
    enable_questions_retrieval: bool = Field(
        default=True, description="Whether to enable questions retrieval or not"
    )
    questions_retriever_params: RetrieverParams = Field(default_factory=RetrieverParams)
    retriever_params: RetrieverParams = Field(
        default_factory=RetrieverParams, description="Parameters for the retriever"
    )
    max_retrievals: int = Field(default=2, description="Maximum number of retrievals")

    def log_parameters(self):
        """Log the current parameters."""
        logger.info(f"Agent parameters: {self.model_dump_json(indent=2)}")


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    parameters: Parameters = Field(...)


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelData]


class DateModel(BaseModel):
    year: Optional[int]
    month: Optional[int]
    day: Optional[int]


class AgentState(TypedDict):
    messages: Annotated[list[ToolMessage], add]
    chat_history: list[BaseMessage]
    question: str
    documents: Annotated[list[Document], add_documents]
    context_is_relevant: bool
    answer: str
    parameters: Parameters
    num_retrievals: int
    retrieval_questions: list[str]
    retrieval_dates: list[DateModel]
    retrieval_entities: list[str]
    decomposed_questions: list[str]


class RetrievalAugmentedMode(str, Enum):
    PREV = "prev"
    NEXT = "next"
    BOTH = "both"
    NONE = "none"
