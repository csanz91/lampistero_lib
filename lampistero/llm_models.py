import datetime

from lampistero.models import (
    ModelData,
    LLMModels,
    Parameters,
    RetrieverParams,
)

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from qdrant_client import QdrantClient

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.chat_models import init_chat_model

from pydantic import SecretStr

from docker_secrets import load_all_secrets, get_docker_secrets

load_all_secrets()


lampistero_llm_gemini = ModelData(
    id="lampistero-gemini",
    created=int(datetime.datetime(year=2025, month=4, day=7).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 12}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_gpt_4_1 = ModelData(
    id="lampistero-gpt-4.1",
    created=int(datetime.datetime(year=2025, month=4, day=7).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.GPT_4_1,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 25}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_deepseek = ModelData(
    id="lampistero-deepseek",
    created=int(datetime.datetime(year=2025, month=4, day=8).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.DEEPSEEK,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 12}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_gemini_2_5_pro = ModelData(
    id="lampistero-gemini_2_5_pro",
    created=int(datetime.datetime(year=2025, month=4, day=16).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.GEMINI_2_5_PRO,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 30}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_o4_mini = ModelData(
    id="lampistero-o4-mini",
    created=int(datetime.datetime(year=2025, month=4, day=17).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.O4_MINI,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 30}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_o3 = ModelData(
    id="lampistero-o3",
    created=int(datetime.datetime(year=2025, month=4, day=17).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.O3,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 12}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_agent = ModelData(
    id="lampistero-agent",
    created=int(datetime.datetime(year=2025, month=4, day=17).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.GEMINI_2_0_FLASH,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 12}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
        agent_mode=True,
    ),
)

lampistero_llm_gemini_2_5_flash = ModelData(
    id="lampistero-gemini-2.5-flash",
    created=int(datetime.datetime(year=2025, month=4, day=18).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.GEMINI_2_5_FLASH,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 30}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 2},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_llm_grok_3_mini = ModelData(
    id="lampistero-grok-3-mini",
    created=int(datetime.datetime(year=2025, month=4, day=20).timestamp()),
    owned_by="csm",
    parameters=Parameters(
        llm_answer_model=LLMModels.GROK_3_MINI,
        enable_reranking=False,
        enable_augmentation=True,
        retriever_params=RetrieverParams(
            search_kwargs={"k": 30}, search_type="similarity"
        ),
        enable_questions_retrieval=True,
        questions_retriever_params=RetrieverParams(
            search_kwargs={"score_threshold": 0.8, "k": 3},
            search_type="similarity_score_threshold",
        ),
    ),
)

lampistero_tasks = ModelData(
    id="lampistero-tasks-001",
    created=int(datetime.datetime(year=2025, month=3, day=19).timestamp()),
    owned_by="csm",
    parameters=Parameters(),
)

models_lookup = {
    "lampistero-gemini": lampistero_llm_gemini,
    "lampistero-gemini-2.5-pro": lampistero_llm_gemini_2_5_pro,
    "lampistero-gemini-2.5-flash": lampistero_llm_gemini_2_5_flash,
    "lampistero-gpt-4.1": lampistero_llm_gpt_4_1,
    "lampistero-tasks-001": lampistero_tasks,
    "lampistero_llm_deepseek": lampistero_llm_deepseek,
    "lampistero-o4-mini": lampistero_llm_o4_mini,
    "lampistero-o3": lampistero_llm_o3,
    "lampistero-agent": lampistero_llm_agent,
    "lampistero-grok-3-mini": lampistero_llm_grok_3_mini,
}


def get_llm_model(
    temperature=1.0,
    max_tokens=None,
    model=LLMModels.GEMINI_2_0_FLASH,
) -> BaseChatModel:
    if model in [
        LLMModels.DEEPSEEK_REASONER,
        LLMModels.GEMINI_PRO,
        LLMModels.GROK_3_MINI,
    ]:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(get_docker_secrets("OPENROUTER_API_KEY")),
            model=model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            max_retries=3,
        )
    else:
        llm = init_chat_model(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=1,
        )

    return llm


def get_embedding_model():
    # embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # embeddings_model = LateInteractionTextEmbedding("antoinelouis/colbert-xm")
    return embeddings_model


embeddings_model = get_embedding_model()

client = QdrantClient(
    url=get_docker_secrets("QDRANT_URL"),
    api_key=get_docker_secrets("QDRANT_API_KEY"),
    prefer_grpc=True,
)

dense_vector_name: str = "dense-embed"
collection_name = "lampistero_20250423_openai"
collection_name_2 = "lampistero_20250423_gemini"
collection_name_questions = "lampistero_20250424_gemini_questions"
# collection_name = "lampistero_20250403_colbert"
sparse_model_name = "Qdrant/bm42-all-minilm-l6-v2-attentions"
vector_size: int = 768
# vector_size: int = 3072


client.set_sparse_model(sparse_model_name)
sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
sparse_vector_name = client.get_sparse_vector_field_name()
assert sparse_vector_name

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings_model,
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name=dense_vector_name,
    sparse_embedding=sparse_embeddings,
    sparse_vector_name=sparse_vector_name,
)

vectorstore_2 = QdrantVectorStore(
    client=client,
    collection_name=collection_name_2,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name=dense_vector_name,
    sparse_embedding=sparse_embeddings,
    sparse_vector_name=sparse_vector_name,
)

vectorstore_questions = QdrantVectorStore(
    client=client,
    collection_name=collection_name_questions,
    embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    retrieval_mode=RetrievalMode.HYBRID,
    vector_name=dense_vector_name,
    sparse_embedding=sparse_embeddings,
    sparse_vector_name=sparse_vector_name,
)
