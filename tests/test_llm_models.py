import pytest
from lampistero.llm_models import get_llm_model, get_embedding_model
from lampistero.models import LLMModels

def test_get_llm_model():
    model = get_llm_model(model=LLMModels.GEMINI_2_0_FLASH)
    assert model is not None
    assert model.temperature == 1.0 # type: ignore

def test_get_embedding_model():
    embedding_model = get_embedding_model()
    assert embedding_model is not None