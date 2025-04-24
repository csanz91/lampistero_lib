import pytest
from lampistero.models import CompletionRequest, Parameters, LLMModels

def test_completion_request():
    request = CompletionRequest(model="test-model", prompt="Hello, world!", max_tokens=10)
    assert request.model == "test-model"
    assert request.prompt == "Hello, world!"
    assert request.max_tokens == 10

def test_parameters():
    params = Parameters(llm_answer_model=LLMModels.GEMINI_2_0_FLASH, max_retrievals=5)
    assert params.llm_answer_model == LLMModels.GEMINI_2_0_FLASH
    assert params.max_retrievals == 5