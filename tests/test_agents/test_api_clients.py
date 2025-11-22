"""Tests for outlier_gpt.agents.api_client module.
"""
import pytest
from unittest.mock import patch, MagicMock
from outlier_gpt.agents.api_client import get_llm_response


def test_get_llm_response_success():
    """Test successful LLM call using mocked OpenAI responses.create()."""

    mock_response = MagicMock()
    mock_response.output_text = "Mocked LLM response"
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )

    with patch("outlier_gpt.agents.api_client.OpenAI") as MockClient:
        # Mock client instance
        instance = MockClient.return_value
        instance.responses.create.return_value = mock_response

        text, usage = get_llm_response(
            prompt="Why is AAPL price an outlier?",
            api_key="fakekey123",
            model="gpt-4.1"
        )

    assert text == "Mocked LLM response"
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 20
    assert usage["total_tokens"] == 30


def test_get_llm_response_retry_and_fail():
    """Test that the function retries and eventually fails after attempts."""

    with patch("outlier_gpt.agents.api_client.OpenAI") as MockClient:
        instance = MockClient.return_value
        instance.responses.create.side_effect = Exception("API failure")

        with pytest.raises(RuntimeError) as excinfo:
            get_llm_response(
                prompt="Test failure",
                api_key="fakekey123",
                model="gpt-4.1",
                attempts=2  # Only 2 attempts for test speed
            )

        assert "failed after 2 attempts" in str(excinfo.value)
