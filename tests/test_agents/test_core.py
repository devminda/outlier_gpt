# tests/test_core.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from outlier_gpt.agents import OutlierAgent


def test_init_requires_api_key():
    with pytest.raises(ValueError):
        OutlierAgent(api_key="")


def test_context_property():
    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")
    assert agent.context is None

    agent.context = "Test context string"
    assert agent.context == "Test context string"


@patch("outlier_gpt.techniques.outlier_detection.z_score_method")
def test_detect_outliers_calls_correct_method(mock_z_score):
    # Arrange
    df = pd.DataFrame({"x": [1, 2, 3]})
    mock_z_score.return_value = [1]  # pretend index 1 is an outlier

    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")

    # Act
    outliers = agent.detect_outliers(df, data_column="x", method="z_score", threshold=2.5)

    # Assert
    mock_z_score.assert_called_once_with(df, "x", threshold=2.5)
    assert outliers == [1]


def test_detect_outliers_invalid_method():
    df = pd.DataFrame({"x": [1, 2, 3]})
    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")

    with pytest.raises(ValueError) as excinfo:
        agent.detect_outliers(df, data_column="x", method="unknown_method")

    msg = str(excinfo.value)
    assert "Method 'unknown_method' not recognized" in msg
    assert "z_score" in msg  # list of available methods should be in message


@patch("outlier_gpt.agents.api_client.get_llm_response")
@patch("outlier_gpt.agents.prompter.build_explanation_prompt")
def test_explain_outlier_shallow(mock_build_prompt, mock_llm):
    df = pd.DataFrame({"x": [1, 100, 2]}, index=[0, 1, 2])

    mock_build_prompt.return_value = "PROMPT_SHALLOW"
    mock_llm.return_value = ("Explanation text", {"total_tokens": 42})

    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")
    agent.context = "Some context"

    text = agent.explain_outlier(
        df=df,
        data_column="x",
        outlier_index=1,
        context_columns=["x"],
        deep_search=False,
        return_usage=True,
    )

    # prompt builder should be called
    mock_build_prompt.assert_called_once()
    # LLM client should be called with our prompt
    mock_llm.assert_called_once_with(
        prompt="PROMPT_SHALLOW",
        api_key="fakekey",
        model="gpt-4.1",
    )

    assert text == "Explanation text"
    assert agent.last_usage == {"total_tokens": 42}


@patch("outlier_gpt.agents.api_client.get_llm_response")
@patch("outlier_gpt.agents.prompter.build_deep_explanation_prompt")
def test_explain_outlier_deep_search(mock_build_deep_prompt, mock_llm):
    df = pd.DataFrame({"x": [1, 100, 2]}, index=[0, 1, 2])

    mock_build_deep_prompt.return_value = "PROMPT_DEEP"
    mock_llm.return_value = ("Deep explanation", {"total_tokens": 99})

    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")

    text = agent.explain_outlier(
        df=df,
        data_column="x",
        outlier_index=1,
        context_columns=["x"],
        deep_search=True,
        return_usage=True,
    )

    mock_build_deep_prompt.assert_called_once()
    mock_llm.assert_called_once_with(
        prompt="PROMPT_DEEP",
        api_key="fakekey",
        model="gpt-4.1",
    )

    assert text == "Deep explanation"
    assert agent.last_usage == {"total_tokens": 99}


@patch("outlier_gpt.agents.api_client.get_llm_response")
@patch("outlier_gpt.agents.prompter.build_explanation_prompt")
def test_explain_outlier_handles_api_error(mock_build_prompt, mock_llm):
    df = pd.DataFrame({"x": [1, 100, 2]}, index=[0, 1, 2])

    mock_build_prompt.return_value = "PROMPT"
    mock_llm.side_effect = Exception("API failure")

    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")

    text = agent.explain_outlier(
        df=df,
        data_column="x",
        outlier_index=1,
        context_columns=["x"],
        deep_search=False,
        return_usage=True,
    )

    assert "Error contacting API" in text
    assert "API failure" in text
    assert agent.last_usage is None


@patch("outlier_gpt.agents.api_client.get_llm_response")
@patch("outlier_gpt.agents.prompter.build_explanation_prompt")
def test_explain_outliers_batch(mock_build_prompt, mock_llm):
    df = pd.DataFrame({"x": [1, 100, 2]}, index=[0, 1, 2])
    outlier_indices = [1, 2]

    # Mock prompt builder: return something dependent on index
    def build_prompt_side_effect(df_arg, col, idx_arg, ctx_cols):
        return f"PROMPT_FOR_{idx_arg}"

    mock_build_prompt.side_effect = build_prompt_side_effect

    # Mock LLM: for each prompt, return different responses & usage
    def llm_side_effect(prompt, api_key, model):
        return f"Response for {prompt}", {"total_tokens": len(prompt)}

    mock_llm.side_effect = llm_side_effect

    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")

    explanations = agent.explain_outliers(
        df=df,
        data_column="x",
        outlier_indices=outlier_indices,
        context_columns=["x"],
        return_usage=True,
    )

    # Check explanations mapping
    assert explanations[1] == "Response for PROMPT_FOR_1"
    assert explanations[2] == "Response for PROMPT_FOR_2"

    # last_usage should be a dict: index -> usage dict
    usage = agent.last_usage
    assert isinstance(usage, dict)
    assert usage[1]["total_tokens"] == len("PROMPT_FOR_1")
    assert usage[2]["total_tokens"] == len("PROMPT_FOR_2")


@patch("outlier_gpt.agents.api_client.get_llm_response")
@patch("outlier_gpt.agents.prompter.build_explanation_prompt")
def test_explain_outliers_handles_errors_per_index(mock_build_prompt, mock_llm):
    df = pd.DataFrame({"x": [1, 100, 2]}, index=[0, 1, 2])
    outlier_indices = [1, 2]

    mock_build_prompt.side_effect = lambda df_arg, col, idx_arg, ctx_cols: f"PROMPT_{idx_arg}"

    # Make first call succeed, second fail
    def llm_side_effect(prompt, api_key, model):
        if "PROMPT_1" in prompt:
            return "OK for 1", {"total_tokens": 10}
        else:
            raise Exception("API failure for index 2")

    mock_llm.side_effect = llm_side_effect

    agent = OutlierAgent(api_key="fakekey", model="gpt-4.1")

    explanations = agent.explain_outliers(
        df=df,
        data_column="x",
        outlier_indices=outlier_indices,
        context_columns=["x"],
        return_usage=True,
    )

    assert explanations[1] == "OK for 1"
    assert "Error contacting API for index 2" in explanations[2]

    # Usage map should store None for failed index
    usage = agent.last_usage
    assert usage[1]["total_tokens"] == 10
    assert usage[2] is None
