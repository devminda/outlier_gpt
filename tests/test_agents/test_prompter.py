import pandas as pd
import pytest

from outlier_gpt.agents.prompter import (
    build_explanation_prompt,
    build_deep_explanation_prompt
)


# ---------------------------------------------------------------------------
# Helper: simple DataFrame
# ---------------------------------------------------------------------------
@pytest.fixture
def df_sample():
    # Index: 0, 1, 2
    return pd.DataFrame({
        "price": [100, 999, 105],
        "volume": [10, 20, 15],
        "region": ["A", "A", "A"],
    })


# ---------------------------------------------------------------------------
# TESTS FOR build_explanation_prompt
# ---------------------------------------------------------------------------
def test_build_explanation_prompt_basic(df_sample):
    prompt = build_explanation_prompt(
        df=df_sample,
        data_column="price",
        outlier_index=1,
        context_columns=["volume", "region"],
        context="Test context"
    )

    # Core expectations
    assert "Outlier detected:" in prompt
    assert "Column: price" in prompt
    assert "Index: 1" in prompt
    assert "999" in prompt  # actual outlier value
    assert "Nearby values:" in prompt
    assert "Hypothesis:" in prompt
    assert "Classification:" in prompt
    assert "Justification:" in prompt
    assert "price 1 anomaly cause" in prompt  # search query included


def test_build_explanation_prompt_missing_column(df_sample):
    with pytest.raises(ValueError):
        build_explanation_prompt(
            df=df_sample,
            data_column="nonexistent",
            outlier_index=1,
            context_columns=["volume"],
            context=""
        )


def test_build_explanation_prompt_missing_index(df_sample):
    with pytest.raises(KeyError):
        build_explanation_prompt(
            df=df_sample,
            data_column="price",
            outlier_index=999,   # wrong index
            context_columns=["volume"],
            context=""
        )


# ---------------------------------------------------------------------------
# TESTS FOR build_deep_explanation_prompt
# ---------------------------------------------------------------------------
def test_build_deep_explanation_prompt_basic(df_sample):
    prompt = build_deep_explanation_prompt(
        df=df_sample,
        data_column="price",
        outlier_index=1,
        context_columns=["volume"],
        context="Deep context"
    )

    # Check key elements
    assert "High-level context" in prompt
    assert "Nearby rows" in prompt
    assert "Deep Search Instructions" in prompt
    assert "perform a web search" in prompt
    assert "No relevant external information found" in prompt  # required fallback rule
    assert "price 1 anomaly cause" in prompt
    assert "Hypothesis:" in prompt
    assert "Classification:" in prompt
    assert "Justification:" in prompt


def test_build_deep_explanation_prompt_invalid_column(df_sample):
    with pytest.raises(ValueError):
        build_deep_explanation_prompt(
            df=df_sample,
            data_column="badcol",
            outlier_index=1,
            context_columns=["volume"],
            context=""
        )


def test_build_deep_explanation_prompt_invalid_index(df_sample):
    with pytest.raises(KeyError):
        build_deep_explanation_prompt(
            df=df_sample,
            data_column="price",
            outlier_index=999,
            context_columns=["volume"],
            context=""
        )
