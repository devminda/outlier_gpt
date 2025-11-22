# anomaly_explainer/prompter.py
"""Prompt builder module for constructing LLM prompts for outlier explanation.

This module provides functions to build well-structured prompts for the LLM,
including context about outliers, nearby data points, and instructions for
generating explanations and hypotheses.
"""
import pandas as pd
from typing import List, Hashable


def build_explanation_prompt(
    df: pd.DataFrame,
    data_column: str,
    outlier_index: Hashable,
    context_columns: List[str],
    context: str,
) -> str:
    """
    Build a prompt for explaining a single outlier to the LLM.

    Constructs a detailed prompt that includes the outlier value, nearby data
    points for context, and instructions for the LLM to generate an explanation.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        data_column (str): Name of the column containing the outlier.
        outlier_index (Hashable): Index/label of the outlier row in the DataFrame.
        context_columns (List[str]): List of column names to include as context.
                                    If None, all columns except data_column are used.
        context (str): Optional domain context or background information to include.

    Returns:
        str: A formatted prompt string for the LLM.

    Raises:
        ValueError: If data_column is not found in the DataFrame.

    Example:
        >>> df = pd.DataFrame({'value': [1, 2, 100, 3], 'category': ['A', 'A', 'B', 'A']})
        >>> prompt = build_explanation_prompt(df, 'value', 2, ['category'], '')
        >>> # prompt now contains the outlier info and instructions for the LLM
    """
    if data_column not in df.columns:
        raise ValueError(f"data_column '{data_column}' not found in DataFrame columns.")
    # Clean context columns
    if context_columns is None:
        context_columns = [c for c in df.columns if c != data_column]
    else:
        context_columns = [
            c for c in context_columns if c in df.columns and c != data_column
        ]

    # Locate row + small context window
    outlier_row = df.loc[outlier_index]
    pos = df.index.get_loc(outlier_index)
    if isinstance(pos, slice) or isinstance(pos, list):
        pos = list(range(len(df)))[pos][0]
    window = df.iloc[max(0, pos - 1):min(len(df), pos + 2)]
    cols = [data_column] + [c for c in context_columns]

    search_query = f"{data_column} {outlier_index} anomaly cause"

    prompt = f"""
You are an analyst explaining an anomaly in a dataset.

Context: {context}

Outlier detected:
- Column: {data_column}
- Index: {outlier_index}
- Value: {outlier_row[data_column]!r}

Nearby values:
{window[cols].to_string()}

Deep Search Task:
1. If available, run a web search using: "{search_query}".
2. Use external info *or* dataset patterns to explain the anomaly.
3. If no relevant external info exists, explicitly say:
   "No external information found. Explanation based only on data."

Output Format:
Hypothesis: [...]
Classification: External Event / Data Quality Issue / Structural Pattern / Unknown
Justification: 2–4 sentences using either search results or feature patterns.
"""
    return prompt


def build_deep_explanation_prompt(
    df: pd.DataFrame,
    data_column: str,
    outlier_index: Hashable,
    context_columns: List[str],
    context: str,
) -> str:
    """
    Build a comprehensive prompt for deep investigation of an outlier.

    This function constructs a more detailed prompt than build_explanation_prompt,
    including the full outlier row and a larger context window. It's designed for
    in-depth analysis when the LLM has access to external search tools.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        data_column (str): Name of the column containing the outlier.
        outlier_index (Hashable): Index/label of the outlier row in the DataFrame.
        context_columns (List[str]): List of column names to include as context.
                                    If None, all columns except data_column are used.
        context (str): Domain context or background information for the LLM.

    Returns:
        str: A detailed formatted prompt string for the LLM with deep search instructions.

    Raises:
        ValueError: If data_column is not found in the DataFrame.
        KeyError: If outlier_index is not found in the DataFrame.
        RuntimeError: If the position of the outlier cannot be determined.

    Example:
        >>> df = pd.DataFrame({'value': [1, 2, 100, 3], 'category': ['A', 'A', 'B', 'A']})
        >>> prompt = build_deep_explanation_prompt(df, 'value', 2, ['category'], 'Sales data')
        >>> # prompt includes deep search instructions and full row context
    """
    if data_column not in df.columns:
        raise ValueError(f"data_column '{data_column}' not found in DataFrame columns.")

    # Validate / clean context columns
    if context_columns is None:
        context_columns = [c for c in df.columns if c != data_column]
    else:
        context_columns = [
            c for c in context_columns if c in df.columns and c != data_column
        ]

    try:
        outlier_row = df.loc[outlier_index]
        pos = df.index.get_loc(outlier_index)
        if isinstance(pos, slice) or isinstance(pos, list):
            if isinstance(pos, slice):
                pos = range(len(df))[pos][0]
            else:
                pos = pos[0]
    except KeyError:
        raise KeyError(f"Outlier index '{outlier_index}' not found in DataFrame index.")
    except Exception as e:
        raise RuntimeError(f"Could not locate position of index '{outlier_index}': {e}")

    # Context window around outlier
    window_start = max(0, pos - 2)
    window_end = min(len(df), pos + 3)
    context_window = df.iloc[window_start:window_end]

    cols_for_context = [data_column] + [c for c in context_columns if c != data_column]
    context_window_print = context_window[cols_for_context]
    outlier_row_print = outlier_row

    # Generic search query – works for many domains
    search_query = f"{data_column} {outlier_index} anomaly cause"

    prompt = f"""
    You are a senior data analyst investigating an anomaly in a tabular dataset.

    High-level context:
    {context}

    An outlier was detected in column '{data_column}' at index '{outlier_index}'.
    Outlier value: {outlier_row[data_column]!r}

    Nearby rows (selected columns):
    {context_window_print.to_string()}

    Full row:
    {outlier_row_print.to_string()}

    ### Deep Search Instructions

    1. If you have access to external tools, perform a web search using the query "{search_query}" (and close variants).
    2. Look for specific external factors that could explain the anomaly (e.g., news, policy changes, local conditions, structural/domain factors).
    3. If you find relevant information, ground your hypothesis in it and mention at least one concrete detail (such as an event name, article, date, location, or specific domain factor).
    4. If you do NOT find any relevant external information, you MUST explicitly state:
    "No relevant external information found. Hypothesis based only on dataset features."

    ### Required Output Format

    Hypothesis: [one concise, most plausible explanation]

    Classification: [choose one: External Event / Data Quality Issue / Structural Pattern in Features / Unknown]

    Justification: [2–5 sentences. If external info was found, reference at least one specific detail and link it to the pattern in the data. Otherwise, clearly state that the explanation is based only on the dataset.]
    """
    return prompt
