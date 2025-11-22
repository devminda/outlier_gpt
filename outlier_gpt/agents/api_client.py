# anomaly_explainer/api_client.py
"""API client module for communicating with OpenAI's language models.

This module handles all interactions with the OpenAI API, including
authentication, request management, and response parsing for LLM-based
outlier explanation.
"""
import time

from openai import OpenAI


def get_llm_response(
    prompt: str, api_key: str, model: str = "gpt-5", attempts: int = 3
):
    """Send a prompt to OpenAI's Chat Completions API and retrieve a response.

    This function communicates with OpenAI's API to generate explanations for
    data anomalies. It includes automatic retry logic with exponential backoff
    for reliability and extracts usage metrics (token counts) from the response.

    Args:
        prompt (str): The input prompt to send to the LLM. Should contain
                     context about the outlier and the data being analyzed.
        api_key (str): OpenAI API key for authentication. Must be a valid key
                      from your OpenAI account.
        model (str, optional): The model to use for completion.
                              Defaults to "gpt-5".
                              Examples: "gpt-4", "gpt-3.5-turbo", etc.
        attempts (int, optional): Number of retry attempts on API failure.
                                 Defaults to 3. Uses exponential backoff
                                 between retries.

    Returns:
        tuple: A two-element tuple containing:
            - str: The generated response text from the LLM
            - dict: Usage metadata including:
                * 'model': The model used
                * 'prompt_tokens': Number of tokens in the prompt
                * 'completion_tokens': Number of tokens in the completion
                * 'total_tokens': Total tokens used in the request

    Raises:
        ValueError: If api_key is None or empty.
        RuntimeError: If the API call fails after all retry attempts are
                     exhausted.

    Example:
        >>> response_text, usage = get_llm_response(
        ...     prompt="Explain why this value is an outlier...",
        ...     api_key="sk-...",
        ...     model="gpt-4"
        ... )
        >>> print(f"Tokens used: {usage['total_tokens']}")
    """
    if not api_key:
        raise ValueError("API key must be provided.")

    client = OpenAI(api_key=api_key)

    for attempt in range(attempts):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt},
                ],
                tools=[{"type": "web_search"}],
            )

            # Extract main text
            text = response.output_text

            # Extract usage safely
            usage_raw = getattr(response, "usage", None)
            if usage_raw:
                prompt_tokens = getattr(usage_raw, "prompt_tokens", 0)
                completion_tokens = getattr(usage_raw, "completion_tokens", 0)
                total_tokens = getattr(
                    usage_raw, "total_tokens", prompt_tokens + completion_tokens
                )
            else:
                prompt_tokens = completion_tokens = total_tokens = 0

            usage = {
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

            return text, usage

        except Exception as e:
            if attempt < attempts - 1:
                time.sleep(2**attempt)  # exponential backoff
            else:
                raise RuntimeError(
                    f"OpenAI API call failed after {attempts} attempts: {e}"
                )

    raise RuntimeError("Unreachable code reached in get_llm_response")
