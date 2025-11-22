# anomaly_explainer/core.py
"""Core module for the outlier detection and explanation agent.

This module contains the OutlierAgent class, which orchestrates outlier
detection and LLM-powered explanation generation. It manages API
credentials, context, and provides the main interface for users to detect
and explain anomalies.
"""
import pandas as pd

from . import api_client
from . import prompter
from .. import techniques


class OutlierAgent:
    """Main agent class for detecting and explaining outliers in tabular data.

    The OutlierAgent combines statistical outlier detection methods with
    LLM-powered explanations. It manages API credentials, maintains context
    for explanations, and provides methods to detect outliers and generate
    hypotheses for their causes.

    Attributes:
        api_key (str): OpenAI API key for LLM calls.
        model (str): The LLM model to use for generating explanations.
        context (str): Optional context string to include in explanation
                      prompts.
        last_usage (dict): Usage metadata from the most recent LLM call(s).

    Example:
        >>> agent = OutlierAgent(api_key="sk-...", model="gpt-4")
        >>> df = pd.DataFrame({'sales': [100, 105, 102, 5000]})
        >>> outliers = agent.detect_outliers(df, 'sales', method='z_score')
        >>> explanations = agent.explain_outliers(
        ...     df, 'sales', outliers, ['store_id']
        ... )
    """

    def __init__(self, api_key: str, model: str = "gpt-5"):
        """Initialize the OutlierAgent with API credentials.

        Args:
            api_key (str): OpenAI API key for authentication.
                          Must not be empty.
            model (str, optional): The LLM model to use. Defaults to "gpt-5".
                                  Examples: "gpt-4", "gpt-3.5-turbo", etc.

        Raises:
            ValueError: If api_key is None or empty.
        """
        if not api_key:
            raise ValueError("API key must be provided.")
        self.api_key = api_key
        self.model = model
        self._context = None
        # internal storage for token/cost data
        self._last_usage = None

    @property
    def last_usage(self):
        """Get usage information from the most recent LLM call(s).

        Returns:
            dict or None: Usage metadata including token counts, or None if
                         no usage was collected.
        """
        return self._last_usage

    @property
    def context(self):
        """Get the current context string used in prompts.

        Returns:
            str or None: The context string, or None if not set.
        """
        return self._context

    @context.setter
    def context(self, value: str):
        """Set the context string to include in explanation prompts.

        This context is prepended to all LLM prompts to provide background
        information about the dataset or domain.

        Args:
            value (str): The context string to use in future prompts.
        """
        self._context = value

    def detect_outliers(
        self, df: pd.DataFrame, data_column: str, method: str = "z_score", **kwargs
    ):
        """
        Detect outliers in a DataFrame using a specified statistical method.

        Applies a chosen outlier detection technique to identify anomalous values
        in the specified column. Supports multiple methods such as Z-score, IQR,
        and Grubbs' test.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data to analyze.
            data_column (str): Name of the column to check for outliers.
            method (str, optional): Detection method to use. Defaults to "z_score".
                                   Available methods:
                                   - "z_score": Standard Z-score method
                                   - "iqr": Interquartile Range method
                                   - "modified_z_score": Robust Z-score using MAD
                                   - "percentile": Percentile-based method
                                   - "threshold": Simple threshold method
                                   - "rolling_window": Rolling Z-score method
                                   - "grubbstest": Grubbs' statistical test
            **kwargs: Additional keyword arguments passed to the detection method.
                     For example, for z_score: threshold=3
                     For iqr: factor=1.5
                     For threshold: lower_threshold=10, upper_threshold=20

        Returns:
            list: Indices of detected outliers in the DataFrame.

        Raises:
            ValueError: If the specified method is not recognized.

        Example:
            >>> df = pd.DataFrame({'values': [1, 2, 3, 100, 4, 5]})
            >>> agent = OutlierAgent(api_key="sk-...")
            >>> outlier_indices = agent.detect_outliers(df, 'values', method='z_score', threshold=2.5)
            >>> print(outlier_indices)  # [3]
        """
        detection_methods = {
            "z_score": techniques.outlier_detection.z_score_method,
            "iqr": techniques.outlier_detection.iqr_method,
            "modified_z_score": techniques.outlier_detection.modified_z_score_method,
            "percentile": techniques.outlier_detection.percentile_method,
            "threshold": techniques.outlier_detection.threshold_method,
            "rolling_window": techniques.outlier_detection.rolling_window_method,
            "grubbstest": techniques.outlier_detection.grubbstest_method,
        }

        if method not in detection_methods:
            raise ValueError(
                f"Method '{method}' not recognized. "
                f"Available methods: {list(detection_methods.keys())}"
            )

        outlier_indices = detection_methods[method](df, data_column, **kwargs)
        return outlier_indices

    def explain_outlier(
        self,
        df,
        data_column,
        outlier_index,
        context_columns,
        deep_search: bool = False,
        return_usage: bool = False,
    ):
        """
        Generate an LLM-powered explanation for a single outlier.

        Analyzes the outlier in context of surrounding data and generates a
        hypothesis explaining its occurrence.

        Args:
            df (pd.DataFrame): The DataFrame containing the outlier.
            data_column (str): Name of the column containing the outlier.
            outlier_index: Index (row label) of the outlier in the DataFrame.
            context_columns (list): Column names to include as context in the prompt.
            deep_search (bool, optional): If True, uses deep search mode to gather
                                         additional context. Defaults to False.
            return_usage (bool, optional): If True, stores token usage information
                                          in self.last_usage. Defaults to False.

        Returns:
            str: The LLM-generated explanation for the outlier, or an error message
                if the API call fails.

        Example:
            >>> explanation = agent.explain_outlier(
            ...     df, 'sales', outlier_index=3,
            ...     context_columns=['store_id', 'day_of_week'],
            ...     return_usage=True
            ... )
            >>> print(explanation)
            >>> print(agent.last_usage)  # Token count info
        """
        context = self._context if self._context is not None else ""
        if deep_search:
            # Perform deep search to gather more context
            prompt = prompter.build_deep_explanation_prompt(
                df, data_column, outlier_index, context_columns, context
            )
        else:
            prompt = prompter.build_explanation_prompt(
                df, data_column, outlier_index, context_columns, context
            )

        try:
            text, usage = api_client.get_llm_response(
                prompt=prompt,
                api_key=self.api_key,
                model=self.model,
            )

            # Only store usage if requested
            if return_usage:
                self._last_usage = usage

        except Exception as e:
            text = f"Error contacting API: {e}"
            if return_usage:
                self._last_usage = None

        return text

    def explain_outliers(
        self,
        df,
        data_column,
        outlier_indices,
        context_columns,
        return_usage: bool = False,
    ):
        """
        Generate LLM-powered explanations for multiple outliers.

        Processes a list of outlier indices and generates explanations for each one.
        Useful for batch analysis of multiple anomalies.

        Args:
            df (pd.DataFrame): The DataFrame containing the outliers.
            data_column (str): Name of the column containing the outliers.
            outlier_indices (list): List of indices (row labels) to explain.
            context_columns (list): Column names to include as context in prompts.
            return_usage (bool, optional): If True, stores token usage for each
                                          outlier in self.last_usage as a dict.
                                          Defaults to False.

        Returns:
            dict: Mapping of outlier indices to their LLM-generated explanations.
                 Format: {index: explanation_str, ...}

        Example:
            >>> outlier_indices = [3, 5, 7]
            >>> explanations = agent.explain_outliers(
            ...     df, 'sales', outlier_indices,
            ...     context_columns=['store_id', 'day_of_week'],
            ...     return_usage=True
            ... )
            >>> for idx, exp in explanations.items():
            ...     print(f"Index {idx}: {exp}")
            >>> print(agent.last_usage)  # {3: {...}, 5: {...}, 7: {...}}
        """
        explanations = {}
        usage_map = {}

        for idx in list(outlier_indices):
            prompt = prompter.build_explanation_prompt(
                df, data_column, idx, context_columns
            )

            try:
                text, usage = api_client.get_llm_response(
                    prompt=prompt,
                    api_key=self.api_key,
                    model=self.model,
                )
            except Exception as e:
                text = f"Error contacting API for index {idx}: {e}"
                usage = None

            explanations[idx] = text
            usage_map[idx] = usage

        # Only store usage if user requested it
        if return_usage:
            self._last_usage = usage_map

        return explanations
