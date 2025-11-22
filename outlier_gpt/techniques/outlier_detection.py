"""Build different outlier detection techniques.

This module provides a collection of statistical methods for detecting outliers
in numerical data. Each method has different assumptions and sensitivities,
suitable for different data distributions and use cases.
"""
import numpy as np
from scipy import stats


def z_score_method(df, data_column, threshold=3):
    """
    Identify outliers using the Z-score method.

    Flags values that deviate significantly from the mean based on the number
    of standard deviations (Z-score). Assumes data is approximately normally
    distributed.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        threshold (float, optional): Number of standard deviations from the mean
                                   to consider as an outlier. Defaults to 3.
                                   Common values: 2.5 (98.8%), 3 (99.7%).

    Returns:
        list: Indices of detected outliers.

    Example:
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
        >>> outliers = z_score_method(df, 'values', threshold=2)
        >>> print(outliers)  # [5]

    Note:
        - Sensitive to extreme values (not robust to very large outliers).
        - Works best with normally distributed data.
    """
    mean = df[data_column].mean()
    std = df[data_column].std()

    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    outlier_indices = df.index[
        (df[data_column] < lower_bound) | (df[data_column] > upper_bound)
    ].tolist()

    return outlier_indices


def iqr_method(df, data_column, factor=1.5):
    """
    Identify outliers using the Interquartile Range (IQR) method.

    Detects outliers as values beyond the lower and upper fences defined by
    the first and third quartiles. This method is robust to extreme values
    and works well with skewed distributions.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        factor (float, optional): Multiplier for IQR to define fence distance.
                                 Defaults to 1.5 (standard Tukey fences).
                                 Common values: 1.5 (standard), 3.0 (very conservative).

    Returns:
        list: Indices of detected outliers.

    Example:
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
        >>> outliers = iqr_method(df, 'values', factor=1.5)
        >>> print(outliers)  # Indices beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR

    Note:
        - Robust to extreme values (doesn't use mean/std).
        - Works well with any distribution.
        - More conservative than Z-score with factor=1.5.
    """
    Q1 = df[data_column].quantile(0.25)
    Q3 = df[data_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outlier_indices = df.index[
        (df[data_column] < lower_bound) | (df[data_column] > upper_bound)
    ].tolist()

    return outlier_indices


def modified_z_score_method(df, data_column, threshold=3.5):
    """
    Identify outliers using the Modified Z-score method.

    A robust variant of the Z-score method that uses the median and Median
    Absolute Deviation (MAD) instead of mean and standard deviation. More
    resistant to extreme outliers than standard Z-score.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        threshold (float, optional): Threshold for modified Z-score magnitude.
                                   Defaults to 3.5. A value of 3.5 is roughly
                                   equivalent to a Z-score of 3.

    Returns:
        list: Indices of detected outliers.

    Example:
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
        >>> outliers = modified_z_score_method(df, 'values', threshold=3.5)
        >>> print(outliers)  # [5]

    Note:
        - More robust than standard Z-score for data with extreme values.
        - Uses constant 0.6745 (related to normal distribution).
        - Recommended for real-world data with potential outliers.
    """
    median = df[data_column].median()
    mad = (df[data_column] - median).abs().median()

    modified_z_scores = 0.6745 * (df[data_column] - median) / mad

    outlier_indices = df.index[modified_z_scores.abs() > threshold].tolist()

    return outlier_indices


def percentile_method(df, data_column, lower_percentile=0.01, upper_percentile=0.99):
    """
    Identify outliers using the Percentile method.

    Treats values below a lower percentile and above an upper percentile as
    outliers. A simple, intuitive method based on data quantiles.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        lower_percentile (float, optional): Lower percentile threshold (0-1).
                                           Defaults to 0.01 (1st percentile).
        upper_percentile (float, optional): Upper percentile threshold (0-1).
                                           Defaults to 0.99 (99th percentile).

    Returns:
        list: Indices of detected outliers.

    Example:
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
        >>> outliers = percentile_method(df, 'values', lower_percentile=0.05, upper_percentile=0.95)
        >>> print(outliers)  # Values below 5th percentile or above 95th percentile

    Note:
        - Simple and interpretable.
        - Always flags exactly (lower + (1-upper))% of data as outliers (approximately).
        - Does not depend on distribution assumptions.
    """
    lower_bound = df[data_column].quantile(lower_percentile)
    upper_bound = df[data_column].quantile(upper_percentile)

    outlier_indices = df.index[
        (df[data_column] < lower_bound) | (df[data_column] > upper_bound)
    ].tolist()

    return outlier_indices


def threshold_method(df, data_column, lower_threshold, upper_threshold):
    """
    Identify outliers using a simple threshold method.

    Flags all values outside a specified range as outliers. Useful when you have
    domain knowledge about acceptable ranges.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        lower_threshold (float): Minimum acceptable value.
        upper_threshold (float): Maximum acceptable value.

    Returns:
        list: Indices of detected outliers.

    Example:
        >>> df = pd.DataFrame({'temperature': [20, 21, 22, 100, 25]})
        >>> outliers = threshold_method(df, 'temperature', lower_threshold=15, upper_threshold=30)
        >>> print(outliers)  # [3] (value 100 is outside [15, 30])

    Note:
        - Requires domain knowledge to set appropriate thresholds.
        - Not data-driven (no statistical basis).
        - Simple and fast.
    """
    outlier_indices = df.index[
        (df[data_column] < lower_threshold) | (df[data_column] > upper_threshold)
    ].tolist()

    return outlier_indices


def rolling_window_method(df, data_column, window_size=5, threshold=3):
    """
    Identify outliers using a rolling window method based on Z-score.

    Applies Z-score analysis within a sliding window, useful for detecting
    anomalies in time-series data where values deviate significantly from
    their local neighborhood.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        window_size (int, optional): Size of the rolling window.
                                    Defaults to 5. Larger windows smooth more.
        threshold (float, optional): Number of standard deviations within the
                                   window to flag as outlier. Defaults to 3.

    Returns:
        list: Indices of detected outliers.

    Example:
        >>> df = pd.DataFrame({'values': [10, 11, 12, 100, 13, 14]})
        >>> outliers = rolling_window_method(df, 'values', window_size=3, threshold=2)
        >>> print(outliers)  # Indices that deviate significantly within their window

    Note:
        - Excellent for time-series data.
        - Uses past data (shifted window) to avoid look-ahead bias.
        - First few rows may have NaN due to window initialization.
    """
    rolling_mean = df[data_column].rolling(window=window_size).mean().shift(1)
    rolling_std = df[data_column].rolling(window=window_size).std().shift(1)

    outlier_mask = (df[data_column] - rolling_mean).abs() > threshold * rolling_std
    outliers = df.index[outlier_mask].tolist()

    return outliers


def grubbstest_method(df, data_column, alpha=0.05):
    """
    Identify outliers using Grubbs' test.

    A statistical hypothesis test that iteratively identifies and removes the
    most extreme outlier until none remain significantly different. More
    conservative than other methods; assumes approximately normal distribution.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        data_column (str): Name of the column to analyze for outliers.
        alpha (float, optional): Significance level for the test. Defaults to 0.05.
                               Smaller alpha = fewer outliers detected.
                               Typical range: 0.01 to 0.1.

    Returns:
        list: Indices of detected outliers (in order they were removed).

    Example:
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5, 100]})
        >>> outliers = grubbstest_method(df, 'values', alpha=0.05)
        >>> print(outliers)  # [5]

    Raises:
        ImportError: If scipy is not installed.

    Note:
        - Assumes data is approximately normally distributed.
        - Tests for a single outlier at a time (iterative).
        - More statistically rigorous than heuristic methods.
        - Stops when no more significant outliers are found.
        - Slower than other methods due to iterative nature.
    """
    outlier_indices = []
    data = df[data_column].copy()

    while len(data) > 2:
        mean = data.mean()
        std = data.std()
        N = len(data)
        G_calculated = np.abs(data - mean) / std
        ppf_val = stats.t.ppf(1 - alpha / (2 * N), N - 2)
        G_critical = ((N - 1) / np.sqrt(N)) * np.sqrt(
            ppf_val**2 / (N - 2 + ppf_val**2)
        )

        max_G = G_calculated.max()
        if max_G > G_critical:
            outlier_index = G_calculated.idxmax()
            outlier_indices.append(outlier_index)
            data = data.drop(outlier_index)
        else:
            break

    return outlier_indices
