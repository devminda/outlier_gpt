"""Tests for outlier detection techniques in the outlier_gpt library.
"""
import pandas as pd
import numpy as np
import pytest
from outlier_gpt.techniques import outlier_detection

# Sample data
data = {
    'value': [10, 12, 12, 13, 12, 14, 13, 100, 12, 11, 13, 12, 14, 13, 15]
}
df = pd.DataFrame(data)

# Sample data for rolling window method
rolling_window_df = pd.DataFrame({
    'timestamp': pd.date_range(start='2023-01-01', periods=30, freq='D'),
    'value': np.random.normal(50, 5, 30)
})
rolling_window_df.set_index('timestamp', inplace=True)
rolling_window_df.iloc[10] = 100  # High outlier
rolling_window_df.iloc[20] = 5    # Low outlier


def test_z_score_method():
    z_outliers = outlier_detection.z_score_method(df, 'value')
    assert z_outliers == [7], f"Expected [7], got {z_outliers}"

def test_iqr_method():
    iqr_outliers = outlier_detection.iqr_method(df, 'value')
    assert iqr_outliers == [7], f"Expected [7], got {iqr_outliers}"

def test_modified_z_score_method():
    mod_z_outliers = outlier_detection.modified_z_score_method(df, 'value')
    assert mod_z_outliers == [7], f"Expected [7], got {mod_z_outliers}"

def test_percentile_method():
    perc_outliers = outlier_detection.percentile_method(df, 'value')
    assert perc_outliers == [0, 7], f"Expected [7], got {perc_outliers}"

def test_threshold_method():
    thresh_outliers = outlier_detection.threshold_method(df, 'value', lower_threshold=10, upper_threshold=20)
    assert thresh_outliers == [7], f"Expected [7], got {thresh_outliers}"

def test_rolling_window_method():
    expected_outliers = [
        pd.Timestamp("2023-01-11"),  # index 10
        pd.Timestamp("2023-01-21")   # index 20
    ]

    rw_outliers = outlier_detection.rolling_window_method(
        rolling_window_df, 
        'value', 
        window_size=5, 
        threshold=3
    )

    for ts in expected_outliers:
        assert ts in rw_outliers, f"Expected {ts} in rolling-window outliers, got {rw_outliers}"


def test_grubbstest_method():
    # Skip this test gracefully if scipy isn't available in the environment
    pytest.importorskip('scipy')

    g_outliers = outlier_detection.grubbstest_method(df, 'value', alpha=0.05)
    # Grubbs' test should identify the extreme value at index 7
    assert 7 in g_outliers, f"Expected index 7 to be in Grubbs' outliers, got {g_outliers}"