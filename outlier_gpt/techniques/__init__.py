"""This is the techniques package for outlier detection methods."""

from .outlier_detection import (
    z_score_method,
    iqr_method,
    modified_z_score_method,
    percentile_method,
    threshold_method,
    rolling_window_method,
    grubbstest_method,
)
