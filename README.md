# outlier_gpt

![PyPI version](https://img.shields.io/pypi/v/outlier-gpt.svg)
[![Documentation Status](https://readthedocs.org/projects/outlier-gpt/badge/?version=latest)](https://outlier-gpt.readthedocs.io/en/latest/?version=latest)

**outlier_gpt** is a Python package that automates anomaly explanation in tabular data. Instead of just identifying outliers, it explains why they might have occurred by leveraging LLMs (like GPT) combined with statistical outlier detection techniques.

You feed it a pandas DataFrame, specify an outlier detection method, and the tool automatically gathers context from surrounding data points and related columns. This rich context is then passed to a language model to generate plausible hypotheses explaining the anomaly—such as "Data Quality Issue," "External Event," or "Structural Pattern."

* **PyPI package:** https://pypi.org/project/outlier-gpt/
* **Free software:** MIT License
* **Documentation:** https://outlier-gpt.readthedocs.io
* **GitHub:** https://github.com/devminda/outlier_gpt

## Features

* **Multiple Outlier Detection Methods:**
  - Z-score method
  - Interquartile Range (IQR) method
  - Modified Z-score method
  - Percentile method
  - Threshold method
  - Rolling window method
  - Grubbs' test method

* **LLM-Powered Explanations:**
  - Automatic context gathering from nearby data points
  - Support for external search context for deep investigation
  - Detailed classification of anomalies (External Event, Data Quality Issue, Structural Pattern, Unknown)
  - Token usage tracking and API cost monitoring

* **Easy Integration:**
  - Simple `OutlierAgent` API for detection and explanation
  - Flexible context configuration
  - Support for both single and batch outlier explanations

## Installation

### Stable Release

```sh
pip install outlier-gpt
```

Or using `uv`:

```sh
uv add outlier-gpt
```

### From Source

```sh
git clone https://github.com/devminda/outlier_gpt.git
cd outlier_gpt
pip install -e .
```

## Quick Start

```python
import pandas as pd
# Install yfinance: pip install yfinance
import yfinance as yf
from outlier_gpt.agents import OutlierAgent

# 1. Fetch Time Series Data (AAPL stock)
TICKER = 'AAPL'
data_column = 'Close'

# Fetch a few months of data, the index will be a DatetimeIndex
df = yf.download(TICKER, start='2024-01-01', end='2024-05-01')
df.columns = df.columns.droplevel(1)
df.reset_index(inplace=True)
df.rename(columns={"Date": "timestamp", "Close": "value"}, inplace=True)
df.set_index('timestamp', inplace=True)
# 2. Initialize the agent 
# Note: Use a model that supports web browsing/tools like gpt-4-turbo or gpt-4o for deep search
agent = OutlierAgent(api_key="your-openai-api-key", model="gpt-5")

# Optional: Set a global context (accessible via the property setter)
agent.context = f"The primary focus of this analysis is extreme volatility in {TICKER} stock."

# 3. Detect Outliers
# We look for outliers in the 'Close' price using the Z-score method.
outlier_indices = outlier_detection.rolling_window_method(
    stock_data, 'value', window_size=5, threshold=3)

# Visualizing
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['value'], label='Value')
plt.scatter(outliers_stock, df.loc[outliers_stock]['value'], color='red', label='Outliers')
plt.title("Rolling Window Outlier Detection")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Detected {len(outlier_indices)} outlier(s) in {TICKER} stock.")
print(f"Indices (Dates): {list(outlier_indices)}")

# 4. Explain Outliers

# --- Option A (Recommended): Explain a Single Outlier with Deep Search ---
# Analyze the first detected outlier in depth.
# The `deep_search=True` flag tells the LLM to use external tools 
# to find news matching the date/ticker, leading to a much stronger "Market Event" justification.

if outlier_indices:
    first_outlier_index = outlier_indices[0]
    
    explanation_single = agent.explain_outlier(
        df,
        data_column=data_column,
        outlier_index=first_outlier_index,
        context_columns=['Volume', 'High', 'Low'], # Additional columns for LLM context
        deep_search=True # Uses web search/tools for external context
    )
    
    print("\n--- Single Outlier Explanation (Deep Search) ---")
    print(f"Index {first_outlier_index}:\n{explanation_single}")


# --- Option B: Explain Multiple Outliers (Batch Processing) ---
# Use explain_outliers for quick, generalized explanations of all detected anomalies.
# This typically does NOT use deep search for cost/speed reasons.

explanations_batch = agent.explain_outliers(
    df,
    data_column=data_column,
    outlier_indices=outlier_indices,
    context_columns=['Volume']
)

print("\n--- Batch Outlier Explanations (Standard) ---")
for idx, explanation in explanations_batch.items():
    print(f"Index {idx} ({TICKER} Price):\n{explanation}\n")
```

## Requirements

* Python 3.10+
* pandas
* requests
* openai
* scipy (optional, for Grubbs' test method)

## Testing

To run the test suite:

```sh
pip install pytest scipy  # Install test dependencies
pytest
```

## License

MIT License. See `LICENSE` file for details.

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyfeldroy/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.

For questions or support, please open an issue on [GitHub](https://github.com/devminda/outlier_gpt).
