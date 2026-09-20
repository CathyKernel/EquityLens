"""Equity Quant ML - source package.

Modules
-------
data_acquisition   Market data download (yfinance) + Parquet cache
sql_layer          SQLite warehouse and analytical SQL queries
data_cleaning      Panel cleaning, validation and quality reporting
eda                Exploratory analysis and headline figures
statistical_tests  Hypothesis-testing suite (JB, ADF, Ljung-Box, ANOVA)
feature_engineering Technical features and next-day target (leakage-safe)
modeling           Classifier comparison with time-series CV
backtest           Vectorised out-of-sample strategy backtest
report_generator   Self-contained HTML report rendering
"""

__version__ = "1.0.0"
