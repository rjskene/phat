# Data used by the documentation notebooks

Two files of daily simple returns, `close[t] / close[t-1] - 1`, one row per
trading day, columns `date,return`.

| File | Series | First return | Last return | Rows |
|---|---|---|---|---|
| `sp500_returns.csv` | S&P 500 index (`^GSPC`), closing level | 1950-01-04 | 2026-11-24 | 19,346 |
| `ko_returns.csv` | The Coca-Cola Company (`KO`), split-adjusted close, dividends not included | 1962-01-03 | 2026-11-24 | 16,333 |

Source: Yahoo Finance, downloaded on 2026-11-25 with `yfinance` 1.8.0
(`yf.download(ticker, start=..., end='2026-11-25', auto_adjust=False)`,
column `Close`). Only the returns are stored, rounded to eight decimals.
The last closes were 7201.09 (`^GSPC`) and 97.62 (`KO`); the notebooks that
simulate prices start from those values.

The files are here so that the notebooks run without a network connection
and always on the same sample. They are illustrations for the documentation,
not a maintained data set. The S&P 500 is a product of S&P Dow Jones Indices
LLC; for any other use, obtain the data from the provider.
