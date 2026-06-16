# Dataset: S&P 500 Stock Market Data

## Source
- **File**: `stock_data.csv`
- **Context**: Financial indices and performance metrics for 340 S&P 500 companies

## Overview
| Property | Value |
|---|---|
| Rows | 340 companies |
| Columns | 15 (11 numeric + 4 categorical) |
| Task | Unsupervised clustering — group companies by financial profile |
| Missing values | None |
| Duplicates | None |

## Features
| Column | Type | Description |
|---|---|---|
| Ticker Symbol | Categorical | Stock ticker abbreviation |
| Security | Categorical | Company name |
| GICS Sector | Categorical | 11 economic sectors (GICS standard) |
| GICS Sub Industry | Categorical | More granular sector classification |
| Current Price | Numeric | Current stock price (USD) |
| Price Change | Numeric | % price change over 13 weeks |
| Volatility | Numeric | Std dev of stock price over 13 weeks |
| ROE | Numeric | Return on Equity (net income / shareholders' equity) |
| Cash Ratio | Numeric | Cash & equivalents / current liabilities |
| Net Cash Flow | Numeric | Cash inflows minus outflows (USD) |
| Net Income | Numeric | Revenues − expenses − interest − taxes (USD) |
| Earnings Per Share | Numeric | Net profit / shares outstanding (USD) |
| Estimated Shares Outstanding | Numeric | Total shares held by shareholders |
| P/E Ratio | Numeric | Stock price / earnings per share |
| P/B Ratio | Numeric | Stock price / book value per share |

## EDA Key Findings
- Most numeric features are **right-skewed with many outliers**
- **Sector patterns**: Health Care has highest avg price & cash ratio; Energy has highest volatility & worst price change
- **Correlations**: Net Income ↔ Estimated Shares Outstanding (positive); Price Change ↔ Volatility (inverse)
- No strong natural cluster boundaries visible in pairplots

## Preprocessing
- Only numeric columns used for clustering (categorical columns excluded)
- `StandardScaler` applied before all clustering
- **Outlier removal** (IQR ×1.5) applied to 4 columns: Current Price, Net Cash Flow, ROE, Net Income → reduces data from 340 to 201 rows

## Models Trained

### K-Means Clustering

#### Full Data (340 rows)
| Model | k | Notes |
|---|---|---|
| `kmeans3` | 3 | Clusters: 293 / 14 / 33 companies |
| `kmeans8` | 8 | Best K-means on full data; interpretable profiles |

k selection: Elbow method (suggests k≈8) + Silhouette scores (best range 3–8).
All k values have some negative silhouette sections due to outliers / high dimensionality.

**k=8 cluster highlights (full data):**
- Cluster 0 (8 companies): highest avg price $509, highest EPS
- Cluster 1 (20 companies): highest Cash Ratio (322), mostly Health Care / IT
- Cluster 2 (264 companies): average values — largest group
- Cluster 3 (27 companies): mostly Energy, highest volatility, -16% price drop
- Cluster 7 (3 companies): highest price change 22%, high P/E (possibly overvalued)

#### Outlier-Removed Data (201 rows) ← **Best Model**
| Model | k | Notes |
|---|---|---|
| `kmeansn8` | 8 | Best overall; balanced cluster sizes, actionable investment insights |

**Investment recommendations from k=8 (outlier-removed):**
- **INVEST** (clusters 1, 2, 5): highest Cash Ratio + Price Change; highest ROE; highest price + EPS
- **AVOID** (clusters 3, 6, 7): highest volatility + negative EPS; high P/E overvalued; low overall performance

### Agglomerative (Hierarchical) Clustering

#### Linkage Selection — Cophenetic Correlation Grid Search
| Dataset | Best combo | Cophenetic corr |
|---|---|---|
| Full data | Euclidean + average | ≈ 0.94 |
| Outlier-removed | Euclidean + centroid | (ward chosen for better cluster sizes) |

#### Full Data
| Model | k | Linkage | Notes |
|---|---|---|---|
| `HCmodel8` | 8 | average | 1 big cluster (330) + single-company outlier clusters |
| `HCmodel9` | 9 | average | Same pattern — method acts as outlier detector |
| `HCmodel8W` | 8 | ward | Better: 285 + several multi-company clusters |

Single-company clusters in k=9 (average): Amazon, Netflix, Facebook, Bank of America, Intel, Chesapeake Energy, Alexion Pharma, Alliance Data, Priceline — all outliers in their feature space.

#### Outlier-Removed Data
| Model | k | Linkage | Notes |
|---|---|---|---|
| `HCmodel8wo` | 8 | ward | Still has single-company clusters; HC worse than K-Means here |

### PCA
- `PCA(n_components=9)` explains **95% of variance**
- Not used further: 9 components is too many for practical use on an 11-feature dataset

## Conclusions
- **K-Means (k=8, outlier-removed)** is the best model for actionable insights
- Agglomerative clustering with average linkage works as an **outlier detector** on this dataset
- Ward linkage produces more balanced hierarchical clusters but still less useful than K-Means here
- Energy sector companies consistently cluster in underperforming groups
- IT, Health Care, and Telecom companies dominate the high-performing clusters
