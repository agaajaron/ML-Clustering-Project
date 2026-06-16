# Stock Market Clustering — K-Means and Agglomerative Analysis

Unsupervised clustering of 340 S&P 500 companies using K-Means and Hierarchical (Agglomerative) clustering to group companies by financial profile and identify investment opportunities.

## Dataset

- **File**: `stock_data.csv` — 340 companies, 15 columns (11 numeric features)
- **Source**: S&P 500 financial indices
- **Features**: Current Price, Price Change, Volatility, ROE, Cash Ratio, Net Cash Flow, Net Income, EPS, Shares Outstanding, P/E Ratio, P/B Ratio
- **Sectors**: 11 GICS sectors (Health Care, IT, Energy, Financials, etc.)

See [data/data.md](data/data.md) for full schema, EDA findings, and cluster profiles.

## Models

| Model | k | Dataset | Key Result |
|---|---|---|---|
| KMeans | 3 | Full (340) | 3 broad groups; cluster 2 = Energy losers |
| KMeans | 8 | Full (340) | Interpretable 8 groups; some overlap |
| **KMeans** | **8** | **Outlier-removed (201)** | **Best model — actionable investment clusters** |
| HC average | 8, 9 | Full | Outlier detector (1 big cluster + singles) |
| HC ward | 8 | Full | Better balance than average linkage |
| HC ward | 8 | Outlier-removed | Still single-company clusters; K-Means preferred |

## Investment Insights (K-Means k=8, outlier-removed)

| Action | Clusters | Why |
|---|---|---|
| Invest | 1, 2, 5 | High Cash Ratio / ROE / EPS; positive price change |
| Avoid | 3, 6, 7 | Highest volatility, negative EPS, overvalued P/E |

## Project Structure

```
clustering-stock-market/
├── data/
│   └── data.md             # Dataset schema, EDA findings, cluster profiles
├── models/
├── notebooks/
│   └── Clustering-Stock-Market.ipynb
├── src/
│   ├── config.py           # All imports
│   ├── data_loader.py      # Load stock_data.csv
│   ├── preprocessing.py    # Select numeric cols, StandardScaler, outlier removal
│   ├── utils.py            # histogram_boxplot, remove_outlier, find_best_cophenetic
│   ├── model.py            # build_kmeans, build_agglomerative, build_pca
│   ├── train.py            # Fit all models, elbow/silhouette search, cophenetic grid
│   ├── evaluate.py         # Cluster profiles, investment recommendations
│   └── visualize.py        # EDA plots, dendrograms, elbow curves, cluster boxplots
└── requirements.txt
```

## Key Techniques

- **K-Means**: Elbow method (`KElbowVisualizer`), Silhouette scores (`SilhouetteVisualizer`), cluster profiling
- **Agglomerative**: Cophenetic correlation grid search (4 distance metrics × 4–6 linkage methods), dendrograms
- **Outlier removal**: IQR ×1.5 filtering on 4 columns
- **PCA**: 9 components explain 95% variance (exploratory only)
- **Yellowbrick**: `KElbowVisualizer`, `SilhouetteVisualizer` for K selection
