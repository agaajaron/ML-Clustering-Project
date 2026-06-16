# ML-Clustering-Project

Unsupervised machine learning project applying K-Means and Agglomerative (Hierarchical) clustering to S&P 500 stock market data to identify company groups and guide investment decisions.

## Project

| Folder | Dataset | Task | Best Model |
|---|---|---|---|
| [clustering-stock-market](clustering-stock-market/) | stock_data.csv (340 companies) | Cluster companies by financial profile | K-Means k=8 on outlier-removed data |

## Project Structure

```
clustering-stock-market/
├── data/
│   └── data.md          # Dataset description, features, cluster summaries
├── models/
├── notebooks/
├── src/
│   ├── config.py        # All imports (sklearn, scipy, yellowbrick)
│   ├── data_loader.py   # Load stock_data.csv
│   ├── preprocessing.py # Numeric selection, StandardScaler, IQR outlier removal
│   ├── utils.py         # histogram_boxplot, remove_outlier, cophenetic grid search
│   ├── model.py         # build_kmeans, build_agglomerative, build_pca
│   ├── train.py         # Fit all models, elbow/silhouette/cophenetic analysis
│   ├── evaluate.py      # Cluster profiles, investment recommendations
│   └── visualize.py     # EDA, dendrograms, elbow curves, cluster boxplots
└── requirements.txt
```

## Key Findings

- **K-Means (k=8, outlier-removed)** produces the most actionable clusters
- Hierarchical clustering with average linkage behaves as an **outlier detector** on this data
- **Energy sector** companies consistently fall in underperforming clusters
- **IT / Health Care / Telecom** companies dominate high-performing clusters
- Clusters 1, 2, 5 (outlier-removed) are recommended for portfolio building
