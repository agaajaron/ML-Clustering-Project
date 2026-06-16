# %% [markdown]
# # Model Definitions — K-Means and Agglomerative Clustering
#
# Two clustering algorithms applied to stock market data (340 S&P 500 companies):
#
# ── K-Means ──────────────────────────────────────────────────────────────────
#
#  Full data (340 rows):
#   kmeans3  — KMeans(n_clusters=3)  → clusters: 293 / 14 / 33 companies
#   kmeans8  — KMeans(n_clusters=8)  → best K-means result (interpretable profiles)
#
#  Outlier-removed data (201 rows):
#   kmeansn8 — KMeans(n_clusters=8)  ← Best overall model
#     Identifies: clusters to invest (1,2,5) vs clusters to avoid (3,6,7)
#
#  k selection: Elbow method + Silhouette scores (range 2–29)
#  All cases have some negative silhouette sections → outliers / high dimensionality
#
# ── Agglomerative (Hierarchical) Clustering ──────────────────────────────────
#
#  Full data:
#   HCmodel8  — AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="average")
#     Result: 1 big cluster (330) + single-company outlier clusters
#   HCmodel9  — AgglomerativeClustering(n_clusters=9, affinity="euclidean", linkage="average")
#     Similar structure; shows method works as outlier detector
#   HCmodel8W — AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="ward")
#     Better: 285 + several multi-company clusters
#
#  Outlier-removed data (201 rows):
#   HCmodel8wo — AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="ward")
#     Still has single-company cluster; HC worse than K-Means after outlier removal
#
#  Linkage selection: cophenetic correlation grid search
#   Full data best:          euclidean + average  (cophenetic ≈ 0.94)
#   Outlier-removed best:    euclidean + centroid (but ward chosen for better cluster sizes)
#
# ── PCA ──────────────────────────────────────────────────────────────────────
#  PCA(n_components=9) explains 95% of variance — too many components to be useful here

# %%
from preprocessing import *


# ── K-Means ───────────────────────────────────────────────────────────────────

def build_kmeans(n_clusters, random_state=0):
    return KMeans(n_clusters=n_clusters, random_state=random_state)


# ── Agglomerative Clustering ──────────────────────────────────────────────────

def build_agglomerative(n_clusters, linkage_method="average", affinity="euclidean"):
    return AgglomerativeClustering(
        n_clusters=n_clusters, affinity=affinity, linkage=linkage_method
    )


# ── PCA ───────────────────────────────────────────────────────────────────────

def build_pca(n_components=9):
    return PCA(n_components=n_components)
