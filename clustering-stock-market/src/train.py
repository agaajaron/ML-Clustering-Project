# %% [markdown]
# # Model Training — K-Means and Agglomerative Clustering

# %%
from preprocessing import *
from model import build_kmeans, build_agglomerative, build_pca
from utils import find_best_cophenetic

# ── K-Means: elbow method (full data) ─────────────────────────────────────────
# %%
clusters = range(1, 16)
meanDistortions = []
for k in clusters:
    m = KMeans(n_clusters=k)
    m.fit(subset_scaled_df)
    distortion = (
        sum(np.min(cdist(subset_scaled_df, m.cluster_centers_, "euclidean"), axis=1))
        / subset_scaled_df.shape[0]
    )
    meanDistortions.append(distortion)
    print(f"k={k}  Distortion={distortion:.4f}")

# ── K-Means: KElbowVisualizer ─────────────────────────────────────────────────
# %%
visualizer = KElbowVisualizer(KMeans(), k=(4, 14))
visualizer.fit(subset_scaled_df)
visualizer.show()

# ── K-Means: silhouette scores (full data) ────────────────────────────────────
# %%
sil_score = []
cluster_list = list(range(2, 30))
for n in cluster_list:
    preds = KMeans(n_clusters=n).fit_predict(subset_scaled_df)
    score = silhouette_score(subset_scaled_df, preds)
    sil_score.append(score)
    print(f"k={n}  silhouette={score:.4f}")

# ── K-Means: SilhouetteVisualizer per k ──────────────────────────────────────
# %%
for k in [3, 4, 5, 6, 7, 8, 9, 12]:
    vis = SilhouetteVisualizer(KMeans(k, random_state=1))
    vis.fit(subset_scaled_df)
    vis.show()

# ── K-Means: fit k=3 and k=8 (full data) ─────────────────────────────────────
# %%
kmeans3 = build_kmeans(3)
kmeans3.fit(subset_scaled_df)

kmeans8 = build_kmeans(8)
kmeans8.fit(subset_scaled_df)

# Attach labels to original data
data1 = data.copy()
data1["K_clust"] = kmeans3.labels_

data8 = data.copy()
data8["K_clust"] = kmeans8.labels_

scaled8 = subset_scaled_df.copy()
scaled8["K_clust"] = kmeans8.labels_

print("K-Means (full data) — cluster sizes k=3:", data1["K_clust"].value_counts().to_dict())
print("K-Means (full data) — cluster sizes k=8:", data8["K_clust"].value_counts().to_dict())

# ── Hierarchical: cophenetic grid search (full data) ─────────────────────────
# %%
dm_list = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]
lm_list = ["single", "complete", "average", "weighted"]
scaledhc = subset_scaled_df.copy()

best_full, best_c_full = find_best_cophenetic(scaledhc, dm_list, lm_list)

# ── Hierarchical: euclidean only — all linkage methods ───────────────────────
# %%
lm_all = ["single", "complete", "average", "centroid", "ward", "weighted"]
best_euc, _ = find_best_cophenetic(scaledhc, ["euclidean"], lm_all)

# ── Hierarchical: fit k=8 average linkage (full data) ────────────────────────
# %%
HCmodel8 = build_agglomerative(8, linkage_method="average")
HCmodel8.fit(scaledhc)

datahc8 = data.copy()
scaledhc8 = scaledhc.copy()
datahc8["HC_Clust"] = HCmodel8.labels_
scaledhc8["HC_Clust"] = HCmodel8.labels_
print("HC k=8 avg — cluster sizes:", pd.Series(HCmodel8.labels_).value_counts().to_dict())

# ── Hierarchical: fit k=9 average linkage (full data) ────────────────────────
# %%
HCmodel9 = build_agglomerative(9, linkage_method="average")
HCmodel9.fit(scaledhc)

datahc9 = data.copy()
scaledhc9 = scaledhc.copy()
datahc9["HC_Clust"] = HCmodel9.labels_
scaledhc9["HC_Clust"] = HCmodel9.labels_
print("Outlier single-company clusters:")
print(datahc9.loc[datahc9['HC_Clust'] > 0][["Security", "GICS Sector", "HC_Clust"]])

# ── Hierarchical: fit k=8 Ward linkage (full data) ───────────────────────────
# %%
data8w = data.copy()
scaled8w = subset_scaled_df.copy()
HCmodel8W = build_agglomerative(8, linkage_method="ward")
HCmodel8W.fit(scaled8w)
data8w["HC_Clust"] = HCmodel8W.labels_
scaled8w["HC_Clust"] = HCmodel8W.labels_
print("HC k=8 ward — cluster sizes:", pd.Series(HCmodel8W.labels_).value_counts().to_dict())

# ── K-Means: outlier-removed data ────────────────────────────────────────────
# %%
print("\n--- Outlier-removed dataset ---")
sil_score_wo = []
for n in cluster_list:
    preds = KMeans(n_clusters=n).fit_predict(scaledwo_df)
    score = silhouette_score(scaledwo_df, preds)
    sil_score_wo.append(score)
    print(f"k={n}  silhouette={score:.4f}")

kmeansn8 = build_kmeans(8)
kmeansn8.fit(scaledwo_df)

datanew8 = datanew.copy()
scaledwo_new = scaledwo_df.copy()
datanew8["K_clust"] = kmeansn8.labels_
scaledwo_new["K_clust"] = kmeansn8.labels_
print("K-Means outlier-removed k=8 sizes:", pd.Series(kmeansn8.labels_).value_counts().to_dict())

# ── Hierarchical: outlier-removed, cophenetic search ─────────────────────────
# %%
best_wo, _ = find_best_cophenetic(scaledwo_df, dm_list, lm_list)
find_best_cophenetic(scaledwo_df, ["euclidean"], lm_all)

# ── Hierarchical: outlier-removed, ward k=8 ──────────────────────────────────
# %%
HCmodel8wo = build_agglomerative(8, linkage_method="ward")
HCmodel8wo.fit(scaledwo_df)

scaledhcwo = scaledwo_df.copy()
datawo8 = subsetwo.copy()
datawo8["HC_Clusters"] = HCmodel8wo.labels_
scaledhcwo["HC_Clusters"] = HCmodel8wo.labels_
print("HC outlier-removed k=8 ward:", pd.Series(HCmodel8wo.labels_).value_counts().to_dict())

# ── PCA (9 components) ────────────────────────────────────────────────────────
# %%
pca = build_pca(n_components=9)
X_pca = pca.fit_transform(scaledwo_new)
print(f"\nPCA 9 components explain {pca.explained_variance_ratio_.sum():.1%} of variance")
