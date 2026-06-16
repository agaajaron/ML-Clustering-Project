# %% [markdown]
# # Visualization — EDA, Elbow Curves, Dendrograms, Cluster Plots

# %%
from evaluate import *
from utils import histogram_boxplot

# ── EDA — univariate distributions (raw data) ─────────────────────────────────
# %%
for col in num_col:
    histogram_boxplot(data, col)

# ── EDA — sector-level barplot ───────────────────────────────────────────────
# %%
sns.catplot(x="GICS Sector", y="Current Price", data=data, kind="box", height=5, aspect=3)
plt.xticks(rotation=90); plt.show()

sns.catplot(x="GICS Sector", y="Cash Ratio", data=data, kind="box", height=7, aspect=2.2)
plt.show()

sns.catplot(x="GICS Sector", y="P/E Ratio", data=data, kind="box", height=7, aspect=2.2)
plt.show()

# ── EDA — barplot + KDE side-by-side ─────────────────────────────────────────
# %%
for feature in ["Cash Ratio", "P/E Ratio"]:
    plt.figure(figsize=(20, 9))
    plt.subplot(1, 2, 1)
    sns.barplot(data=data, x="GICS Sector", y=feature)
    plt.xticks(rotation=90)
    plt.subplot(1, 2, 2)
    sns.kdeplot(data=data, x=feature, hue="GICS Sector", common_norm=False)
    plt.xticks(rotation=90)
    plt.tight_layout(); plt.show()

# ── EDA — pairplot and correlation heatmap ────────────────────────────────────
# %%
sns.pairplot(data, diag_kind='kde')
plt.show()

plt.figure(figsize=(15, 7))
sns.heatmap(data.corr(), annot=True, vmin=-1, vmax=1, cmap="Spectral")
plt.title("Correlation Matrix — raw data")
plt.show()

sns.pairplot(data, diag_kind='kde', hue='GICS Sector')
plt.show()

# ── EDA — sector aggregates ──────────────────────────────────────────────────
# %%
datagroupsector = data.groupby(['GICS Sector'])
print(datagroupsector.mean())
print(datagroupsector.median())

# ── Scaled data distributions ────────────────────────────────────────────────
# %%
for col in num_col:
    histogram_boxplot(subset_scaled_df, col, kde=True, figsize=(10, 5))

# ── Elbow curve ──────────────────────────────────────────────────────────────
# %%
plt.plot(list(range(1, 16)), meanDistortions, "bx-")
plt.xlabel("k"); plt.ylabel("Average Distortion")
plt.title("Elbow Method — full data"); plt.show()

plt.plot(cluster_list, sil_score)
plt.xlabel("k"); plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores — full data"); plt.show()

plt.plot(cluster_list, sil_score_wo)
plt.xlabel("k"); plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores — outlier-removed data"); plt.show()

# ── Dendrograms — full data, all linkage methods ──────────────────────────────
# %%
lm_all = ["single", "complete", "average", "centroid", "ward", "weighted"]
fig, axs = plt.subplots(len(lm_all), 1, figsize=(15, 30))
for i, method in enumerate(lm_all):
    Z = linkage(scaledhc, metric="euclidean", method=method)
    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")
    c, _ = cophenet(Z, pdist(scaledhc))
    axs[i].annotate(f"Cophenetic\n{c:0.2f}", (0.80, 0.80), xycoords="axes fraction")
plt.tight_layout(); plt.show()

# ── Dendrograms — outlier-removed ────────────────────────────────────────────
# %%
fig, axs = plt.subplots(len(lm_all), 1, figsize=(15, 30))
for i, method in enumerate(lm_all):
    Z = linkage(scaledwo_df, metric="euclidean", method=method)
    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram outlier-removed ({method.capitalize()} Linkage)")
    c, _ = cophenet(Z, pdist(scaledwo_df))
    axs[i].annotate(f"Cophenetic\n{c:0.2f}", (0.80, 0.80), xycoords="axes fraction")
plt.tight_layout(); plt.show()

# ── Cluster boxplots — K-Means k=3 ──────────────────────────────────────────
# %%
fig, axes = plt.subplots(1, 11, figsize=(20, 5))
fig.suptitle("K-Means k=3 — Boxplot per cluster")
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=data1[num_col[ii]], x=data1["K_clust"])
fig.tight_layout(pad=2.0); plt.show()

sns.pairplot(scaled.assign(K_clust=kmeans3.labels_), hue='K_clust')
plt.show()

# ── Cluster boxplots — K-Means k=8 (full data) ──────────────────────────────
# %%
fig, axes = plt.subplots(1, 11, figsize=(20, 5))
fig.suptitle("K-Means k=8 — Boxplot per cluster")
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=data8[num_col[ii]], x=data8["K_clust"])
fig.tight_layout(pad=2.0); plt.show()

sns.pairplot(data8, hue='K_clust', palette='hls')
plt.show()

# ── Cluster boxplots — HC k=8 Ward (full data) ───────────────────────────────
# %%
sns.pairplot(data8w, hue='HC_Clust', palette='hls')
plt.show()

# ── Cluster boxplots — K-Means k=8 (outlier-removed) — best ─────────────────
# %%
fig, axes = plt.subplots(1, 11, figsize=(24, 7))
fig.suptitle("K-Means k=8 (outlier-removed) — Boxplot per cluster")
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=datanew8[num_col[ii]], x=datanew8["K_clust"])
fig.tight_layout(pad=2.0); plt.show()

# ── Outlier-removed correlation heatmap ──────────────────────────────────────
# %%
plt.figure(figsize=(15, 7))
sns.heatmap(datanew.corr(), annot=True, vmin=-1, vmax=1, cmap="Spectral")
plt.title("Correlation Matrix — outlier-removed data")
plt.show()

# ── HC outlier-removed boxplots ──────────────────────────────────────────────
# %%
fig, axes = plt.subplots(1, 11, figsize=(22, 7))
fig.suptitle("HC k=8 Ward (outlier-removed) — Boxplot per cluster")
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=datawo8[num_col[ii]], x=datawo8["HC_Clusters"])
fig.tight_layout(pad=2.0); plt.show()
