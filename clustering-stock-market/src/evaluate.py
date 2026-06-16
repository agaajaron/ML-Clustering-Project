# %% [markdown]
# # Cluster Profiling and Evaluation

# %%
from train import *
from utils import cluster_profile_summary

# ── K-Means k=3 — full data ───────────────────────────────────────────────────
# %%
print("=== K-Means k=3 (full data) ===")
profile3 = cluster_profile_summary(data1, "K_clust", num_col)
profile3.style.highlight_max(color="lightgreen", axis=0)

# %%
print("Sector distribution per cluster:")
data1.groupby(["K_clust", "GICS Sector"]).mean()

# ── K-Means k=8 — full data ───────────────────────────────────────────────────
# %%
print("=== K-Means k=8 (full data) ===")
profile8 = cluster_profile_summary(data8, "K_clust", num_col)
profile8["No"] = data8.groupby("K_clust")["Current Price"].count().values
profile8.style.highlight_max(color="lightgreen", axis=0)

# %%
print("Companies per sector per cluster:")
data8.groupby(["K_clust", "GICS Sector"]).Security.count()

# ── Hierarchical k=8 average — full data ─────────────────────────────────────
# %%
print("=== Hierarchical k=8 average linkage (full data) ===")
hc8_profile = cluster_profile_summary(datahc8, "HC_Clust", num_col)
hc8_profile["No_in_clust"] = datahc9.groupby("HC_Clust")["ROE"].count().values
hc8_profile.style.highlight_max(color="lightgreen", axis=0)

# %%
print("Companies per sector per HC cluster:")
datahc8.groupby(["HC_Clust", "GICS Sector"]).Security.count()

# ── Hierarchical k=9 — single-company outlier clusters ────────────────────────
# %%
print("=== HC k=9: single-company clusters (outlier detection mode) ===")
print(datahc9.loc[datahc9['HC_Clust'] > 0][
    ["Security", "GICS Sector", "HC_Clust", "Current Price", "Volatility", "ROE"]
])

# ── Hierarchical k=8 Ward — full data ────────────────────────────────────────
# %%
print("=== Hierarchical k=8 Ward linkage (full data) ===")
hc8w_profile = cluster_profile_summary(data8w, "HC_Clust", num_col)
hc8w_profile.style.highlight_max(color="lightgreen", axis=0)

# ── K-Means k=8 — outlier-removed ← Best model ───────────────────────────────
# %%
print("=== K-Means k=8 outlier-removed — BEST MODEL ===")
profile8n = cluster_profile_summary(datanew8, "K_clust", num_col)
profile8n["No"] = datanew8.groupby("K_clust")["Current Price"].count().values
profile8n.style.highlight_max(color="lightgreen", axis=0)

# %%
# Investment recommendations — companies in good clusters
print("\n--- Cluster 1 (high Cash Ratio + Price Change) ---")
print(datanew8.loc[datanew8['K_clust'] == 1][["Security", "GICS Sector", "Current Price", "Price Change"]])

print("\n--- Cluster 2 (highest ROE) ---")
print(datanew8.loc[datanew8['K_clust'] == 2][["Security", "GICS Sector", "Current Price", "ROE"]])

print("\n--- Cluster 5 (highest price + EPS) ---")
print(datanew8.loc[datanew8['K_clust'] == 5][["Security", "GICS Sector", "Current Price", "Earnings Per Share"]])

# ── Hierarchical k=8 Ward — outlier-removed ───────────────────────────────────
# %%
print("=== HC k=8 Ward outlier-removed ===")
hcwo_profile = cluster_profile_summary(datawo8, "HC_Clusters", num_col)
hcwo_profile["No_in_clust"] = datawo8.groupby("HC_Clusters")["ROE"].count().values
hcwo_profile.style.highlight_max(color="lightgreen", axis=0)
