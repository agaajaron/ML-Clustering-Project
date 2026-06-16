# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *
from utils import remove_outlier

# %%
# Sanity checks
print("Duplicates:", data.duplicated().sum())
print("Missing values:\n", data.isna().sum())

# %%
# Unique value counts per column
for col in data.columns:
    print(f"  {col}: {data[col].nunique()} unique values")

# ── Select numerical features for clustering ──────────────────────────────────
# %%
num_col = data.select_dtypes(include=np.number).columns.tolist()
print("Numeric columns:", num_col)

# ── Scale full dataset ─────────────────────────────────────────────────────────
# %%
scaler = StandardScaler()
subset = data[num_col].copy()
subset_scaled = scaler.fit_transform(subset)
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)
print("Scaled data shape:", subset_scaled_df.shape)

# ── Outlier removal (IQR) for 4 most skewed columns ──────────────────────────
# %%
datan = data.copy()
df  = remove_outlier(datan,  "Current Price")
df1 = remove_outlier(df,    "Net Cash Flow")
df2 = remove_outlier(df1,   "ROE")
datanew = remove_outlier(df2, "Net Income")
print(f"Rows after outlier removal: {len(datanew)} (was {len(data)})")

# ── Scale outlier-removed dataset ─────────────────────────────────────────────
# %%
scaler_wo = StandardScaler()
subsetwo = datanew[num_col].copy()
subset_scaledwo = scaler_wo.fit_transform(subsetwo)
scaledwo_df = pd.DataFrame(subset_scaledwo, columns=subsetwo.columns)
print("Outlier-removed scaled shape:", scaledwo_df.shape)
