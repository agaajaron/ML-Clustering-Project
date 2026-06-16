# %% [markdown]
# # Data Loading — Stock Market Dataset
#
# Source: stock_data.csv
# 340 S&P 500 companies, 15 columns (11 numeric + 4 categorical)
# Task: Unsupervised clustering — group companies by financial profile

# %%
from config import *

data = pd.read_csv("stock_data.csv")

print("Shape:", data.shape)
print("\nColumn types:")
print(data.dtypes)
print("\nSample:")
data.head(10)
