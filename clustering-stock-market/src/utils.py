# %%
from config import *


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter")
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1
    fence_low  = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    return df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]


def find_best_cophenetic(scaled_df, distance_metrics, linkage_methods):
    high_c = 0
    best = [None, None]
    for dm in distance_metrics:
        for lm in linkage_methods:
            Z = linkage(scaled_df, metric=dm, method=lm)
            c, _ = cophenet(Z, pdist(scaled_df))
            print(f"Cophenetic ({dm.capitalize()}, {lm}): {c:.4f}")
            if c > high_c:
                high_c = c
                best = [dm, lm]
    print(f"\nBest: {best[0]} distance + {best[1]} linkage → cophenetic = {high_c:.4f}")
    return best, high_c


def cluster_profile_summary(data_orig, label_col, num_col):
    profile = data_orig.groupby(label_col).mean()
    profile["count"] = data_orig.groupby(label_col)[num_col[0]].count().values
    return profile
