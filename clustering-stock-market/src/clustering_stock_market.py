# %% [markdown]
# ## Project: Stock Market K-means and Agglomerative Clustering Analysis of data for  340 companies

# %% [markdown]
# ## Importing necessary libraries and data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# %matplotlib inline
from scipy.stats import zscore
import seaborn as sns

# %%
# to scale the data using z-score
from sklearn.preprocessing import StandardScaler
# to compute distances
from scipy.spatial.distance import pdist

# to perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# %%

# to compute distances
from scipy.spatial.distance import cdist

# to perform k-means clustering and compute silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# to visualize the elbow curve and silhouette scores
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# %% [markdown]
# ## What kind of data are we dealing with:

# %% [markdown]
# Data Dictionary
# 
# Ticker Symbol: An abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market
# 
# Company: Name of the company
# 
# GICS Sector: The specific economic sector assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# 
# GICS Sub Industry: The specific sub-industry group assigned to a company by the Global Industry Classification Standard (GICS) that best defines its business operations
# 
# Current Price: Current stock price in dollars
# 
# Price Change: Percentage change in the stock price in 13 weeks
# 
# Volatility: Standard deviation of the stock price over the past 13 weeks
# 
# ROE: A measure of financial performance calculated by dividing net income by shareholders' equity (shareholders' equity is equal to a company's assets minus its debt)
# 
# Cash Ratio: The ratio of a company's total reserves of cash and cash equivalents to its total current liabilities
# 
# Net Cash Flow: The difference between a company's cash inflows and outflows (in dollars)
# 
# Net Income: Revenues minus expenses, interest, and taxes (in dollars)
# 
# Earnings Per Share: Company's net profit divided by the number of common shares it has outstanding (in dollars)
# 
# Estimated Shares Outstanding: Company's stock currently held by all its shareholders
# 
# P/E Ratio: Ratio of the company's current stock price to the earnings per share
# 
# P/B Ratio: Ratio of the company's stock price per share by its book value per share (book value of a company is the net difference between that company's total assets and total liabilities)
# 

# %%
# reading the CSV file into pandas dataframe
data = pd.read_csv("stock_data.csv") 
data.head(10)

# %% [markdown]
# ## Note: we will only use numerical data for clustering.  

# %%
for col in data.columns:
    print("Number of unique values in ", col, len(data[col].unique()))

# %% [markdown]
# The companies are grouped by GICS Sector - there are 11 of them andthey represent a natural groupidn of the companies. 
# I will consider this feature in analysis as someof the ther paraeter/indices values can depend on given sector. 

# %% [markdown]
# ## Comment: I was interested if one of the non-numerical features could be encoded. I think sector could be label-encoded. I will try it (I will onsider 2 cases for each -best- clustering  method).  The other non-numerical columns will not be included in clustering. 
# 

# %% [markdown]
# ## Data Overview
# 
# - Observations
# - Sanity checks

# %%
data.describe()

# %% [markdown]
# ### Data will need scaling. 

# %%
data.info()

# %% [markdown]
# ## Checking for duplicated data

# %%
data[data.duplicated()].count()

# %% [markdown]
# ## Checking for missing data

# %%
# checking missing values
data.isna().sum()

# %% [markdown]
# ## Conclusion: Dataset has no duplicated or missing values. 

# %% [markdown]
# Clean data set with informations about companies. 
# 
# 15 columns. 340 rows. 4 columns object tye I will not consider for clustering at first. Later I will label encode the sector variable- Iam curious if that affects the clustering (it can and sector in itself is a kind of 'industry-based-grouping' method).

# %% [markdown]
# ## Outlier issue

# %% [markdown]
# Outliers can affect k-means clustering (Euclidean distance) - outliers can group as separate cluster or they can cause other clusters to merge. One can consider method less sensitive to outliers such as k medians or use DBSCAN.
# 
# I will see if outliers cause a problem. There are many outliers and removing all of them causes loss of data and it becomes different problem to study. 
# 
# Agglomerative clustering can use various measures to calculate distance between two clusters. Regarding outliers 
#  The Applied Multivariate Statistics book says :
# "... single link method which is also very sensitive to errors of measurement, but somewhat robust to outliers. The complete link and Wardâ€™s method tend to find compact clusters of nearly equal size with the clustering solution adversely affected by outliers."
# 

# %% [markdown]
# I will comeback to outlier issue after I see how the results for original data set look like. 

# %% [markdown]
# ## Exploratory Data Analysis (EDA)
# 
# - EDA is an important part of any project involving data.
# - It is important to investigate and understand the data better before building a model with it.
# - A few questions have been mentioned below which will help you approach the analysis in the right manner and generate insights from the data.
# - A thorough analysis of the data, in addition to the questions mentioned below, should be done.

# %% [markdown]
# ## Univariate analysis

# %% [markdown]
# Note: I plot separately plots for each variable because when I used iteration it did not look good for html save.

# %%
# function to plot a boxplot and a histogram along the same scale.


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid= 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # boxplot will be created and a star will indicate the mean value of the column
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram

# %% [markdown]
# ### Histograms and boxplots for each variable

# %%
histogram_boxplot(data,'Current Price')

# %% [markdown]
# Comment: Right skewed distrbution with many outliers.

# %%
histogram_boxplot(data,'Price Change')

# %% [markdown]
# Comment: Distribution for prie change is very slightly sewed tothe left,and has many outliers.

# %%
histogram_boxplot(data,'Volatility')

# %% [markdown]
# Comment: Right Skewed distribution with quitea few outliers.

# %%
histogram_boxplot(data,'ROE')

# %% [markdown]
# Comment: Right skewed distributionwith many distanct outliers. 

# %%
histogram_boxplot(data,'Cash Ratio')

# %% [markdown]
# Comment Right Skewed distribution with some outliers. The distribution looks more lke half of the normal distribution. 

# %%
histogram_boxplot(data,'Net Cash Flow')

# %% [markdown]
# Comment: distribution has many outliers. 

# %%
histogram_boxplot(data,'Net Income')

# %% [markdown]
# Comment; right skewed distribution with many outliers.

# %%
histogram_boxplot(data,'Earnings Per Share')

# %%
histogram_boxplot(data,'Estimated Shares Outstanding')

# %% [markdown]
# Riht skewed distribtuion with many outliers. 

# %%
histogram_boxplot(data,'P/E Ratio')

# %% [markdown]
# Right skewed distribution. SOmeoutlier. 3 very distant outliers.

# %%
histogram_boxplot(data,'P/B Ratio')

# %% [markdown]
# Conclusions:
# 1) some features have skewed dirtibutions
# 
# 2) Some feaures have distribution that resembles normal distribution
# 
# 3) Many otliers for majoriy of the variables

# %% [markdown]
# ## Bivariate analysis

# %%
sns.pairplot(data,diag_kind='kde')

# %% [markdown]
# ### Conclusions based on pairplot: 
#     
# 1. Volatility, Estimated Shares Outstanding and P/E Ratio kde seems to show some clusters. For the other features 
# one cannot clearly see clusters onthe plot, there seem to be one bog cluster and maybe some small ones.
# 
# 2. Most of the featres do not seem to be linearly dependent. Except NetIncome vs Estimated Shares Outstanding - where one can see some linear dependence. Maybe Price Change vs Volatility also have inversely proportional linear dependence. 

# %%
plt.figure(figsize=(15,7))
sns.heatmap(data.corr(),annot=True,vmin=-1,vmax=1,cmap="Spectral")
plt.show()

# %% [markdown]
# ### Conclusion
# 
# Correlation matrix confirms the  previous conclusions - 
# 
# 1)Most of the featres do not seem to be linearly dependent. 
# 
# 2) Except some weaker correlation between Net Income vs Estimated Shares Outstanding AND NetIncome and Earnings per Share, positive correlation. 
# 
# 3) Also Price Change vs Volatility are inversely proportional correlated as could be seen on pairplot. 

# %% [markdown]
# Plotting pairplot (using hue="GICS Sector") to see if different business sectors cluster together - naturally.
# (Features like ROE depend on the sector).

# %%
sns.pairplot(data,diag_kind='kde',hue='GICS Sector')

# %% [markdown]
# The price distrbutions does not differ between sectors - a lot  ... 
# Health Care Real Estate and Consumer Discretionary have the most outliers.

# %%


# %% [markdown]
# ## Conclusions:
#     
# A. Pairplot for all the data 
# 
# 1. One cannot see clearly linear dependence between features
# 
# 2. One cannot see any natural clustersin the data so one can expect that 
# the division and separation into meaningful clusters will be more difficult
#     
# B. Correlation matrix confirms the  previous conclusions - 
# 
# 1) Most of the featres do not seem to be linearly dependent. 
# 
# 2) Except some weaker correlation between Net Income vs Estimated Shares Outstanding AND Net Income and Earnings per Share, positive correlation. 
# 
# 3) Also Price Change vs Volatility are inversely proportional correlated as could be seen on pairplot. 
# 
# C. Pairplot with marked different industry Sectors data
# 
# 1. The price distrbutions does not differ between sectors - a lot ... 
# 
# 2. Health Care Real Estate and Consumer Discretionary have the most outliers.
# 
# 
# 

# %%


# %% [markdown]
# ## Questions and Answers

# %% [markdown]
# **Questions**:
# 
# 1. What does the distribution of stock prices look like?
# 2. The stocks of which economic sector have seen the maximum price increase on average?
# 3. How are the different variables correlated with each other?
# 4. Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. How does the average cash ratio vary across economic sectors?
# 5. P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. How does the P/E ratio vary, on average, across economic sectors?

# %% [markdown]
# ## Question 1. What does the distribution of stock prices look like?

# %%
histogram_boxplot(data,'Current Price')

# %%
sns.catplot(x="GICS Sector", y="Current Price", data=data, kind="box", height=5, aspect=3)

# %% [markdown]
# ### Answer: The distribution of the stock prices is right skewed with outliers.  The highest average stock price is for Health Care sector (132 dollars), 2nd highest price is Consumer Discretionary (128 dollars), while the lowest  averagestock price is for Telecommunication Services (33 dollars).

# %% [markdown]
# ## NOTE:questions about sectors I will answer using the groupby calculations below:

# %%
datagroupsector = data.groupby(['GICS Sector'])

# %%
datagroupsector.mean()

# %% [markdown]
# Comments: Highest average price is for Consumer Discretionary and Health care.
# The worst price change is for energy (-10%), and best for Health Care.
# Highest average volatility is for Energy sector.
# Largest Cash Ratio is again for Health Care and IT and Telecom. Service.
# 
# 

# %% [markdown]
# ## Question 2. The stocks of which economic sector have seen the maximum price increase on average?

# %%
datagroupsector.median()

# %% [markdown]
# ## Answer: On average maximum price increase on average was for Heath Care sector (10.3 %).

# %% [markdown]
# ## Question 3 How are the different variables correlated with each other?

# %%
sns.pairplot(data,diag_kind='kde')

# %%
plt.figure(figsize=(15,7))
sns.heatmap(data.corr(),annot=True,vmin=-1,vmax=1,cmap="Spectral")
plt.show()

# %% [markdown]
# ## Answer:

# %% [markdown]
# ## A. Conclusions about correlation based on pairplot: *
# 
# 1)    Volatility, Estimated Shares Outstanding and P/E Ratio kde seems to show some clusters. For the other features one cannot clearly see clusters onthe plot, there seem to be one bog cluster and maybe some small ones.
# 
# 2)    Most of the featres do not seem to be linearly dependent. Except NetIncome vs Estimated Shares Outstanding - where one can see some linear dependence. Maybe Price Change vs Volatility also have inversely proportional linear dependence.
# 
# 

# %% [markdown]
# ## B. Conclusion based on correlation matrix 
# 
# 1)weak correlation between Net Income vs Estimated Shares Outstanding,positive correlation 
# 2) weak correlation between NetIncome and Earnings per Share, positive correlation. 
# 3)  Price Change vs Volatility are inversely proportional correlated

# %% [markdown]
# ## Question 4: Cash ratio provides a measure of a company's ability to cover its short-term obligations using only cash and cash equivalents. How does the average cash ratio vary across economic sectors?

# %%
sns.catplot(x="GICS Sector", y="Cash Ratio", data=data, kind="box", height=7, aspect=2.2)

# %%
datagroupsector["Cash Ratio"].mean()

# %%
plt.figure(figsize=(20, 9))

plt.subplot(1, 2, 1)
sns.barplot(data=data, x="GICS Sector", y="Cash Ratio")
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.kdeplot(data=data, x="Cash Ratio", hue="GICS Sector", common_norm=False)
plt.xticks(rotation=90)

plt.show()


# %% [markdown]
# ## Answer: The largest average cash ratio is highest for Information Technology (150),Telecommunication Services (117) and Health care (104). The smallest average cash ratio is for Utilities (13).

# %% [markdown]
# ## Question 5. P/E ratios can help determine the relative value of a company's shares as they signify the amount of money an investor is willing to invest in a single share of a company per dollar of its earnings. How does the P/E ratio vary, on average, across economic sectors?

# %%
sns.catplot(x="GICS Sector", y="P/E Ratio", data=data, kind="box", height=7, aspect=2.2)

# %%
datagroupsector["P/E Ratio"].mean()

# %%
plt.figure(figsize=(20, 9))

plt.subplot(1, 2, 1)
sns.barplot(data=data, x="GICS Sector", y="P/E Ratio")
plt.xticks(rotation=90)

plt.subplot(1, 2, 2)
sns.kdeplot(data=data, x="P/E Ratio", hue="GICS Sector", common_norm=False)
plt.xticks(rotation=90)

plt.show()


# %% [markdown]
# ## Answer. The average values for P/E Ratio fordifferent sectors ranges from 12  to 73. 
# Largest average P/E Ratio is for sectors: Energy (73), Inf. Tech. (44) and Real Estate (43).
# The smallest average PE Ratio is for Telecom.Services (12). 
# 
# The median values for P/E Ratio is the highest for Energy (93) and Real Estate (34). For  rest of the sectors median P/E ratio falls into narrower range of 14-25.

# %% [markdown]
# ## Prepaing the data for clustering - because the features have very different ranges wened to scale the data. I will use the Standard Scaler.

# %% [markdown]
# ### Scaling the data using standard scaler 

# %%
# selecting numerical columns
num_col = data.select_dtypes(include=np.number).columns.tolist()

# %%
# Scaling the data set before clustering
scaler = StandardScaler()
subset = data[num_col].copy()
subset_scaled = scaler.fit_transform(subset)

# %%
# Creating a dataframe from the scaled data
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)

# %%
subset_scaled_df

# %% [markdown]
# ## EDA
# 
# - It is a good idea to explore the data once again after manipulating it.

# %%
histogram_boxplot(subset_scaled_df, "Current Price", kde=True, figsize=(10, 5))

# %%
histogram_boxplot(subset_scaled_df, "Price Change", kde=True, figsize=(10, 5))

# %%
histogram_boxplot(subset_scaled_df, "Volatility", kde=True, figsize=(10, 5))

# %%
histogram_boxplot(subset_scaled_df, "ROE", kde=True, figsize=(14, 5))

# %%
histogram_boxplot(subset_scaled_df, "Cash Ratio", kde=True, figsize=(10, 5))

# %%
histogram_boxplot(subset_scaled_df, "Net Cash Flow", kde=True, figsize=(15, 5))

# %%
histogram_boxplot(subset_scaled_df, "Net Income", kde=True, figsize=(15, 5))

# %%
histogram_boxplot(subset_scaled_df, "Earnings Per Share", kde=True, figsize=(15, 5))

# %%
histogram_boxplot(subset_scaled_df, "Estimated Shares Outstanding", kde=True, figsize=(10, 5))

# %%
histogram_boxplot(subset_scaled_df, "P/E Ratio", kde=True, figsize=(10, 5))

# %%
histogram_boxplot(subset_scaled_df, "P/B Ratio", kde=True, figsize=(10, 5))

# %% [markdown]
# Scaled data looks as one expects. 

# %% [markdown]
# ## K-means Clustering

# %%
clusters = range(1, 16)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(subset_scaled_df)
    prediction = model.predict(subset_scaled_df)
    distortion = (
        sum(
            np.min(cdist(subset_scaled_df, model.cluster_centers_, "euclidean"), axis=1)
        )
        / subset_scaled_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)

# %% [markdown]
# Conclusion - it seems based on the Elbow method number of clusters s 4 (maybe 8 and maybe 12). 
# There are 11 industry sectors so I will consider all 3 possibilities.

# %% [markdown]
# Now I check the Elbow methd using KElbowVisualizer from Yellowbrick library
# 

# %%
model1 = KMeans()
visualizer = KElbowVisualizer(model1, k=(4,14))

visualizer.fit(subset_scaled_df)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# %% [markdown]
# Conclusion - both Elbow mehtod visualization show that 8 is possible value for clusters.

# %% [markdown]
# No I check the silhouette score

# %% [markdown]
# The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 
# 
# 1) The Silhouette coefficient of +1 indicates that the sample is far away from the neighboring clusters.
# 
# 2) The Silhouette coefficient of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters.
# 
# 3) Silhouette coefficient <0 indicates that those samples might have been assigned to the wrong cluster or are outliers.

# %%
sil_score = []
cluster_list = list(range(2, 30))
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict((subset_scaled_df))
    # centers = clusterer.cluster_centers_
    score = silhouette_score(subset_scaled_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

plt.plot(cluster_list, sil_score)

# %% [markdown]
# Conclusion - the best range for cluster number  seems to be 3-8. If I would guess from this figure then 3 or 7,8 look like best options.But I will compare the cases fom 3 to 8 clusters below to pick the best case. 
# (Please note that this plot changes when I re-run it so it depends on initial data). 

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(3, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %% [markdown]
# Negative  Silhouette coefficient values for 2 out of 3 clusters is  rather poor clustering example

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(4, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %% [markdown]
# Negative values for silhoutette coefficient for 3 outof 4 clusters - example of poor results again. 

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(5, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %% [markdown]
# Similarly poor clustering. 

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(6, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %% [markdown]
# Still one big cluster and many small clusters with negative values of silhoutte coefficient. Not a good result. 

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(7, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(8, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(9, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %% [markdown]
# I add one more example for k=12 clusters.I do not expect good results here.

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(12, random_state=1))
visualizer.fit(subset_scaled_df)
visualizer.show()

# %% [markdown]
# ## Conclusion: It seems the best is k=8 clusters. I will also consider 3 cluster case for cluster profiling. All of the attempts had negative silhouette sections which one can attribute to outliers.
# 
# Comment: All of the clustering cases have instances with negatic silhuette score so I cannot with K-means clustering 
#     get a a better separation of clusters with the data provided. It might be due to outliers or due to high 
#     dimensionality. 
#     I can try 
#     1) remove outliers - but this is original data and there are not too many rows so I do not expect better results if I 
#     drop data 
#     2) PCA and clusters using features that are not dependent. 
#     3) focus on few features. 
#     
#   For comparison I show silhuette plot for one of the larger number of clusters(k=12).

# %% [markdown]
# As one can see the results arenot getting better with the larger number of clusters.

# %% [markdown]
# All have portion with negative silhuette. The smalles negative portion have 3 and 8 clustering cases. 

# %%
data2=subset_scaled_df.copy()

# %% [markdown]
# ## Cluster profiling

# %% [markdown]
# ## Case 1 : Number of clusters =3

# %% [markdown]
# I start with K-mean clustering for 3 clusters.

# %%
kmeans3 = KMeans(n_clusters=3, random_state=0)
kmeans3.fit(data2)

# %%
data1=data.copy()

# %%
# adding kmeans cluster labels to the original dataframe
data1["K_clust"] = kmeans3.labels_

# %%
cluster_profile = data1.groupby("K_clust").mean()

# %%
cluster_profile["count_in_each_segments"] = (
    data1.groupby("K_clust")["Current Price"].count().values
)

# %%
# lets display cluster profile
cluster_profile.style.highlight_max(color="lightgreen", axis=0)

# %% [markdown]
# The largest cluster (denoted as 0) containing 293 companies has the highest average price and the highest 'earnings per share'

# %% [markdown]
# The Two remaining clusters are smaller cluster denoted as 1 has 14 companies and the cluster denotes as 2 has 33 companies.

# %% [markdown]
# Cluster 1 has the highest average price change and the by far largest cash ratio and net cash flow, net income and estimated shares outstanding. This cluster has slightly lower average earnings per share. 
# 
# Cluster 2 has the highest average volatility, ROE P/ERato and P/B Ratio, but negative price change, net income (with positive net cash flow) and earnings per share. 
# 
# Cluster2 seems to group companies that have the highest volatility and for the period consdered these companies brought losses to the investors but judging from P/E ratio these companies represnt high relative value for investors. 
# 
# I have checked that cluster number 2 is dominated by companies that belong to Energy sector.
# 

# %%
fig, axes = plt.subplots(1, 11, figsize=(20, 5))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=data1[num_col[counter]], x=data1["K_clust"])
    counter = counter + 1

fig.tight_layout(pad=2.0)

# %%
scaled=subset_scaled_df.copy()

# %%
# adding kmeans cluster labels to the original dataframe
scaled["K_clust"] = kmeans3.labels_

# %%
scaled

# %%
sns.pairplot(scaled,hue='K_clust') 

# %% [markdown]
# Pairplot shows  scatterplots with cluster separated data for given 2 variables.  

# %% [markdown]
# Comment: Pairplot show we have overlapping clusters. Becaus of that onemight not get good value for metrics 
#     related to separation of clusters. I will later try bigger number of clusters. 

# %% [markdown]
# I plot now boxplots for numerical variables using scaled dataframe data.

# %%
fig, axes = plt.subplots(1, 11, figsize=(20, 5))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=scaled[num_col[counter]], x=scaled["K_clust"])
    counter = counter + 1

fig.tight_layout(pad=2.0)

# %% [markdown]
# I caclulate averages for each sector in each cluster.

# %%
data1.groupby(["K_clust","GICS Sector"]).mean()

# %% [markdown]
# Comment: Cluster 2 acros allsectors groups companies with highest volatility. It has the highest average price for Health care and consumer discretionary.
# And only those 2 sectorshad positive price change. The companies from other sectors had negative price change. 
#     
# This feature suggests that maybe this cluster is not very homogenous  - since only when I look at 2 features the companies seem to fall into separate 2 groups. 

# %% [markdown]
# Comment: Cluster 0 has companies from all of the sectors. Cluster 1 has companies from 7 sectors. 
#     cluster 2 has companies from 6 sectors with the biggest part of the cluster beloging to energy sector companies. 

# %% [markdown]
# ## Case 2 : Number of clusters k=8

# %%
data8=data.copy()

# %%
scaled8=subset_scaled_df.copy()

# %%
kmeans8 = KMeans(n_clusters=8, random_state=0)
kmeans8.fit(scaled8)

# %% [markdown]
# Adding K-means labels to original dataframe

# %%
# adding kmeans cluster labels to the original dataframe
data8["K_clust"] = kmeans8.labels_

# %% [markdown]
# Analysis of cluster profiles

# %%
cluster_profile8 = data8.groupby("K_clust").mean()

# %%
cluster_profile8["No"] = (
    data8.groupby("K_clust")["Current Price"].count().values
)

# %%
# lets display cluster profile
cluster_profile8.style.highlight_max(color="lightgreen", axis=0)

# %%
data8.groupby(['K_clust','GICS Sector']).Security.count()

# %%
import warnings
warnings.filterwarnings('ignore')
sns.pairplot(data8,hue='K_clust',palette='hls') 

# %% [markdown]
# Comment: Negative P/B ratio means company can be in trouble: that means companies can have problems.

# %%
fig, axes = plt.subplots(1, 11, figsize=(20, 5))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=data8[num_col[counter]], x=data8["K_clust"])
    counter = counter + 1

fig.tight_layout(pad=2.0)

# %% [markdown]
# Comments about clusters:
# 
# 1)  Cluster 0 has 8 companies that have the highest average price of 509 dollars, and the highest earnings per share 16 % and highest P/BRatio (30)  with modest price change of 6% and average volatility (1.5). Half of the companies belong to Health care sector.
# 
# 2) Cluster 1 has 20 companies, highest Cash Ratio (322) with relatively large Price change and Current price close to overall average.The average price 80 dollars  is close the overall average price 81 dollars. Sectors mostly form Health care and IT
# 
# 3) Cluster 2 has 264 companies and the average values of the features for this cluster are close to overall average values. Companies with average price and price change and volatility. The biggest value for this cluster is the number of companies.
# 
# 
# 4) Cluster 3 has 27 companies, mostly from Energy sector and this cluster has the highest volatility 2.8,large loss (per Price Changes) and relatiley low current price, with negative net cash flow, net income, Earnings per share. Because ofthe price change -16% - it is a group of the biggest drop of the price. 
# 
# 5) Cluster 4 has 9 companies. The highest Net Income and Estimated shares outstanding. It has lowets average volatility and low current price while price change of 5 % is not far from overall average price change 4%. Has the lowest average P/B Ratio and lowest Net Income.
# 
# 6) Cluster 5 has 7 companies, current price close to overall average, highest ROE.  It has price change lower than overal average price change. It has negative Net Cash Flow, Net Income, Earning per share and P/R Ratio. In this respect it is similar to another cluster - cluster 3. It seems the investors did not lose money but because of the negative indicators this cluster seem to be similar to biggest "loosers" cluster 3. 
# 
# 7) Cluster 6 has 2 companies the largest Net Cash Flow, with lower  price but Price change - 2-3 times larger than overall average.  
# 
# 8) Cluster 7 has 3 companies with the highest price change of 22% and P/E Ratio. High P/E ratio could mean the stock price is overvalued. It has relatively high volatility of 2 and average current price for this cluster 327 dollars - far larger than overall average price.  One should include the cluster in portfolio bt one has to be careful due to ROE and volatility.
# 
# 
# 

# %% [markdown]
# Overall: It seems grouping seems more useful than when we had 3 clusters. One can analyse the market and make decision for investing for example build portoflio.  
# 
# 
# In view of some linear dependence between the features and that some of the features have similar values for given sector it might be interesting to either 
# 1) remove the features that are sector dependent or 
# 2) add Sector-label (?).
# 
# I think also removing correlated variables or PCA might help to improve the K-means clustering for this case. 

# %%
scaledhc=subset_scaled_df.copy()

# %% [markdown]
# ## Hierarchical Clustering

# %%
# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]
high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(scaledhc, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(scaledhc))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm

# %%
# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)

# %% [markdown]
# **Let's explore different linkage methods with Euclidean distance only.**

# %%
# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(scaledhc, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(scaledhc))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm

# %%
# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)

# %% [markdown]
# Comment - still the highest cophenetic correlation coefficient is for average linkage.

# %% [markdown]
# **We see that the cophenetic correlation is maximum with Euclidean distance and average linkage.**
# 
# **Let's see the dendrograms for the different linkage methods.**

# %%
# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(scaledhc, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(scaledhc))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )

# %% [markdown]
# **Observations from dendrograms**
# 
# - The cophenetic correlation is highest for average linkage method (0.94).
# - We will move ahead with average linkage 
# - 8-9 appears to be the appropriate number of clusters from the dendrogram for average linkage.

# %%
datahc8=data.copy()

# %%
HCmodel8 = AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="average")
HCmodel8.fit(scaledhc)

# %%
scaledhc8=scaledhc.copy()

# %%
datahs=subset_scaled_df

# %%
datahc8["HC_Clust"] = HCmodel8.labels_
scaledhc8["HC_Clust"] = HCmodel8.labels_

# %% [markdown]
# Cluster profiling for 8 clusters

# %% [markdown]
# ## Cluster profiling k-8

# %%
cluster_profile1 = datahc8.groupby("HC_Clust").mean()

# %%
cluster_profile2 = scaledhc8.groupby("HC_Clust").mean()

# %%
cluster_profile1["No_in_clust"] = (
    datahc8.groupby("HC_Clust")["ROE"].count().values
)

# %%
fig, axes = plt.subplots(1, 11, figsize=(22, 7))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=datahc8[num_col[counter]], x=datahc8["HC_Clust"])
    counter = counter + 1

fig.tight_layout(pad=2.0)

# %% [markdown]
# Checking for the structure of the clusters

# %%
datahc8.groupby(['HC_Clust','GICS Sector']).Security.count()

# %% [markdown]
# We have one big cluster with 330 companies and the rest of the clusters has 1 company and one cluster has 2 companies and one 3. Hierarchical clustering seems for this data worse than the k-means clustering and I did not expect that. It is difficult to imagine what could go wrong here. 

# %%
# lets display cluster profile
cluster_profile1.style.highlight_max(color="lightgreen", axis=0)

# %% [markdown]
# Conclusion:  have started with 8 clusters as this ws the best case for K-means clustering. 
# 
# In the case of K-means clustering with 8 clusters - we had clusters that were more mixed.
# 
# Now we have one big cluster with averages for featuresclose to overall averages and one cluster with 3 companies one with 2 and 5 with one company. The single company clusters are the clusters with e.g. highest Current Price or highest volatility or highest Cash Ratio and the highest ROE
# 
# So we have the clustering method selecting only "highest price" (culster -company 4) and "highest volatility"+lowest price drop (cluster-company 6) or "largest price change" (cluster 0), "Largest ROE" (cluster) companies and put it as separate clusters. In this sense the method of hierarchical clustering worked like method detecting outliers. 
# 
# 
# This method did not seem to work well. It might be that we have too many features or the correlations between features play role and the clustering only finds outliers at this point. On the pair plots onecould see that for K-means the clusters were overlapping. This often means PCA can help to 
# 
# 
# What I think is worth a try before PCA is 
# - Internet suggests a method independent of outliers (K-medoids and DBSCAN).
# - And I will try Ward linkage as this seems have more clusters that are not a single company cluster. And is less dependent on outliers. 
# - feture selection removing correlated features - based on correlation matrix (it is like a step before PCA)
# - looking at seaborn pairplots from k-means clustering I believe one can pick the features which clustered there better(less overlap between clusters) and check k-means and agglomerative clustering again - for this subset of features (if we do not change features to PCA  new features then the overlapping of the clusters vvisible on pair plots will be a problem for most of the methods).  
# 
# 
# 

# %% [markdown]
# ## 9 cluster hierarcical clustering 

# %%
HCmodel9 = AgglomerativeClustering(n_clusters=9, affinity="euclidean", linkage="average")
HCmodel9.fit(scaledhc)

# %%
scaledhc9=scaledhc.copy()

# %%
datahc9=data.copy()

# %%
datahc9["HC_Clust"] = HCmodel9.labels_
scaledhc9["HC_Clust"] = HCmodel9.labels_

# %% [markdown]
# ## Cluster profiling k=9

# %%
cluster_profile19 = datahc9.groupby("HC_Clust").mean()

# %%
cluster_profile29 = scaledhc9.groupby("HC_Clust").mean()

# %%
cluster_profile19["No_in_clust"] = (
    datahc9.groupby("HC_Clust")["ROE"].count().values
)

# %% [markdown]
# This is not getting better for 9 clusters - we have one cluster with 330 companies 2 with 2 companies and 6 - single companies clusters.
# 
# I will explore the single company clusters n more detail below.

# %%
# lets display cluster profile
cluster_profile19.style.highlight_max(color="lightgreen", axis=0)

# %%
print(datahc9.loc[datahc9['HC_Clust'] > 0])

# %% [markdown]
# Alexion is subsidiary of Astra Zeneca, which among others recently was on the news as vaccine producent 

# %% [markdown]
# Amazon, Netlix, Bank of America, Facebook and Intel - are the companies in "their own clusters".Funny they are related to 
# ML/AI. 

# %% [markdown]
# Chesapeak is and Energy company that is losing value on stock market due to the lowe/falling oil prices.

# %% [markdown]
# Alliance Data Systems Corporation is a publicly traded provider of loyalty and marketing services, such as private label credit cards, coalition loyalty programs, and direct marketing, derived from the capture and analysis of transaction-rich data.

# %% [markdown]
# Alliance and Prceline belong to Internet & Direct Marketing Retail Sector. 

# %% [markdown]
# Conclusion - increasing the number of clusters does not change the overal structure - it seems rather the 3 companies cluster from previous (8 cluster) method got split. 
# It seems to be working even more like outlier detector. 
# 
# BUT the clustering works at the moment - it detect a special companies with exceptional data - companies that are either excellent to invest in or should be avoided.  

# %% [markdown]
# ## 8 clusters hierarchical clustering Ward linkage 

# %%
data8w=data.copy()

# %%
scaled8w=subset_scaled_df.copy()

# %%
HCmodel8W = AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="ward")
HCmodel8W.fit(scaled8w)

# %%
data8w["HC_Clust"] = HCmodel8W.labels_
scaled8w["HC_Clust"] = HCmodel8W.labels_

# %%
cluster_profile1 = data8w.groupby("HC_Clust").mean()
cluster_profile2 = scaled8w.groupby("HC_Clust").mean()

# %%
cluster_profile2["No_in_clust"] = (
    scaled8w.groupby("HC_Clust")["ROE"].count().values
)
cluster_profile1["No_in_clust"] = (
    data8w.groupby("HC_Clust")["ROE"].count().values
)


# %%
# lets display cluster profile
cluster_profile1.style.highlight_max(color="lightgreen", axis=0)

# %% [markdown]
# This method is a bit better.We have one big cluster of 285 companies and one with 22 companies and one with 9 one with 7, 2 clusters with 6 companies, one with 3 and one with 2.

# %%
import warnings
warnings.filterwarnings('ignore')
sns.pairplot(data8w,hue='HC_Clust',palette='hls')

# %%


# %% [markdown]
# ## Removing  Outliers

# %% [markdown]
# Comment 1: According to https://medium.com/analytics-vidhya/effect-of-outliers-on-k-means-algorithm-using-python-7ba85821ea23 outliers can affect k-means clustering (Euclidean distance) - outliers can group as separate cluster or they can cause other clusters to merge. One can consider method less sensitive to outliers such as k medians or use DBSCAN.
# 
# Comment 2: Agglomerative clustering can use various measures to calculate distance between two clusters, which is then used to decide which two clusters to merge.
# 
# Two popular approaches are single-link and complete-link. There seems to be some discrepancy in whether single-link or complete-link is sensitive to outliers. I am stating a few examples below but I am sure that there are many more.
# 
# (1) Stanford NLP IR book states "[complete-link] causes sensitivity to outliers".
# 
# (2) Chapter 8 in the Data Mining book by Tan and Kumar says the following in the section 8.3.2:
# 
# "The single link technique is good at handling non-elliptical shapes, but is sensitive to noise and outliers."
# 
# "Complete link is less susceptible to noise and outliers, but it can break large clusters and it favors globular shapes."
# 
# (3) The Applied Multivariate Statistics book says :
# 
# "... single link method which is also very sensitive to errors of measurement, but somewhat robust to outliers. The complete link and Wardâ€™s method tend to find compact clusters of nearly equal size with the clustering solution adversely affected by outliers."
# 

# %%
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

# %%
datan=data.copy()

# %%
df=remove_outlier(datan,"Current Price")

# %%
df1=remove_outlier(df,"Net Cash Flow")

# %%
df2=remove_outlier(df1,"ROE")

# %%
df3=remove_outlier(df2,"Net Income")

# %%
for item in num_col:
    histogram_boxplot(df3, item, kde=True, figsize=(8, 4))

# %%
df3.info()

# %% [markdown]
# I have removed some extreme outliers from Current Price ROE, Net Income and Net Cash Flow columns. I have now set of 201 entries. I will repeat hierarchical clustering first.

# %%
datanew=df3.copy()

# %% [markdown]
# ###  Note: instead of 350+ data we have now 200+ rows (and I did not reomve all outliers.).

# %%
plt.figure(figsize=(15,7))
sns.heatmap(datanew.corr(),annot=True,vmin=-1,vmax=1,cmap="Spectral")
plt.show()

# %% [markdown]
# It seems with outlier removed the correlaiton of features has changed a bit. That means we really have a diferent data set and a different problem to study. I will check how the methods performnow. 

# %%
datagroupsectornew = datanew.groupby(['GICS Sector'])

# %%
datagroupsectornew.mean()

# %%
# Scaling the data set before clustering
scaler = StandardScaler()
subsetwo = datanew[num_col].copy()
subset_scaledwo = scaler.fit_transform(subsetwo)

# %%
# Creating a dataframe from the scaled data
scaledwo_df = pd.DataFrame(subset_scaledwo, columns=subsetwo.columns)

# %%
scaledwo_df

# %%
clusters = range(1, 16)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(scaledwo_df)
    prediction = model.predict(scaledwo_df)
    distortion = (
        sum(
            np.min(cdist(scaledwo_df, model.cluster_centers_, "euclidean"), axis=1)
        )
        / scaledwo_df.shape[0]
    )

    meanDistortions.append(distortion)

    print("Number of Clusters:", k, "\tAverage Distortion:", distortion)

plt.plot(clusters, meanDistortions, "bx-")
plt.xlabel("k")
plt.ylabel("Average Distortion")
plt.title("Selecting k with the Elbow Method", fontsize=20)

# %% [markdown]
# Elbow method suggests 4,9and 13 clusters.

# %%
sil_score = []
cluster_list = list(range(2, 30))
for n_clusters in cluster_list:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict((scaledwo_df))
    # centers = clusterer.cluster_centers_
    score = silhouette_score(scaledwo_df, preds)
    sil_score.append(score)
    print("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))

plt.plot(cluster_list, sil_score)

# %% [markdown]
# Maybe 8 clusters ? The silhouettes coefficint seems to be very small. 

# %% [markdown]
# Clluster profiling:

# %%
kmeansn8 = KMeans(n_clusters=8, random_state=0)
kmeansn8.fit(scaledwo_df)

# %%
# finding optimal no. of clusters with silhouette coefficients
visualizer = SilhouetteVisualizer(KMeans(8, random_state=1))
visualizer.fit(scaledwo_df)
visualizer.show()

# %%
datanew8=datanew.copy()

# %%
scaledwo_new=scaledwo_df.copy()

# %%
# adding kmeans cluster labels to the original dataframe
datanew8["K_clust"] = kmeansn8.labels_

# %%
scaledwo_new["K_clust"] = kmeansn8.labels_

# %%
cluster_profile8n = datanew8.groupby("K_clust").mean()

# %%
cluster_profile8n["No"] = (
    datanew8.groupby("K_clust")["Current Price"].count().values
)

# %%
fig, axes = plt.subplots(1, 11, figsize=(24, 7))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=datanew8[num_col[counter]], x=datanew8["K_clust"])
    counter = counter + 1

fig.tight_layout(pad=2.0)

# %%
# lets display cluster profile
cluster_profile8n.style.highlight_max(color="lightgreen", axis=0)

# %% [markdown]
# ### Conclusion - without outliers we get different sized with K-means clustering. 
# 
# Cluster 0 has 63 companies with data similar to average.
# 
# Cluster 1 has 12 companies with highes average P/B Ratio Net Cash Flow,Cash Ratio and Price Change (11%)
# 
# cluster 2 has 33 companies with highest average ROE. and negative average Net Cash flow and average P.B Ratio.
# 
# Cluster 3 has 9 companies with highest average volatility,lowest price. It is the only cluster that has negative average Earnings per Share. 
# Since price is not high and the cluster has the biggest loss - due to -14% price change - this companies should be avoided.  
# 
# cluster 4 has 27 companies that have the highest average Net Income
# 
# Cluster 5 has 7 companies tand has the highest average price 102 dollars and highest earnigs per share.
# 
# Cluster 6 has 4 companies, and  highest average P/E ratio. Because of relatively high volatility and -10% price drop, smallest positive Earning per Share, 
# it is probably better to avoid these companies as they seem overvalued. 
# 
# Cluster 7 has 46 companies with 2nd lowest average stock price. All of the other indexes are low. Not the lowest 
# but this cluster seems to grop companies thatoverall donot perform well and it is reflected in low P/B Ratio, P/E Ratio, Earnings per Share.
# 
# ## Conclusion: 
# ## For investments the companies in clusters 7 6 and 3 should be avoided. 
# 
# ## One should pick companies from cluster 1,  cluster 2, cluster 5.
# 
# 
# 

# %%
datanew8

# %% [markdown]
# ## Companies from the clusters that one can recomend investing: 

# %% [markdown]
# ### Cluster 1

# %%
print(datanew8.loc[datanew8['K_clust'] == 1])

# %% [markdown]
# ### Cluster 2

# %%
print(datanew8.loc[datanew8['K_clust'] == 2])

# %% [markdown]
# ### Cluster 5 

# %%
print(datanew8.loc[datanew8['K_clust'] == 5])

# %% [markdown]
# I am curious how the hierarchical clustering would work for these data set with some outlier removed

# %% [markdown]
# ### Hierarchical Clustering with some outliers removed

# %%
# list of distance metrics
distance_metrics = ["euclidean", "chebyshev", "mahalanobis", "cityblock"]

# list of linkage methods
linkage_methods = ["single", "complete", "average", "weighted"]
high_cophenet_corr = 0
high_dm_lm = [0, 0]

for dm in distance_metrics:
    for lm in linkage_methods:
        Z = linkage(scaledwo_df, metric=dm, method=lm)
        c, coph_dists = cophenet(Z, pdist(scaledwo_df))
        print(
            "Cophenetic correlation for {} distance and {} linkage is {}.".format(
                dm.capitalize(), lm, c
            )
        )
        if high_cophenet_corr < c:
            high_cophenet_corr = c
            high_dm_lm[0] = dm
            high_dm_lm[1] = lm

# %%

 # printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} distance and {} linkage.".format(
        high_cophenet_corr, high_dm_lm[0].capitalize(), high_dm_lm[1]
    )
)


# %%
# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

high_cophenet_corr = 0
high_dm_lm = [0, 0]

for lm in linkage_methods:
    Z = linkage(scaledwo_df, metric="euclidean", method=lm)
    c, coph_dists = cophenet(Z, pdist(scaledwo_df))
    print("Cophenetic correlation for {} linkage is {}.".format(lm, c))
    if high_cophenet_corr < c:
        high_cophenet_corr = c
        high_dm_lm[0] = "euclidean"
        high_dm_lm[1] = lm


# %%
# printing the combination of distance metric and linkage method with the highest cophenetic correlation
print(
    "Highest cophenetic correlation is {}, which is obtained with {} linkage.".format(
        high_cophenet_corr, high_dm_lm[1]
    )
)

# %% [markdown]
# ## Conclusion:Centroid linkage and Enclidean metric give the highest cophenetic correlation. But I think the ward linkage clusters look better (not like "outlier detection"). 

# %% [markdown]
# ## Dendrograms:

# %%
# list of linkage methods
linkage_methods = ["single", "complete", "average", "centroid", "ward", "weighted"]

# lists to save results of cophenetic correlation calculation
compare_cols = ["Linkage", "Cophenetic Coefficient"]

# to create a subplot image
fig, axs = plt.subplots(len(linkage_methods), 1, figsize=(15, 30))

# We will enumerate through the list of linkage methods above
# For each linkage method, we will plot the dendrogram and calculate the cophenetic correlation
for i, method in enumerate(linkage_methods):
    Z = linkage(scaledwo_df, metric="euclidean", method=method)

    dendrogram(Z, ax=axs[i])
    axs[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)")

    coph_corr, coph_dist = cophenet(Z, pdist(scaledwo_df))
    axs[i].annotate(
        f"Cophenetic\nCorrelation\n{coph_corr:0.2f}",
        (0.80, 0.80),
        xycoords="axes fraction",
    )

# %%
HCmodel8wo = AgglomerativeClustering(n_clusters=8, affinity="euclidean", linkage="ward")
HCmodel8wo.fit(scaledwo_df)

# %%
scaledhcwo=scaledwo_df.copy()

# %%
datawo8=subsetwo.copy()

# %%
datawo8["HC_Clusters"] = HCmodel8wo.labels_
scaledhcwo["HC_Clusters"] = HCmodel8wo.labels_

# %%
cluster_profile1wohc = datawo8.groupby("HC_Clusters").mean()

# %%
cluster_profile2wohc = scaledhcwo.groupby("HC_Clusters").mean()

# %%
cluster_profile1wohc["No_in_clust"] = (
    datawo8.groupby("HC_Clusters")["ROE"].count().values
)


# %%
fig, axes = plt.subplots(1, 11, figsize=(22, 7))
fig.suptitle("Boxplot of numerical variables for each cluster")
counter = 0
for ii in range(11):
    sns.boxplot(ax=axes[ii], y=datawo8[num_col[counter]], x=datawo8["HC_Clusters"])
    counter = counter + 1

fig.tight_layout(pad=2.0)


# %%

# lets display cluster profile
cluster_profile1wohc.style.highlight_max(color="lightgreen", axis=0)

# %% [markdown]
# ## Conclusion: I do not see that agglomerative clustering performed better or worse than K-means clustering. Agglomerative clustering has found a cluster wiht 1 company so again it worked morelike outlier detector. 

# %% [markdown]
# ## (Note: I have tried DBSCAN and the highest number of clusters I was getting with positive silhouette score  was 3 (for DBSCAN(eps=3, min_samples=2)) - so the DBSCAN was not working on our data, even on the data with partially removed outliers. I did not know how to continue with this so I do not include it here. )

# %% [markdown]
# ## PCA 

# %%
scaledwo_new["K_clust"] = kmeansn8.labels_

# %%
# importing library
from sklearn.decomposition import PCA

# setting the number of components to 2
pca = PCA(n_components=9)

# transforming data and storing results in a dataframe
X_reduced_pca = pca.fit_transform(scaledwo_new)
reduced_df_pca = pd.DataFrame(
    data=X_reduced_pca, columns=["Component 1", "Component 2","Component 3", "Component 4", "Component 5", "Component 6", "Component 7", "Component 8","Component 9"]
)

# %%
# checking the amount of variance explained
pca.explained_variance_ratio_.sum()

# %% [markdown]
# reduced_df_pca["K_clust"]=scaledwo_new["K_clust"]

# %% [markdown]
# ### 9 componets PCA  explain 95% of infomration in the dataset. 

# %% [markdown]
# ### Conclusion: These are a bit too many components to include. I do not think it helps for this project to replace 11 features with 9 PCA components. 

# %% [markdown]
# ### 

# %% [markdown]
# ## K-means vs Hierarchical Clustering

# %% [markdown]
# K- means seemed to perform bad but when I did hierarchical clustering the method worked like outlier "detector" by acicng most of the companies in one clster and the outliers each in separate cluster. 
# The best cophenetic distance ws for Euclidean and average linkage method. But this method sufered from the "detecting outliers" 
# behavior.
# 
# Dendrograms suggest that ward linkage hierarchical clustering can produce bigger clusters. So I have tried and indeed the clusters structure changed - and we have still one big cluster but the other clusters contain more than company. 

# %% [markdown]
# At the moment the data set serves more for detection of outliers and can be used to single out companies that have exceptionaly good and bed 
# stock market results.
# 
# 

# %% [markdown]
# After removal of outliers the K-mean clustering produced clusters that have more decent sizes and one can connect cluster with 
# distinct stock performance and use this for recomendation - both fromwhich clusters toselect the companies and from which clusters to avoid the companies at the moment.

# %% [markdown]
# ## Actionable Insights and Recommendations
# 

# %% [markdown]
# Hierarchical clustering can be used to quickly group companies and detect outliers for given feature  this can help to find best and worst performins company/companies. 

# %% [markdown]
# It seems that both methods did not work. I have tried to increase for K-means clustering the nmber of clusters to 20 and the 
# it did not seem to work better with larger number of clusters.
# 
# One can see with dendrograms that for anything except ward linkage - the clusters that structure of dendrograms suggests that the "outlier detection" behavior would continue also for larger number of clusters. 

# %% [markdown]
# When designing investment portfolio it is useful to diversify. It means to invest in several companies stocks.
# 
# One cannot find a clear rules/suggestions for investing based on the companies performance and based on the 
# indexes such as Ratios provided in the dataset.  It seems it is typical to present investing like a kind of 
# "trade magic".
# 
# Machine Learning methods - for example clustering offers comparisons of the data for many variables. 
# One an both study correlations and clustering of the stock market as a whoe or within industry sector. 
# 
# Machine Learning can help discover trends/correlations. This tpic has been studied in the past for many years and now it can be studied 
# in real time as the new data is generatedevery day/month etc. 
# 
# Clustering offers grouping of the companies. When such grouping is done by hand one cannot easily decide what are 
# the best ways to group companies. 
# 

# %% [markdown]
# I have indentified 3 clusters of companies to avoid and 3 clusters of companies one can use to build portfolio.
# 
# One recomendation overall seem sot be to avoid the energy Sector. 
# 
# Instead invest in IT, Telecom Service, and Machine Learning - based businesses (please see the section where songle company sized clusters were discussed - I included the names of the companies).

# %% [markdown]
# ## Other comment:
# I think it would be interesting to study the indexes and other parameters provided by the companies ...if one cannot define 
# better perfomance indicators or relative risk indicators (long and short term). 
# 
# We were provided data that do not reflect dynamics of the stock market. And that is a second topic - how the parameters/indices 
# related to companies performance can be mapped onto their stocks. 
# 
# It has been known that disasters, major events, news-twitter-"rumours" also can afect the stock market. In the project here in the first part with outliers - the clustering served as outler detection. It could maybe serve as 
# a way to detect stock market manipulations.  
# 

# %%


