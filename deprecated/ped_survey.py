"""
An analysis based on pedestrian counts was abandoned due to insufficient
counting stations and limited focus on inner urban areas.
"""

# %%
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

# Set the aesthetics for the plots
sns.set_theme(style="dark")
sns.set_context("paper")

# %%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("../temp/dataset_subset.gpkg")
mad_ped_gpd = gpd.read_file("../temp/ped_counts.gpkg")
mad_ped_gpd.set_index('date', inplace=True)

#%%
def report_nan(df):
    # some values have NaN - e.g. near distance hillier and some landuses
    nan_count_per_column = df.isna().sum()
    nan_columns = nan_count_per_column[nan_count_per_column > 0]
    print(nan_columns)
    rows_with_nan = df.isna().any(axis=1).sum()
    print(f'total rows with NaN: {rows_with_nan}')

# %%
# coerce to zeros
report_nan(mad_gpd)
mad_gpd.fillna(0, inplace=True)

#%%
cent_cols = [
    'cc_hillier_200',
    'cc_hillier_200_ang',
    'cc_hillier_500',
    'cc_hillier_500_ang',
    'cc_hillier_1000',
    'cc_hillier_1000_ang',
    'cc_hillier_2000',
    'cc_hillier_2000_ang',
    'cc_hillier_5000',
    'cc_hillier_5000_ang',
    'cc_hillier_10000',
    'cc_hillier_10000_ang',
    'cc_betweenness_200',
    'cc_betweenness_200_ang',
    'cc_betweenness_500',
    'cc_betweenness_500_ang',
    'cc_betweenness_1000',
    'cc_betweenness_1000_ang',
    'cc_betweenness_2000',
    'cc_betweenness_2000_ang',
    'cc_betweenness_5000',
    'cc_betweenness_5000_ang',
    'cc_betweenness_10000',
    'cc_betweenness_10000_ang',
]

#%%
# Filter the data for PCA
data_for_pca = mad_gpd[cent_cols]
# Standardize the data
scaler = preprocessing.StandardScaler()
data_scaled = scaler.fit_transform(data_for_pca)
# PCA
pca = PCA(n_components=4)
principal_components = pca.fit_transform(data_scaled)
# save
for i in range(4):
    mad_gpd[f'pca_{i + 1}'] = principal_components[:,i]
print("Explained Variance Ratio:", pca.explained_variance_ratio_)


# %%
mad_ped_gpd_daily = mad_ped_gpd[['identifier', 'pedestrians']].groupby('identifier').resample('D').sum()
mad_ped_gpd_daily.drop(columns=['identifier'], inplace=True)
mad_ped_gpd_daily

# %%
mad_ped_gpd_daily_reset = mad_ped_gpd_daily.reset_index()
mad_ped_gpd_daily_reset

#%%
lightweight_gdf = mad_ped_gpd.drop_duplicates(subset='identifier')[['identifier', 'network_key']]
# double check that all pedestrian gates are assigned to a valid node index
check_indices = lightweight_gdf['network_key'].isin(mad_gpd['index'])
if not check_indices.all():
    print('Some gate locations are not matched to a valid network key')
    print(lightweight_gdf[~lightweight_gdf['network_key'].isin(mad_gpd['index'])])
    raise ValueError
lightweight_gdf

#%%
merged_gpd_a = pd.merge(mad_ped_gpd_daily_reset, lightweight_gdf, left_on='identifier', right_on='identifier', how='left')
merged_gpd_a

#%%
merged_gpd = pd.merge(merged_gpd_a, mad_gpd, left_on='network_key', right_on='index', how='left')
merged_gpd.drop(columns=['index'], inplace=True)
merged_gpd = gpd.GeoDataFrame(merged_gpd, geometry='geometry')
report_nan(merged_gpd)
merged_gpd.to_file('../temp/merged.gpkg')
merged_gpd

#%%
# Plotting the pedestrian counts
sns.lineplot(x='date', y='pedestrians', data=merged_gpd, color='blue', marker='o')
plt.title('Daily Pedestrian Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Pedestrian Count')
plt.show()

#%%
sns.lineplot(x='cc_retail_100_wt', y='pedestrians', data=merged_gpd, color='blue', marker='o')
plt.xlabel('cc_retail_100_wt')
plt.ylabel('pedestrians')
plt.show()

#%%
for i in range(4):
    sns.lineplot(x=f'pca_{i + 1}', y='pedestrians', data=merged_gpd)
    plt.xlabel(f'PCA {i + 1}')
    plt.ylabel('Pedestrian Count')
    plt.show()

#%%
for col in cent_cols:
    color = 'red' if 'ang' in col else 'blue'
    sns.lineplot(x=col, y='pedestrians', data=merged_gpd, color=color)
    plt.title('Pedestrian Count vs Network Centrality')
    plt.xlabel(col)
    plt.ylabel('Pedestrian Count')
    plt.show()

#%%
# Histogram of pedestrian counts
sns.histplot(merged_gpd['pedestrians'], bins=30, kde=True)
plt.title('Distribution of Pedestrian Counts')
plt.xlabel('Pedestrian Count')
plt.show()

#%%
for col in cent_cols:
    color = 'red' if 'ang' in col else 'blue'
    sns.jointplot(data=merged_gpd, x=col, y='pedestrians', kind='hex')
    plt.show()