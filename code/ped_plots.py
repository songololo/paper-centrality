# %%
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

sns.set_theme(style="dark")
sns.set_context("paper")

# %%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("../temp/dataset_subset.gpkg")
mad_ped_gpd = gpd.read_file("../temp/ped_counts.gpkg")
mad_ped_gpd.set_index('date', inplace=True)

# %%
mad_ped_gpd_daily = mad_ped_gpd[['identifier', 'pedestrians']].groupby('identifier').resample('D').sum()
mad_ped_gpd_daily.drop(columns=['identifier'], inplace=True)
mad_ped_gpd_daily

# %%
mad_ped_gpd_daily_reset = mad_ped_gpd_daily.reset_index()
mad_ped_gpd_daily_reset

#%%
lightweight_gdf = mad_ped_gpd.drop_duplicates(subset='identifier')[['identifier', 'network_key']]
lightweight_gdf

#%%
merged_gpd_a = pd.merge(mad_ped_gpd_daily_reset, lightweight_gdf, left_on='identifier', right_on='identifier', how='left')
merged_gpd_a

#%%
merged_gpd = pd.merge(merged_gpd_a, mad_gpd, left_on='network_key', right_on='index', how='left')
merged_gpd.drop(columns=['index'], inplace=True)
merged_gpd
