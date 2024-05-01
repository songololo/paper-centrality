# %%
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA

import util
from importlib import reload

reload(util)

sns.set_theme(style="dark")
sns.set_context("paper")

# %%
images_path = pathlib.Path("../plots")

# %%
survey_data = gpd.read_file("../data/edm_2018_viajes.csv")
survey_zones = gpd.read_file("../data/zones_ZT1259.gpkg")
survey_zones = survey_zones.to_crs(25830)

#%%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("../temp/dataset.gpkg")

#%%
# column names
survey_gpd = survey_data.rename(columns={
    'ID_HOGAR':'household_id',
    'ID_IND':'individual_id',
    'ID_VIAJE':'trip_id',
    'VORI':'origin_reason',
    'VORIHORAINI':'start_time',
    'VDES':'dest_reason',
    'VDESHORAFIN':'end_time',
    'VFRECUENCIA':'frequency',
    'VVEHICULO':'private_vehicle',
    'VNOPRIVADO':'private_reason',
    'VNOPUBLICO':'public_reason',
    'VORIZT1259':'origin_zone',
    'VDESZT1259':'dest_zone',
    'TIPO_ENCUESTA':'survey_type',
    'N_ETAPAS_POR_VIAJE':'stages_per_trip',
    'MOTIVO_PRIORITARIO':'main_reason',
    'DISTANCIA_VIAJE':'trip_distance',
    'MODO_PRIORITARIO':'main_mode',
    'ELE_G_POND_ESC2':'weight_factor',
})
# empty
survey_gpd = survey_gpd.replace('', -1)
# types
survey_gpd = survey_gpd.astype({
    'household_id': int,
    'individual_id': int,
    'trip_id': int,
    'origin_reason': int,  # Using category if there are repeated text values
    'start_time': int,
    'dest_reason': int,  # Using category for text with repeated values
    'end_time': int,
    'frequency': int,
    'private_vehicle': int,
    'private_reason': int,
    'public_reason': int,
    'origin_zone': str,
    'dest_zone': str,
    'survey_type': str,
    'stages_per_trip': int,
    'main_reason': int,
    'trip_distance': float,  # Assuming distance could be a float for decimal values
    'main_mode': int,
    'weight_factor': float  # Assuming weight could be a float
})


#%% Shopping
filtered_gpd = survey_gpd[survey_gpd['main_reason'].isin([2, 3, 4, 5, 6])]
# filtered_gpd = filtered_gpd[filtered_gpd['private_vehicle'] == 2]
# trip counts
origin_counts = filtered_gpd.groupby('origin_zone').size()
origin_counts.name = 'origin_count'
dest_counts = filtered_gpd.groupby('dest_zone').size()
dest_counts.name = 'dest_count'
# merge
walk_counts = survey_zones.merge(origin_counts, left_on='ZT1259', right_index=True, how='left')
walk_counts = walk_counts.merge(dest_counts, left_on='ZT1259', right_index=True, how='left')
# CRS
walk_counts = walk_counts.dropna()

#%%
ax = walk_counts.plot('origin_count')
ax.set_xlim(430000, 450000)
ax.set_ylim(4465000, 4485000)
ax = walk_counts.plot('dest_count')
ax.set_xlim(430000, 450000)
ax.set_ylim(4465000, 4485000)

#%%
mad_gpd = util.rename_cent_cols(mad_gpd)
distances_cent = [500, 1000, 2000, 5000, 10000]
mad_gpd = util.generate_close_n_cols(mad_gpd, distances_cent)

#%%
walk_counts['origin_by_area'] = walk_counts['origin_count'] /  walk_counts.geometry.area
walk_counts['dest_by_area'] = walk_counts['dest_count'] / walk_counts.geometry.area

#%%
# cent
cent_cols = util.generate_cent_columns(
    [
        "density_{d}",
        "density_{d}_seg",
        "far_{d}",
        "far_{d}_ang",
        "gravity_{d}",
        "gravity_{d}_seg",
        "harmonic_{d}",
        "harmonic_{d}_seg",
        "harmonic_{d}_ang",
        "closeness_{d}",
        "closeness_{d}_ang",
        "close_N1_{d}",
        "close_N1_{d}_ang",
        "close_N1.2_{d}",
        "close_N1.2_{d}_ang",
        "close_N2_{d}",
        "close_N2_{d}_ang",
        "cycles_{d}",
    ], distances_cent
)
cent_labels = [
    "density",
    "density seg.",
    "farness",
    "farness ang.",
    "gravity",
    "gravity seg.",
    "harmonic",
    "harmonic seg.",
    "harmonic ang.",
    "closeness",
    "closeness ang.",
    r"closen. $N^{1}$",
    r"closen. $N^{1}$ ang.",
    r"closen. $N^{1.2}$",
    r"closen. $N^{1.2}$ ang.",
    r"closen. $N^{2}$",
    r"closen. $N^{2}$ ang.",
    "cycles",
]

overlap = gpd.sjoin(mad_gpd, walk_counts, how="left", op='intersects')
merged_gpd = walk_counts.copy(deep=True)
# drop periphery areas not intersecting streets data
for col in cent_cols:
    val = overlap.groupby('ZT1259')[col].mean()
    merged_gpd = merged_gpd.merge(val, left_on='ZT1259', right_index=True, how='left')
    merged_gpd = merged_gpd.dropna()

#%%
ax = merged_gpd.plot('harmonic_1000')
ax.set_xlim(430000, 450000)
ax.set_ylim(4465000, 4485000)

#%%
sns.jointplot(data=merged_gpd, x='dest_count', y='harmonic_1000')

#%%
merged_gpd_cent_filter = merged_gpd[cent_cols]
corr_labels = [
    "Trips - Dest.",
    "Trips - Origin",
    "Trips/Area - Dest",
    "Trips/Area - Origin",
]
corrs = {}
for col in [
    "dest_count",
    "origin_count",
    "dest_by_area",
    "origin_by_area",
]:
    corrs[col] = merged_gpd_cent_filter.corrwith(
        merged_gpd[col], method="spearman", numeric_only=True
    )
# create heatmaps for original variables plotted against correlations
fig, axes = plt.subplots(
    1, 4, figsize=(10, 8), sharey=True, dpi=200, constrained_layout=True
)
suptitle = " Spearman Rank correlations"
fig.suptitle(suptitle)
for n, (corr, corr_label) in enumerate(zip(list(corrs.values()), corr_labels)):
    heatmap_ax = axes[n]
    heatmap_ax.set_title(corr_label)
    heatmap_corr = corr.values.reshape(len(cent_labels), len(distances_cent))
    # set this before the plot otherwise ticks are off centre
    heatmap_ax.set_xticks(np.arange(len(distances_cent)))
    heatmap_ax.set_yticks(np.arange(len(cent_labels)))
    sns.heatmap(
        heatmap_corr,
        ax=heatmap_ax,
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        yticklabels=cent_labels,
        xticklabels=distances_cent,
        cbar=False,
        square=True,
    )
    heatmap_ax.set_xticklabels(distances_cent, rotation=90, ha="center")
    heatmap_ax.set_yticklabels(cent_labels, va="center")
    # Add the correlation values on top
    for i, row in enumerate(heatmap_corr):
        for j, value in enumerate(row):
            heatmap_ax.text(
                j + 0.5,
                i + 0.5,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black" if abs(value) < 0.5 else "white",
                size=8,
            )
fig.savefig(images_path / 'cent_ts_corrs.png')