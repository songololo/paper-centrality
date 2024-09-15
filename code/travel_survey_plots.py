# %%
import pathlib
from importlib import reload

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import util

reload(util)

sns.set_theme(style="dark")
sns.set_context("paper")

# %%
images_path = pathlib.Path("../plots")

# %%
survey_data = gpd.read_file("../data/edm_2018_viajes.csv")
survey_zones = gpd.read_file("../data/zones_ZT1259.gpkg")
survey_zones = survey_zones.to_crs(25830)


# %%
len(survey_zones)
# 1259

# %%
survey_zones.geometry.area.values.round(3).mean() / 1000**2
# 6.37

# %%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("../temp/dataset.gpkg")

# %%
# column names
survey_gpd = survey_data.rename(
    columns={
        "ID_HOGAR": "household_id",
        "ID_IND": "individual_id",
        "ID_VIAJE": "trip_id",
        "VORI": "origin_reason",
        "VORIHORAINI": "start_time",
        "VDES": "dest_reason",
        "VDESHORAFIN": "end_time",
        "VFRECUENCIA": "frequency",
        "VVEHICULO": "private_vehicle",
        "VNOPRIVADO": "private_reason",
        "VNOPUBLICO": "public_reason",
        "VORIZT1259": "origin_zone",
        "VDESZT1259": "dest_zone",
        "TIPO_ENCUESTA": "survey_type",
        "N_ETAPAS_POR_VIAJE": "stages_per_trip",
        "MOTIVO_PRIORITARIO": "main_reason",
        "DISTANCIA_VIAJE": "trip_distance",
        "MODO_PRIORITARIO": "main_mode",
        "ELE_G_POND_ESC2": "weight_factor",
    }
)
# empty
survey_gpd = survey_gpd.replace("", -1)
# types
survey_gpd = survey_gpd.astype(
    {
        "household_id": int,
        "individual_id": int,
        "trip_id": int,
        "origin_reason": int,  # Using category if there are repeated text values
        "start_time": int,
        "dest_reason": int,  # Using category for text with repeated values
        "end_time": int,
        "frequency": int,
        "private_vehicle": int,
        "private_reason": int,
        "public_reason": int,
        "origin_zone": str,
        "dest_zone": str,
        "survey_type": str,
        "stages_per_trip": int,
        "main_reason": int,
        "trip_distance": float,  # Assuming distance could be a float for decimal values
        "main_mode": int,
        "weight_factor": float,  # Assuming weight could be a float
    }
)


# %%
filtered_gpd = survey_gpd[survey_gpd["main_reason"].isin([2, 3, 4, 5, 6])]
# trip distance and strictly non private don't make a big difference
# trip counts
origin_counts = filtered_gpd.groupby("origin_zone").size()
origin_counts.name = "origin_count"
dest_counts = filtered_gpd.groupby("dest_zone").size()
dest_counts.name = "dest_count"
# merge
counts = survey_zones.merge(origin_counts, left_on="ZT1259", right_index=True, how="left")
counts = counts.merge(dest_counts, left_on="ZT1259", right_index=True, how="left")
# CRS
counts = counts.dropna()

# %%
ax = counts.plot("origin_count")
ax.set_axis_off()
ax.set_xlim(430000, 450000)
ax.set_ylim(4465000, 4485000)

ax = counts.plot("dest_count")
ax.set_axis_off()
ax.set_xlim(430000, 450000)
ax.set_ylim(4465000, 4485000)

# %%
reload(util)
mad_gpd = util.rename_cent_cols(mad_gpd)

# %%
distances_cent = [500, 1000, 2000, 5000, 10000]
mad_gpd = util.generate_close_n_cols(mad_gpd, distances_cent, length_weighted=True)
mad_gpd = util.generate_close_n_cols(mad_gpd, distances_cent, length_weighted=False)

# %%
counts["origin_by_area"] = counts["origin_count"] / counts.geometry.area
counts["dest_by_area"] = counts["dest_count"] / counts.geometry.area

# %%
# cent
cent_cols = util.generate_cent_columns(
    [
        "density_{d}",
        "far_{d}",
        "far_{d}_ang",
        "far_norm_{d}",
        "far_norm_{d}_ang",
        "gravity_{d}",
        "harmonic_{d}",
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
        "betw_{d}",
        "betw_wt_{d}",
        "betw_{d}_ang",
        "NACH_{d}",
        "NACH_{d}_ang",
    ],
    distances_cent,
)
lw_cent_cols = util.generate_cent_columns(
    [
        "lw_density_{d}",
        "density_{d}_seg",
        "lw_far_{d}",
        "lw_far_{d}_ang",
        "lw_far_norm_{d}",
        "lw_far_norm_{d}_ang",
        "lw_gravity_{d}",
        "gravity_{d}_seg",
        "lw_harmonic_{d}",
        "harmonic_{d}_seg",
        "lw_harmonic_{d}_ang",
        "lw_closeness_{d}",
        "lw_closeness_{d}_ang",
        "lw_close_N1_{d}",
        "lw_close_N1_{d}_ang",
        "lw_close_N1.2_{d}",
        "lw_close_N1.2_{d}_ang",
        "lw_close_N2_{d}",
        "lw_close_N2_{d}_ang",
        "lw_betw_{d}",
        "lw_betw_wt_{d}",
        "betw_{d}_seg",
        "lw_betw_{d}_ang",
        "lw_NACH_{d}",
        "lw_NACH_{d}_ang",
    ],
    distances_cent,
)

# %%
cent_labels = [
    "density",
    "farness",
    "farness ang.",
    "farness norm.",
    "farness norm. ang.",
    "gravity",
    "harmonic",
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
    "betweenness",
    "betweenness wt.",
    "betweenness ang.",
    "NACH",
    "NACH ang.",
]
lw_cent_labels = [
    "density",
    "density cont.",
    "farness",
    "farness ang.",
    "farness norm.",
    "farness norm. ang.",
    "gravity",
    "gravity cont.",
    "harmonic",
    "harmonic cont.",
    "harmonic ang.",
    "closeness",
    "closeness ang.",
    r"closen. $N^{1}$",
    r"closen. $N^{1}$ ang.",
    r"closen. $N^{1.2}$",
    r"closen. $N^{1.2}$ ang.",
    r"closen. $N^{2}$",
    r"closen. $N^{2}$ ang.",
    "betweenness",
    "betweenness wt.",
    "betweenness cont.",
    "betweenness ang.",
    "NACH",
    "NACH ang.",
]

overlap = gpd.sjoin(mad_gpd, counts, how="left", op="intersects")
merged_gpd = counts.copy(deep=True)
for col in cent_cols + lw_cent_cols:
    val = overlap.groupby("ZT1259")[col].mean()
    merged_gpd = merged_gpd.merge(val, left_on="ZT1259", right_index=True, how="left")
# drop periphery areas not intersecting streets data
merged_gpd = merged_gpd.dropna()

# %%
ax = merged_gpd.plot("harmonic_1000")
ax.set_axis_off()
ax.set_xlim(430000, 450000)
ax.set_ylim(4465000, 4485000)

# %%
sns.jointplot(data=merged_gpd, x="dest_count", y="harmonic_1000")

# %%
for cols, corr_labels, suptitle, cent_lu_corr_path, c_cols, c_labels in [
    [
        ("pca_1", "cc_hill_q0_200_wt"),
        ("Trips/Area - Dest", "Trips/Area - Origin"),
        "Spearman Rank - average unweighted centralities - number of trips",
        "cent_ts_corrs.pdf",
        cent_cols,
        cent_labels,
    ],
    [
        ("pca_1", "cc_hill_q0_200_wt"),
        ("Trips/Area - Dest", "Trips/Area - Origin"),
        "Spearman Rank - average weighted centralities - number of trips",
        "cent_ts_corrs_lw.pdf",
        lw_cent_cols,
        lw_cent_labels,
    ],
]:
    merged_gpd_cent_filter = merged_gpd[c_cols]
    cols = [
        "dest_by_area",
        "origin_by_area",
    ]
    # create heatmaps for original variables plotted against correlations
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 10), sharey=True, dpi=200, constrained_layout=True)
    fig.suptitle(suptitle)
    for n in range(2):
        col = cols[n]
        corr = merged_gpd_cent_filter.corrwith(
            merged_gpd[col], method="spearman", numeric_only=True
        )
        corr_label = corr_labels[n]
        heatmap_ax = axes[n]
        heatmap_ax.set_title(corr_label)
        heatmap_corr = corr.values.reshape(len(c_labels), len(distances_cent))
        # set this before the plot otherwise ticks are off centre
        heatmap_ax.set_xticks(np.arange(len(distances_cent)))
        heatmap_ax.set_yticks(np.arange(len(c_labels)))
        sns.heatmap(
            heatmap_corr,
            ax=heatmap_ax,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            yticklabels=c_labels,
            xticklabels=distances_cent,
            cbar=False,
            square=True,
        )
        heatmap_ax.set_xticklabels(distances_cent, rotation=90, ha="center")
        heatmap_ax.set_yticklabels(c_labels, va="center")
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
    fig.savefig(images_path / cent_lu_corr_path)
