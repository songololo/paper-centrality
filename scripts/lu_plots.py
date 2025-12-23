# %%
import importlib
import pathlib

import contextily as cx
import esda
import geopandas as gpd
import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib_map_utils import NorthArrow
from matplotlib_scalebar import scalebar
from scipy import stats
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from scripts import common

importlib.reload(common)

sns.set_theme(style="dark")
sns.set_context("paper")

# %%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("./temp/dataset.gpkg")

# %%
images_path = pathlib.Path("./paper/plots")
tables_path = pathlib.Path("./paper/tables")
# rename columns for clarity
mad_gpd = common.rename_cent_cols(mad_gpd)

# %%
# generate
for is_angular, lu_cols in zip(
    [False, True], [common.LU_COLS_SHORTEST, common.LU_COLS_SIMPLEST], strict=False
):
    # create a copy of the dataframe with non variable cols removed
    mad_gpd_lu_filter = mad_gpd[lu_cols]
    ## Correlation Matrix
    corr = mad_gpd_lu_filter.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9), constrained_layout=True)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    ax.set_title("Correlation matrix" if is_angular is False else "Correlation matrix - angular")
    lu_corr_path = "lu_corr_matrix"
    if is_angular is True:
        lu_corr_path += "_ang"
    fig.savefig(images_path / f"{lu_corr_path}.pdf")

    ## PCA
    # extract 90% of explained variance
    n_components = 0.9
    # scale data
    stand_scaler = preprocessing.PowerTransformer()
    X_transf = stand_scaler.fit_transform(mad_gpd_lu_filter)
    model = PCA(n_components=4)
    X_latent = model.fit_transform(X_transf)
    exp_var = model.explained_variance_
    # eigenvector by eigenvalue - i.e. correlation to original
    loadings = model.components_.T * np.sqrt(exp_var)
    loadings = loadings.T  # transform for slicing
    for i in range(X_latent.shape[1]):
        vals = np.clip(X_latent[:, i], -10, 10)
        if is_angular is False:
            mad_gpd[f"pca_{i + 1}"] = vals
        else:
            mad_gpd[f"pca_{i + 1}_ang"] = vals
    print("explained variance ratio in %", model.explained_variance_ratio_ * 100)
    # plot PCA loadings
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), sharey=True, dpi=200, constrained_layout=True)
    lu_distances = [100, 200, 500, 1000, 2000]
    column_labels = [
        "food_bev",
        "retail",
        "services",
        "creat_entert",
        "accommod",
    ]
    # create heatmaps for original vectors plotted against PCA components
    for n in range(X_latent.shape[1]):
        heatmap_ax = axes[n]
        heatmap_ax.set_title(f"PCA {n + 1}" if is_angular is False else f"PCA {n + 1} - angular")
        heatmap_ax.set_yticks(np.arange(len(column_labels)))
        heatmap_ax.set_xticks(np.arange(len(lu_distances)))
        heatmap_corr = loadings[n].reshape(len(column_labels), len(lu_distances))
        sns.heatmap(
            heatmap_corr,
            ax=heatmap_ax,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            yticklabels=column_labels,
            xticklabels=lu_distances,
            cbar=False,
        )
        heatmap_ax.set_xticklabels(lu_distances, rotation=90, ha="right")
        heatmap_ax.set_xlabel(
            r"explained $\sigma^{2}$" + f" {model.explained_variance_ratio_[n]:.1%}"
        )
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
    pca_corr_path = "pca_corr_matrix"
    if is_angular is True:
        pca_corr_path += "_ang"
    fig.savefig(images_path / f"{pca_corr_path}.pdf")

# %%
lu_cols = ["pca_1", "cc_hill_q0_200_wt", "food_bev_200", "retail_200"]
lu_labels = [
    "PCA 1",
    "Landuse Richness 70m avg. walk dist. (200m max.)",
    "Food & Beverage 70m avg. walk dist. (200m max.)",
    "Retail 70m avg. walk dist. (200m max.)",
]
for col, label in zip(lu_cols, lu_labels, strict=False):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, constrained_layout=True)
    mad_gpd.plot(
        ax=ax,
        column=col,
        cmap="turbo",
        linewidths=1.5,
        vmin=0,
        vmax=np.percentile(mad_gpd[col], 99),
        figsize=(6, 12),
    )
    ax.set_axis_off()
    ax.set_xlim(438000, 444400)
    ax.set_ylim(4472000, 4478400)
    ax.set_title(label)
    fig.savefig(images_path / f"map_{col}.png")

# %%
distances_cent = [500, 1000, 2000, 5000, 10000]
mad_gpd = common.generate_close_n_cols(mad_gpd, distances_cent, length_weighted=True)
mad_gpd = common.generate_close_n_cols(mad_gpd, distances_cent, length_weighted=False)
mad_gpd = mad_gpd.copy()  # fragmentation warnings

# %%
close_cols = common.generate_cent_columns(
    [
        "closeness_{d}",
        "close_N1_{d}",
        "close_N1.2_{d}",
        "close_N2_{d}",
        "harmonic_{d}",
    ],
    distances_cent,
)
# Generate labels for each distance
close_labels = []
for _pattern, label in [
    ("closeness", "Closeness"),
    ("close_N1", r"Normalised $N^{1}$"),
    ("close_N1.2", r"NAIN $N^{1.2}$"),
    ("close_N2", r"Improved $N^{2}$"),
    ("harmonic", "Harmonic"),
]:
    for d in distances_cent:
        close_labels.append(f"{label} {d}m")

# filter columns
mad_gpd_close_filter = mad_gpd[close_cols]
corr_close = mad_gpd_close_filter.corr()
fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr_close,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    xticklabels=close_labels,
    yticklabels=close_labels,
)
ax.set_title("Correlation Matrix - Closeness Centrality Measures", pad=20)
fig.savefig(images_path / "cent_corr_matrix_close.pdf")

# %%
betw_cols = common.generate_cent_columns(
    [
        "betw_{d}",
        "betw_wt_{d}",
        "betw_{d}_ang",
        "NACH_{d}",
        "NACH_{d}_ang",
    ],
    distances_cent,
)
# Generate labels for each distance
betw_labels = []
for _pattern, label in [
    ("betw", "Betw."),
    ("betw_wt", "Wt Betw."),
    ("betw_ang", "Ang. Betw."),
    ("NACH", "NACH"),
    ("NACH_ang", "Ang. NACH"),
]:
    for d in distances_cent:
        betw_labels.append(f"{label} {d}m")

# filter columns
mad_gpd_betw_filter = mad_gpd[betw_cols]
corr_betw = mad_gpd_betw_filter.corr()
fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(
    corr_betw,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
    xticklabels=betw_labels,
    yticklabels=betw_labels,
)
ax.set_title("Correlation Matrix - Betweenness Centrality Measures", pad=20)
fig.savefig(images_path / "cent_corr_matrix_betw.pdf")

# %%
# plot correlations
cent_cols = common.generate_cent_columns(
    [
        "density_{d}",
        "far_{d}",
        "far_norm_{d}",
        "closeness_{d}",
        "close_N1_{d}",
        "close_N1.2_{d}",
        "close_N2_{d}",
        "harmonic_{d}",
        "gravity_{d}",
        "cycles_{d}",
        "betw_{d}",
        "betw_wt_{d}",
        "NACH_{d}",
    ],
    distances_cent,
)
cent_cols_ang = common.generate_cent_columns(
    [
        "far_{d}_ang",
        "far_norm_{d}_ang",
        "closeness_{d}_ang",
        "close_N1_{d}_ang",
        "close_N1.2_{d}_ang",
        "close_N2_{d}_ang",
        "harmonic_{d}_ang",
        "betw_{d}_ang",
        "NACH_{d}_ang",
    ],
    distances_cent,
)
lw_cent_cols = common.generate_cent_columns(
    [
        "lw_density_{d}",
        "density_{d}_seg",
        "lw_far_{d}",
        "lw_far_norm_{d}",
        "lw_closeness_{d}",
        "lw_close_N1_{d}",
        "lw_close_N1.2_{d}",
        "lw_close_N2_{d}",
        "lw_harmonic_{d}",
        "harmonic_{d}_seg",
        "lw_gravity_{d}",
        "gravity_{d}_seg",
        "lw_betw_{d}",
        "lw_betw_wt_{d}",
        "betw_{d}_seg",
        "lw_NACH_{d}",
    ],
    distances_cent,
)
lw_cent_cols_ang = common.generate_cent_columns(
    [
        "lw_far_{d}_ang",
        "lw_far_norm_{d}_ang",
        "lw_closeness_{d}_ang",
        "lw_close_N1_{d}_ang",
        "lw_close_N1.2_{d}_ang",
        "lw_close_N2_{d}_ang",
        "lw_harmonic_{d}_ang",
        "lw_betw_{d}_ang",
        "lw_NACH_{d}_ang",
    ],
    distances_cent,
)

# %%
cent_labels = [
    "Density",
    "Farness",
    "Farness N",
    r"Closeness",
    r"Closeness $N^{1}$",
    r"NAIN $N^{1.2}$",
    r"Improved $N^{2}$",
    "Harmonic",
    "Gravity",
    "Cycles",
    "Between.",
    "wt. Between.",
    "NACH",
]
cent_labels_ang = [
    "ang. Farness",
    "ang. Farness N",
    r"ang. Closeness",
    r"ang. Closeness $N^{1}$",
    r"ang. NAIN $N^{1.2}$",
    r"ang. Improved $N^{2}$",
    "ang. Harmonic",
    "ang. Between.",
    "ang. NACH",
]
lw_cent_labels = [
    "Density",
    "Density cont.",
    "Farness",
    "Farness N",
    "Closeness",
    r"Closeness $N^{1}$",
    r"NAIN $N^{1.2}$",
    r"Improved $N^{2}$",
    "Harmonic",
    "Harmonic cont.",
    "Gravity",
    "Gravity cont.",
    "Between.",
    "wt. Between.",
    "Between. cont.",
    "NACH",
]
lw_cent_labels_ang = [
    "ang. Farness",
    "ang. Farness N",
    "ang. Closeness",
    r"ang. Closeness $N^{1}$",
    r"ang. NAIN $N^{1.2}$",
    r"ang. Improved $N^{2}$",
    "ang. Harmonic",
    "ang. Between.",
    "ang. NACH",
]

# %%
# distances: [100, 200, 500]
# betas: [0.03999999910593033, 0.019999999552965164, 0.00800000037997961]
# avg dist: [35.11949157714844, 70.23898315429688, 175.59747314453125]
for cols, corr_labels, suptitle, cent_lu_corr_path, c_cols, c_labels in [
    [
        ("pca_1", "food_bev_200", "retail_200"),
        (
            "PCA 1",
            r"Food/Bev $\mu=70m$ / $d_{max}=200$",
            r"Retail $\mu=70m$ / $d_{max}=200$",
        ),
        "Spearman Rank - centrality",
        "cent_lu_corrs.pdf",
        cent_cols,
        cent_labels,
    ],
    [
        ("pca_1", "food_bev_200", "retail_200"),
        (
            "PCA 1",
            r"Food/Bev $\mu=70m$ / $d_{max}=200$",
            r"Retail $\mu=70m$ / $d_{max}=200$",
        ),
        "Spearman Rank - length weighted centrality",
        "cent_lu_corrs_length_wtd.pdf",
        lw_cent_cols,
        lw_cent_labels,
    ],
    [
        ("pca_1", "food_bev_200", "retail_200"),
        (
            "PCA 1",
            r"Food/Bev $\mu=70m$ / $d_{max}=200$",
            r"Retail $\mu=70m$ / $d_{max}=200$",
        ),
        "Spearman Rank - angular centrality",
        "cent_lu_corrs_ang.pdf",
        cent_cols_ang,
        cent_labels_ang,
    ],
    [
        ("pca_1", "food_bev_200", "retail_200"),
        (
            "PCA 1",
            r"Food/Bev $\mu=70m$ / $d_{max}=200$",
            r"Retail $\mu=70m$ / $d_{max}=200$",
        ),
        "Spearman Rank - length weighted angular centrality",
        "cent_lu_corrs_length_wtd_ang.pdf",
        lw_cent_cols_ang,
        lw_cent_labels_ang,
    ],
    [
        ("pca_1_ang", "food_bev_200_ang", "retail_200_ang"),
        (
            "PCA 1",
            r"Food/Bev $\mu=70m$ / $d_{max}=200$",
            r"Retail $\mu=70m$ / $d_{max}=200$",
        ),
        "Spearman Rank - centrality (geometric lu. dist.)",
        "cent_lu_corrs_pca_geometric.pdf",
        cent_cols,
        cent_labels,
    ],
]:
    mad_gpd_cent_filter = mad_gpd[c_cols]
    # create heatmaps for original variables plotted against correlations

    heatmap_height = len(c_labels) * 0.55
    fig, axes = plt.subplots(
        1, 3, figsize=(8, heatmap_height), sharey=True, dpi=200, constrained_layout=True
    )
    fig.suptitle(suptitle, fontsize=14)

    for col, corr_label, heatmap_ax in zip(cols, corr_labels, axes, strict=True):
        corr = mad_gpd_cent_filter.corrwith(mad_gpd[col], method="spearman", numeric_only=True)
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
    fig.savefig(images_path / cent_lu_corr_path, bbox_inches="tight")

# %%
bw_log_1000 = "betw_wt_1000_log"
mad_gpd[bw_log_1000] = np.log(mad_gpd["betw_wt_1000"] + 1)

fig, axes = plt.subplots(2, 2, sharey=True, figsize=(8, 8))
sns.kdeplot(
    ax=axes[0][0],
    data=mad_gpd,
    x=bw_log_1000,
    y="pca_1",
)
axes[0][0].set_xlabel(r"Log distance weighted betweenness $d_{\max}=1000m$")
axes[0][0].set_ylabel(r"PCA 1 - Landuses")

sns.kdeplot(
    ax=axes[0][1],
    data=mad_gpd,
    x="close_N2_1000",
    y="pca_1",
    kind="kde",
)
axes[0][1].set_xlabel(r"Improved Closeness $N^{2}$ $d_{\max}=1000m$")

bw_log_5000 = "betw_wt_5000_log"
mad_gpd[bw_log_5000] = np.log(mad_gpd["betw_wt_5000"] + 1)

sns.kdeplot(
    ax=axes[1][0],
    data=mad_gpd,
    x=bw_log_5000,
    y="pca_1",
)
axes[1][0].set_xlabel(r"Log distance weighted betweenness $d_{\max}=5000m$")
axes[1][0].set_ylabel(r"PCA 1 - Landuses")

sns.kdeplot(
    ax=axes[1][1],
    data=mad_gpd,
    x="close_N2_5000",
    y="pca_1",
    kind="kde",
)
axes[1][1].set_xlabel(r"Improved Closeness $N^{2}$ $d_{\max}=5000m$")

plt.tight_layout()
plt.savefig(images_path / "closen_vs_betw_vs_mixed.pdf")

mad_gpd.drop(columns=[bw_log_1000], inplace=True)
mad_gpd.drop(columns=[bw_log_5000], inplace=True)

# %%
ax = mad_gpd.plot(
    column="close_N2_1000",
    cmap="Reds",
    linewidth=1,
    # vmin=0,
    # vmax=1,
)
ax.set_axis_off()
ax.set_xlim(438000, 444400)
ax.set_ylim(4472000, 4478400)

ax = mad_gpd.plot(
    column="harmonic_1000",
    cmap="Reds",
    linewidth=1,
    # vmin=0,
    # vmax=1,
)
ax.set_axis_off()
ax.set_xlim(438000, 444400)
ax.set_ylim(4472000, 4478400)

# %%
for is_angular in [False, True]:
    fig, axes = plt.subplots(3, 2, figsize=(8, 12), dpi=150, constrained_layout=True)
    # Iterate over the subplot axes
    for n, (col, label) in enumerate(
        [
            ("closeness_{d}", r"Closeness"),
            ("close_N1_{d}", r"Closeness $N^{1}$ - Normalised"),
            ("close_N1.2_{d}", r"Closeness $N^{1.2}$ - NAIN"),
            ("close_N2_{d}", r"Closeness $N^{2}$ - Improved"),
            ("harmonic_{d}", "Harmonic"),
            ("gravity_{d}", "Gravity"),
        ]
    ):
        target_col = col.format(d=1000)
        col_temp = f"{target_col}_plot_temp"
        title = f"1000m {label}"
        row_n = n // 2
        col_n = n % 2
        if is_angular is True:
            target_col += "_ang"
            title += " - angular"
        if "gravity" in target_col and is_angular is True:
            axes[row_n][col_n].axis("off")
        else:
            vals = mad_gpd[target_col]
            vals.fillna(0, inplace=True)
            vals -= np.nanmin(vals)
            vals = np.clip(vals, 0, np.nanpercentile(vals, 99))
            # enhance contrast
            vals = vals**1.5
            vals /= np.nanmax(vals)
            vals = 0.2 + vals * 0.8
            mad_gpd[col_temp] = vals
            ax = mad_gpd.plot(
                ax=axes[row_n][col_n],
                column=col_temp,
                cmap="Reds",
                linewidth=vals,
                vmin=0,
                vmax=1,
            )
            axes[row_n][col_n].set_title(title)
            axes[row_n][col_n].set_axis_off()
            axes[row_n][col_n].set_xlim(438000, 444400)
            axes[row_n][col_n].set_ylim(4472000, 4478400)
            cx.add_basemap(
                ax, crs=mad_gpd.crs.to_epsg(), source=cx.providers.CartoDB.PositronNoLabels
            )
            ax.add_artist(NorthArrow(location="upper right", scale=0.25, rotation={"degrees": 0}))
            ax.add_artist(
                scalebar.ScaleBar(1, units="m", length_fraction=0.25, location="lower right")
            )
            mad_gpd.drop(columns=[col_temp], inplace=True)

    if is_angular is False:
        # Save the figure
        fig.savefig(images_path / "closeness_compare.png")
    else:
        # Save the figure
        fig.savefig(images_path / "closeness_compare_ang.png")

# %%
fig, axes = plt.subplots(3, 2, figsize=(8, 12), dpi=150, constrained_layout=True)
for col_n, (col, label) in enumerate(
    [("betw_{d}", "Betweenness"), ("betw_{d}_ang", "Betweenness - angular")]
):
    for row_n, dist in enumerate([1000, 5000, 10000]):
        target_col = col.format(d=dist)
        vals = mad_gpd[target_col]
        vals.fillna(0, inplace=True)
        vals -= np.nanmin(vals)
        vals = np.clip(vals, 0, np.nanpercentile(vals, 98))
        vals /= np.nanmax(vals)
        vals = 0.2 + vals * 0.8
        col_temp = f"{target_col}_plot_temp"
        mad_gpd[col_temp] = vals
        ax = mad_gpd.plot(
            ax=axes[row_n][col_n],
            column=col_temp,
            cmap="Reds",
            linewidth=vals,
            vmin=0,
            vmax=1,
        )
        axes[row_n][col_n].set_title(f"{label} {int(dist / 1000)}km")
        axes[row_n][col_n].set_axis_off()
        axes[row_n][col_n].set_xlim(438000, 444400)
        axes[row_n][col_n].set_ylim(4472000, 4478400)
        cx.add_basemap(ax, crs=mad_gpd.crs.to_epsg(), source=cx.providers.CartoDB.PositronNoLabels)
        ax.add_artist(NorthArrow(location="upper right", scale=0.25, rotation={"degrees": 0}))
        ax.add_artist(scalebar.ScaleBar(1, units="m", length_fraction=0.25, location="lower right"))
        mad_gpd.drop(columns=[col_temp], inplace=True)

    fig.savefig(images_path / "betweenness_compare.png")

# %%
# Descriptive statistics tables
lu_cols = ["pca_1", "food_bev_200", "retail_200"]
tables_dir = pathlib.Path(tables_path)
tables_dir.mkdir(exist_ok=True)

for col_set, cols_label in zip(
    [
        lu_cols,
        [
            "density_{d}",
            "far_{d}",
            "far_norm_{d}",
            "closeness_{d}",
            "close_N1_{d}",
            "close_N1.2_{d}",
            "close_N2_{d}",
            "harmonic_{d}",
            "gravity_{d}",
        ],
        [
            "far_{d}_ang",
            "far_norm_{d}_ang",
            "closeness_{d}_ang",
            "close_N1_{d}_ang",
            "close_N1.2_{d}_ang",
            "close_N2_{d}_ang",
            "harmonic_{d}_ang",
        ],
        [
            "lw_density_{d}",
            "lw_far_{d}",
            "lw_far_norm_{d}",
            "lw_closeness_{d}",
            "lw_close_N1_{d}",
            "lw_close_N1.2_{d}",
            "lw_close_N2_{d}",
            "lw_harmonic_{d}",
            "lw_gravity_{d}",
        ],
        [
            "lw_far_{d}_ang",
            "lw_far_norm_{d}_ang",
            "lw_closeness_{d}_ang",
            "lw_close_N1_{d}_ang",
            "lw_close_N1.2_{d}_ang",
            "lw_close_N2_{d}_ang",
            "lw_harmonic_{d}_ang",
        ],
    ],
    [
        "Land Use",
        "Centrality",
        "Centrality Angular",
        "Centrality Length Weighted",
        "Centrality Length Weighted Angular",
    ],
):
    # construct full column names for distance variants
    if "Land Use" not in cols_label:
        full_col_set = []
        for pattern in col_set:
            for d in distances_cent:
                full_col_set.append(pattern.format(d=d))
        col_set = full_col_set
    # Generate descriptive stats with IQR
    desc_stats = mad_gpd[col_set].describe().T
    # Add IQR calculation
    desc_stats["IQR"] = desc_stats["75%"] - desc_stats["25%"]
    # Select columns in desired order: N, mean, median, Q1, Q3, IQR, min, max
    desc_stats_formatted = desc_stats[
        ["count", "mean", "50%", "25%", "75%", "IQR", "min", "max"]
    ].copy()
    desc_stats_formatted.columns = ["N", "Mean", "Median", "Q1", "Q3", "IQR", "Min", "Max"]
    # Convert N column to integer
    desc_stats_formatted["N"] = desc_stats_formatted["N"].astype(int)

    # Format numeric columns
    # Bigger than 10: 0 decimals; 0.1-10: 3 decimals; smaller than 0.1: scientific notation
    numeric_cols = ["Mean", "Median", "Q1", "Q3", "IQR", "Min", "Max"]

    for idx in desc_stats_formatted.index:
        for col in numeric_cols:
            val = float(desc_stats_formatted.loc[idx, col])
            if abs(val) > 10:
                desc_stats_formatted.loc[idx, col] = f"{val:.0f}"
            elif abs(val) >= 0.1:
                desc_stats_formatted.loc[idx, col] = f"{val:.3f}"
            else:
                desc_stats_formatted.loc[idx, col] = f"{val:.2e}"

    # Escape underscores in index names for LaTeX
    desc_stats_formatted.index = [idx.replace("_", "\\_") for idx in desc_stats_formatted.index]

    # Save as LaTeX table
    desc_stats_path = tables_dir / f"descriptive_stats_{cols_label.lower().replace(' ', '_')}.tex"
    latex_str = desc_stats_formatted.to_latex(escape=False)
    with open(desc_stats_path, "w") as f:
        f.write(latex_str)

    print(f"Saved {desc_stats_path}")
    print(f"\n{cols_label}:")
    print(desc_stats_formatted.to_string())

# %%
# Moran's I calculation using k-nearest neighbours
# k is derived from median network density at each distance threshold

# Columns to test for Moran's I
morans_cent_cols = [
    "closeness_{d}",
    "close_N1_{d}",
    "close_N1.2_{d}",
    "close_N2_{d}",
    "harmonic_{d}",
    "gravity_{d}",
    "betw_{d}",
    "betw_wt_{d}",
    "NACH_{d}",
]
morans_distances = [500, 1000, 2000]  # , 5000, 10000

# Precompute coordinates once
coords = np.array(list(mad_gpd.geometry.centroid.apply(lambda p: (p.x, p.y))))

# Store results
morans_results = []

# Use median network density at each distance as k
# This matches the spatial scale of analysis to the neighbour definition
for d in morans_distances:
    density_col = f"density_{d}"
    k = int(mad_gpd[density_col].median())
    # Ensure k is at least 1 and not larger than n-1
    k = max(1, min(k, len(mad_gpd) - 1))

    print(f"\n{'=' * 60}")
    print(f"Distance threshold: {d}m, median network density (k): {k}")
    print(f"{'=' * 60}")

    # Build weights matrix for this distance threshold
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = "R"

    for col_pattern in morans_cent_cols:
        col = col_pattern.format(d=d)
        if col not in mad_gpd.columns:
            continue

        y = mad_gpd[col].values.copy()
        # Handle NaN values by replacing with mean
        nan_mask = np.isnan(y)
        if nan_mask.any():
            y[nan_mask] = np.nanmean(y)

        try:
            mi = esda.Moran(y, w, permutations=999)
            morans_results.append(
                {
                    "variable": col,
                    "distance": d,
                    "k": k,
                    "I": mi.I,
                    "p_norm": mi.p_norm,
                    "p_sim": mi.p_sim,
                    "z_norm": mi.z_norm,
                }
            )
            print(
                f"{col}: I={mi.I:.4f}, z={mi.z_norm:.2f}, p_norm={mi.p_norm:.4g}, p_sim={mi.p_sim:.4g}"
            )
        except Exception as e:
            print(f"{col}: Error - {e}")
# %%
# Also run sensitivity analysis with fixed k values for comparison
print("\n" + "=" * 60)
print("Sensitivity analysis: fixed k values for comparison")
print("=" * 60)

for k_fixed in [8, 20, 50]:
    print(f"\nk = {k_fixed}")
    w_fixed = libpysal.weights.KNN.from_array(coords, k=k_fixed)
    w_fixed.transform = "R"

    # Test one representative variable at each distance
    for d in morans_distances:
        col = f"close_N2_{d}"
        if col not in mad_gpd.columns:
            continue
        y = mad_gpd[col].values.copy()
        nan_mask = np.isnan(y)
        if nan_mask.any():
            y[nan_mask] = np.nanmean(y)
        try:
            mi = esda.Moran(y, w_fixed, permutations=999)
            morans_results.append(
                {
                    "variable": col,
                    "distance": d,
                    "k": k_fixed,
                    "I": mi.I,
                    "p_norm": mi.p_norm,
                    "p_sim": mi.p_sim,
                    "z_norm": mi.z_norm,
                }
            )
            print(f"  {col}: I={mi.I:.4f}, p_sim={mi.p_sim:.4g}")
        except Exception as e:
            print(f"  {col}: Error - {e}")

# Convert to DataFrame
morans_df = pd.DataFrame(morans_results)

# Save full results to CSV
morans_df.to_csv(tables_path / "morans_i_segments.csv", index=False)
print(f"\nSaved full Moran's I results to {tables_path / 'morans_i_segments.csv'}")

# Create summary table using network-based k for supplementary materials
# Filter to only rows where k matches the median density (exclude sensitivity analysis)
k_map = {d: int(mad_gpd[f"density_{d}"].median()) for d in morans_distances}
morans_network = morans_df[
    morans_df.apply(lambda x: x["k"] == k_map.get(x["distance"]), axis=1)
].copy()
morans_network["variable"] = morans_network["variable"].str.replace("_", "\\_", regex=False)
morans_network_formatted = morans_network[
    ["variable", "distance", "k", "I", "z_norm", "p_norm", "p_sim"]
].copy()
morans_network_formatted.columns = [
    "Variable",
    "Distance (m)",
    "k (median density)",
    "Moran's I",
    "z-score",
    "p (analytic)",
    "p (permutation)",
]
morans_network_formatted["Moran's I"] = morans_network_formatted["Moran's I"].apply(
    lambda x: f"{x:.4f}"
)
morans_network_formatted["z-score"] = morans_network_formatted["z-score"].apply(
    lambda x: f"{x:.2f}"
)
morans_network_formatted["p (analytic)"] = morans_network_formatted["p (analytic)"].apply(
    lambda x: f"{x:.4g}"
)
morans_network_formatted["p (permutation)"] = morans_network_formatted["p (permutation)"].apply(
    lambda x: f"{x:.4g}"
)

latex_str = morans_network_formatted.to_latex(index=False, escape=False)
with open(tables_path / "morans_i_segments_network.tex", "w") as f:
    f.write(latex_str)
print(f"Saved LaTeX table to {tables_path / 'morans_i_segments_network.tex'}")

# Print summary of stability across k values for close_N2
print("\n" + "=" * 60)
print("Sensitivity analysis: Moran's I for close_N2 across k values")
print("=" * 60)
close_n2_results = morans_df[morans_df["variable"].str.startswith("close_N2_")]
pivot = close_n2_results.pivot_table(index="variable", columns="k", values="I")
print(pivot.round(4).to_string())

# Effective sample size (Neff) calculation
# Neff ≈ N * (1 - I) / (1 + I) where I is Moran's I
# This accounts for positive spatial autocorrelation reducing effective information

N = len(mad_gpd)
print("\n" + "=" * 60)
print("Effective Sample Size (Neff) based on Moran's I")
print("=" * 60)

# Compute Neff for all measures tested in Moran's I (using network-based k only)
neff_results = []
for d in morans_distances:
    density_col = f"density_{d}"
    k = int(mad_gpd[density_col].median())
    k = max(1, min(k, len(mad_gpd) - 1))

    print(f"\n{d}m (k={k}):")
    for col_pattern in morans_cent_cols:
        col = col_pattern.format(d=d)
        # Find the Moran's I value from our results (network-based k only)
        mask = (morans_df["variable"] == col) & (morans_df["k"] == k)
        if mask.any():
            I = morans_df.loc[mask, "I"].values[0]
            # Neff formula for positive autocorrelation
            neff = N * (1 - I) / (1 + I)
            neff_ratio = neff / N
            neff_results.append(
                {
                    "variable": col,
                    "distance": d,
                    "k": k,
                    "I": I,
                    "N": N,
                    "Neff": int(neff),
                    "Neff_ratio": neff_ratio,
                }
            )
            print(f"  {col}: I={I:.4f}, Neff={int(neff):,} ({neff_ratio:.1%})")

neff_df = pd.DataFrame(neff_results)
neff_df.to_csv(tables_path / "neff_segments.csv", index=False)

# Create formatted LaTeX table for Neff
neff_formatted = neff_df[["variable", "distance", "I", "Neff", "Neff_ratio"]].copy()
neff_formatted["variable"] = neff_formatted["variable"].str.replace("_", "\\_", regex=False)
neff_formatted.columns = ["Variable", "Distance (m)", "Moran's I", "Neff", "Neff/N"]
neff_formatted["Moran's I"] = neff_formatted["Moran's I"].apply(lambda x: f"{x:.4f}")
neff_formatted["Neff"] = neff_formatted["Neff"].apply(lambda x: f"{x:,}")
neff_formatted["Neff/N"] = neff_formatted["Neff/N"].apply(lambda x: f"{x:.1%}".replace("%", "\\%"))

latex_str = neff_formatted.to_latex(index=False, escape=False)
with open(tables_path / "neff_segments.tex", "w") as f:
    f.write(latex_str)
print(f"\nSaved Neff results to {tables_path / 'neff_segments.csv'}")
print(f"Saved LaTeX table to {tables_path / 'neff_segments.tex'}")

# %%
# Block bootstrap confidence intervals for key correlations
# Resample spatial blocks to preserve local spatial structure


def block_bootstrap_spearman(x, y, block_labels, n_boot=1000, seed=42):
    """
    Block bootstrap for Spearman correlation.
    Resamples entire blocks to preserve spatial structure.
    """
    # Handle NaN values upfront
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if not valid_mask.any():
        return np.nan, np.nan, np.nan  # Return NaN if all values invalid

    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    block_labels_clean = block_labels[valid_mask]

    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block_labels_clean)
    n_blocks = len(unique_blocks)

    rhos = []
    for _ in range(n_boot):
        # Sample blocks with replacement
        sampled_blocks = rng.choice(unique_blocks, size=n_blocks, replace=True)
        # Get indices for all observations in sampled blocks
        indices = np.concatenate([np.where(block_labels_clean == b)[0] for b in sampled_blocks])
        # Compute Spearman correlation on resampled data
        rho, _ = stats.spearmanr(x_clean[indices], y_clean[indices])
        rhos.append(rho)

    rhos = np.array(rhos)
    ci_low, ci_high = np.percentile(rhos, [2.5, 97.5])
    return np.nanmean(rhos), ci_low, ci_high


# Create spatial blocks using KMeans on coordinates
n_blocks = 100  # ~400 segments per block on average
print("\n" + "=" * 60)
print(f"Block Bootstrap CI (n_blocks={n_blocks}, n_boot=1000)")
print("=" * 60)

kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
block_labels = kmeans.fit_predict(coords)
print(f"Created {n_blocks} spatial blocks, median size: {np.median(np.bincount(block_labels)):.0f}")

# Compute bootstrap CI for all measures tested in Moran's I vs pca_1
bootstrap_results = []
pca_1 = mad_gpd["pca_1"].values

for d in morans_distances:
    print(f"\n{d}m:")
    for col_pattern in morans_cent_cols:
        col = col_pattern.format(d=d)
        if col not in mad_gpd.columns:
            continue

        x = mad_gpd[col].values

        # Standard Spearman correlation with NaN/Inf handling
        valid_mask = ~(np.isnan(x) | np.isnan(pca_1) | np.isinf(x) | np.isinf(pca_1))
        if not valid_mask.any():
            rho_obs = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            x_clean = x[valid_mask]
            pca_1_clean = pca_1[valid_mask]
            rho_obs, p_obs = stats.spearmanr(x_clean, pca_1_clean)

            # Block bootstrap CI (only if we have valid data)
            block_labels_clean = block_labels[valid_mask]
            rho_boot, ci_low, ci_high = block_bootstrap_spearman(
                x_clean, pca_1_clean, block_labels_clean, n_boot=1000
            )

        # Get Neff for this measure
        neff_row = neff_df[(neff_df["variable"] == col) & (neff_df["distance"] == d)]
        neff = neff_row["Neff"].values[0] if len(neff_row) > 0 else N
        I_val = neff_row["I"].values[0] if len(neff_row) > 0 else np.nan

        bootstrap_results.append(
            {
                "variable": col,
                "distance": d,
                "rho": rho_obs,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_width": ci_high - ci_low,
                "I": I_val,
                "Neff": neff,
            }
        )
        print(f"  {col} vs pca_1: ρ={rho_obs:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")

# Save results
bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(tables_path / "bootstrap_ci_segments.csv", index=False)

# Create formatted LaTeX table
bootstrap_formatted = bootstrap_df[
    ["variable", "distance", "rho", "ci_low", "ci_high", "I", "Neff"]
].copy()
bootstrap_formatted.columns = [
    "Variable",
    "Distance (m)",
    "rho",
    "CI Low",
    "CI High",
    "Moran's I",
    "Neff",
]
bootstrap_formatted["rho"] = bootstrap_formatted["rho"].apply(lambda x: f"{x:.3f}")
bootstrap_formatted["CI Low"] = bootstrap_formatted["CI Low"].apply(lambda x: f"{x:.3f}")
bootstrap_formatted["CI High"] = bootstrap_formatted["CI High"].apply(lambda x: f"{x:.3f}")
bootstrap_formatted["Moran's I"] = bootstrap_formatted["Moran's I"].apply(lambda x: f"{x:.4f}")
bootstrap_formatted["Neff"] = bootstrap_formatted["Neff"].apply(lambda x: f"{x:,}")

# Rename for LaTeX
bootstrap_formatted.rename(columns={"rho": "$\\rho$"}, inplace=True)

# Escape underscores in the Variable column for LaTeX output only
bootstrap_formatted["Variable"] = bootstrap_formatted["Variable"].str.replace(
    "_", "\\_", regex=False
)

latex_str = bootstrap_formatted.to_latex(index=False, escape=False)
with open(tables_path / "bootstrap_ci_segments.tex", "w") as f:
    f.write(latex_str)

print(f"\nSaved bootstrap CI results to {tables_path / 'bootstrap_ci_segments.csv'}")
print(f"Saved LaTeX table to {tables_path / 'bootstrap_ci_segments.tex'}")

# %%
