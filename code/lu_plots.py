# %%
import importlib
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import util
from sklearn import preprocessing
from sklearn.decomposition import PCA

importlib.reload(util)

sns.set_theme(style="dark")
sns.set_context("paper")

# %%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("../temp/dataset.gpkg")

# %%
images_path = pathlib.Path("../plots")
mad_gpd = util.rename_cent_cols(mad_gpd)

# %%
# generate
for is_angular, lu_cols in zip(
    [False, True], [util.LU_COLS_SHORTEST, util.LU_COLS_SIMPLEST]
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
    ax.set_title(
        "Correlation matrix" if is_angular is False else "Correlation matrix - angular"
    )
    lu_corr_path = "lu_corr_matrix"
    if is_angular is True:
        lu_corr_path += "_ang"
    fig.savefig(images_path / lu_corr_path)

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
    fig, axes = plt.subplots(
        1, 4, figsize=(10, 3), sharey=True, dpi=200, constrained_layout=True
    )
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
        heatmap_ax.set_title(
            f"PCA {n + 1}" if is_angular is False else f"PCA {n + 1} - angular"
        )
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
    fig.savefig(images_path / pca_corr_path)

# %%
lu_cols = ["pca_1", "cc_hill_q0_200_wt", "food_bev_200", "retail_200"]
lu_labels = [
    "PCA 1",
    "Landuse Richness 70m avg. walk dist. (200m max.)",
    "Food & Beverage 70m avg. walk dist. (200m max.)",
    "Retail 70m avg. walk dist. (200m max.)",
]
for col, label in zip(lu_cols, lu_labels):
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
    lu_map_path = f"map_{col}"
    fig.savefig(images_path / lu_map_path)

# %%
distances_cent = [500, 1000, 2000, 5000, 10000]
mad_gpd = util.generate_close_n_cols(mad_gpd, distances_cent)

# %%
close_cols = util.generate_cent_columns(
    [
        "harmonic_{d}",
        "closeness_{d}",
        "close_N1_{d}",
        "close_N1.2_{d}",
        "close_N2_{d}",
    ],
    distances_cent,
)
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
)
cent_corr_path = "cent_corr_matrix"
fig.savefig(images_path / cent_corr_path)

# %%
# plot correlations
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
        "betw_{d}",
        "betw_wt_{d}",
        "betw_{d}_seg",
        "betw_{d}_ang",
    ],
    distances_cent,
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
    "betweenness",
    "betweenness wt.",
    "betweenness seg.",
    "betweenness ang.",
]
corr_labels = [
    "PCA 1",
    "Landuses - 70m avg.",
    "Food/Bev - 70m avg.",
    "Retail - 70m avg.",
]
mad_gpd_cent_filter = mad_gpd[cent_cols]

# %%
# distances: [100, 200, 500]
# betas: [0.03999999910593033, 0.019999999552965164, 0.00800000037997961]
# avg dist: [35.11949157714844, 70.23898315429688, 175.59747314453125]
for is_angular in [False, True]:
    corrs = {}
    for col in [
        "pca_1",
        "cc_hill_q0_200_wt",
        "food_bev_200",
        "retail_200",
    ]:
        if is_angular is True:
            if col == "cc_hill_q0_200_wt":
                col = "cc_hill_q0_200_ang_wt"
            else:
                col += "_ang"
        corrs[col] = mad_gpd_cent_filter.corrwith(
            mad_gpd[col], method="spearman", numeric_only=True
        )
    # create heatmaps for original variables plotted against correlations
    fig, axes = plt.subplots(
        1, 4, figsize=(8, 8), sharey=True, dpi=200, constrained_layout=True
    )
    if is_angular is False:
        suptitle = "Spearman Rank correlations - landuse distances by shortest paths"
    else:
        suptitle = " Spearman Rank correlations - landuse distances by angular paths"
    fig.suptitle(suptitle)
    for n, (corr, corr_label) in enumerate(zip(list(corrs.values()), corr_labels)):
        heatmap_ax = axes[n]
        if is_angular is True:
            corr_label += " ang."
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
    cent_lu_corr_path = "cent_lu_corrs"
    if is_angular is True:
        cent_lu_corr_path += "_ang"
    fig.savefig(images_path / cent_lu_corr_path)

# %%
col = "betw_wt_10000"
col_log = f"{col}_log"
mad_gpd_filter = mad_gpd[mad_gpd[col] > 1]
mad_gpd_filter[col_log] = np.log(mad_gpd_filter["betw_wt_10000"])
mad_gpd_filter = mad_gpd_filter[
    (mad_gpd_filter[col_log] > np.percentile(mad_gpd_filter[col_log], 1))
]
g = sns.jointplot(
    data=mad_gpd_filter,
    x="gravity_5000",
    y=col_log,
    kind="hex",
)
