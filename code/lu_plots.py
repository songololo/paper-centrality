# %%
import pathlib

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA

sns.set_theme(style="dark")
sns.set_context("paper")

# %%
# this is a relatively large dataset and may take a while to load
mad_gpd = gpd.read_file("../temp/dataset.gpkg")

# %%
images_path = pathlib.Path("../plots")

# %%
# rename columns for handling / plots
mad_gpd = mad_gpd.rename(
    columns={
        # centralities
        "cc_beta_500": "gravity_500",
        "cc_beta_1000": "gravity_1000",
        "cc_beta_2000": "gravity_2000",
        "cc_beta_5000": "gravity_5000",
        "cc_beta_10000": "gravity_10000",
        "cc_cycles_500": "cycles_500",
        "cc_cycles_1000": "cycles_1000",
        "cc_cycles_2000": "cycles_2000",
        "cc_cycles_5000": "cycles_5000",
        "cc_cycles_10000": "cycles_10000",
        "cc_density_500": "density_500",
        "cc_density_1000": "density_1000",
        "cc_density_2000": "density_2000",
        "cc_density_5000": "density_5000",
        "cc_density_10000": "density_10000",
        "cc_farness_500": "far_500",
        "cc_farness_1000": "far_1000",
        "cc_farness_2000": "far_2000",
        "cc_farness_5000": "far_5000",
        "cc_farness_10000": "far_10000",
        "cc_harmonic_500": "harmonic_500",
        "cc_harmonic_1000": "harmonic_1000",
        "cc_harmonic_2000": "harmonic_2000",
        "cc_harmonic_5000": "harmonic_5000",
        "cc_harmonic_10000": "harmonic_10000",
        "cc_hillier_500": "close_N2_500",
        "cc_hillier_1000": "close_N2_1000",
        "cc_hillier_2000": "close_N2_2000",
        "cc_hillier_5000": "close_N2_5000",
        "cc_hillier_10000": "close_N2_10000",
        "cc_betweenness_500": "betw_500",
        "cc_betweenness_1000": "betw_1000",
        "cc_betweenness_2000": "betw_2000",
        "cc_betweenness_5000": "betw_5000",
        "cc_betweenness_10000": "betw_10000",
        "cc_betweenness_beta_500": "betw_wt_500",
        "cc_betweenness_beta_1000": "betw_wt_1000",
        "cc_betweenness_beta_2000": "betw_wt_2000",
        "cc_betweenness_beta_5000": "betw_wt_5000",
        "cc_betweenness_beta_10000": "betw_wt_10000",
        # segment
        "cc_seg_density_500": "density_500_seg",
        "cc_seg_density_1000": "density_1000_seg",
        "cc_seg_density_2000": "density_2000_seg",
        "cc_seg_density_5000": "density_5000_seg",
        "cc_seg_density_10000": "density_10000_seg",
        "cc_seg_harmonic_500": "harmonic_500_seg",
        "cc_seg_harmonic_1000": "harmonic_1000_seg",
        "cc_seg_harmonic_2000": "harmonic_2000_seg",
        "cc_seg_harmonic_5000": "harmonic_5000_seg",
        "cc_seg_harmonic_10000": "harmonic_10000_seg",
        "cc_seg_beta_500": "gravity_500_seg",
        "cc_seg_beta_1000": "gravity_1000_seg",
        "cc_seg_beta_2000": "gravity_2000_seg",
        "cc_seg_beta_5000": "gravity_5000_seg",
        "cc_seg_beta_10000": "gravity_10000_seg",
        "cc_seg_betweenness_500": "betw_500_seg",
        "cc_seg_betweenness_1000": "betw_1000_seg",
        "cc_seg_betweenness_2000": "betw_2000_seg",
        "cc_seg_betweenness_5000": "betw_5000_seg",
        "cc_seg_betweenness_10000": "betw_10000_seg",
        # angular
        "cc_density_500_ang": "density_500_ang",
        "cc_density_1000_ang": "density_1000_ang",
        "cc_density_2000_ang": "density_2000_ang",
        "cc_density_5000_ang": "density_5000_ang",
        "cc_density_10000_ang": "density_10000_ang",
        "cc_harmonic_500_ang": "harmonic_500_ang",
        "cc_harmonic_1000_ang": "harmonic_1000_ang",
        "cc_harmonic_2000_ang": "harmonic_2000_ang",
        "cc_harmonic_5000_ang": "harmonic_5000_ang",
        "cc_harmonic_10000_ang": "harmonic_10000_ang",
        "cc_hillier_500_ang": "close_N2_500_ang",
        "cc_hillier_1000_ang": "close_N2_1000_ang",
        "cc_hillier_2000_ang": "close_N2_2000_ang",
        "cc_hillier_5000_ang": "close_N2_5000_ang",
        "cc_hillier_10000_ang": "close_N2_10000_ang",
        "cc_farness_500_ang": "far_500_ang",
        "cc_farness_1000_ang": "far_1000_ang",
        "cc_farness_2000_ang": "far_2000_ang",
        "cc_farness_5000_ang": "far_5000_ang",
        "cc_farness_10000_ang": "far_10000_ang",
        "cc_betweenness_500_ang": "betw_500_ang",
        "cc_betweenness_1000_ang": "betw_1000_ang",
        "cc_betweenness_2000_ang": "betw_2000_ang",
        "cc_betweenness_5000_ang": "betw_5000_ang",
        "cc_betweenness_10000_ang": "betw_10000_ang",
        # landuses
        "cc_food_bev_200_wt": "food_bev_200",
        "cc_food_bev_500_wt": "food_bev_500",
        "cc_food_bev_1000_wt": "food_bev_1000",
        "cc_food_bev_2000_wt": "food_bev_2000",
        "cc_retail_200_wt": "retail_200",
        "cc_retail_500_wt": "retail_500",
        "cc_retail_1000_wt": "retail_1000",
        "cc_retail_2000_wt": "retail_2000",
        "cc_services_200_wt": "services_200",
        "cc_services_500_wt": "services_500",
        "cc_services_1000_wt": "services_1000",
        "cc_services_2000_wt": "services_2000",
        "cc_creat_entert_200_wt": "creat_entert_200",
        "cc_creat_entert_500_wt": "creat_entert_500",
        "cc_creat_entert_1000_wt": "creat_entert_1000",
        "cc_creat_entert_2000_wt": "creat_entert_2000",
        "cc_accommod_200_wt": "accommod_200",
        "cc_accommod_500_wt": "accommod_500",
        "cc_accommod_1000_wt": "accommod_1000",
        "cc_accommod_2000_wt": "accommod_2000",
        # angular
        "cc_food_bev_200_ang_wt": "food_bev_200_ang",
        "cc_food_bev_500_ang_wt": "food_bev_500_ang",
        "cc_food_bev_1000_ang_wt": "food_bev_1000_ang",
        "cc_food_bev_2000_ang_wt": "food_bev_2000_ang",
        "cc_retail_200_ang_wt": "retail_200_ang",
        "cc_retail_500_ang_wt": "retail_500_ang",
        "cc_retail_1000_ang_wt": "retail_1000_ang",
        "cc_retail_2000_ang_wt": "retail_2000_ang",
        "cc_services_200_ang_wt": "services_200_ang",
        "cc_services_500_ang_wt": "services_500_ang",
        "cc_services_1000_ang_wt": "services_1000_ang",
        "cc_services_2000_ang_wt": "services_2000_ang",
        "cc_creat_entert_200_ang_wt": "creat_entert_200_ang",
        "cc_creat_entert_500_ang_wt": "creat_entert_500_ang",
        "cc_creat_entert_1000_ang_wt": "creat_entert_1000_ang",
        "cc_creat_entert_2000_ang_wt": "creat_entert_2000_ang",
        "cc_accommod_200_ang_wt": "accommod_200_ang",
        "cc_accommod_500_ang_wt": "accommod_500_ang",
        "cc_accommod_1000_ang_wt": "accommod_1000_ang",
        "cc_accommod_2000_ang_wt": "accommod_2000_ang",
    }
)

# %%
# landuse PCA columns
lu_cols_shortest = [
    "food_bev_200",
    "food_bev_500",
    "food_bev_1000",
    "food_bev_2000",
    "retail_200",
    "retail_500",
    "retail_1000",
    "retail_2000",
    "services_200",
    "services_500",
    "services_1000",
    "services_2000",
    "creat_entert_200",
    "creat_entert_500",
    "creat_entert_1000",
    "creat_entert_2000",
    "accommod_200",
    "accommod_500",
    "accommod_1000",
    "accommod_2000",
]
lu_cols_simplest = [lc + "_ang" for lc in lu_cols_shortest]
# generate
for is_angular, lu_cols in zip([False, True], [lu_cols_shortest, lu_cols_simplest]):
    # create a copy of the dataframe with non variable cols removed
    mad_gpd_lu_filter = mad_gpd[lu_cols]
    dens_col = 'density_'
    for col in lu_cols:
        if '200' in col:
            dens_col += '200'
        elif '500' in col:
            dens_col += '500'
        elif '1000' in col:
            dens_col += '1000'
        elif '2000' in col:
            dens_col += '2000'
        else:
            raise ValueError('missing distance')
        if is_angular is True:
            dens_col += '_ang'
        mad_gpd_lu_filter[col] /= mad_gpd_lu_filter[dens_col]
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
    model = PCA(n_components=n_components)
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
    lu_distances = [200, 500, 1000, 2000]
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
for col in ["pca_1", "cc_hill_q0_2000_wt", "food_bev_500", "retail_500"]:
    for is_angular in [False, True]:
        if is_angular is True:
            if col == "cc_hill_q0_2000_wt":
                col = "cc_hill_q0_2000_ang_wt"
            else:
                col += "_ang"
        fig, ax = plt.subplots(figsize=(5, 10), dpi=150, constrained_layout=True)
        mad_gpd.plot(
            ax=ax,
            column=col,
            cmap="turbo",
            linewidths=1,
            figsize=(6, 12),
        )
        ax.set_axis_off()
        ax.set_xlim(438000, 444400)
        ax.set_ylim(4470000, 4482000)
        ax.set_title(col)
        lu_map_path = f"map_{col}"
        fig.savefig(images_path / lu_map_path)

# %%
distances_cent = [500, 1000, 2000, 5000, 10000]
# generate the closeness N, N*1.2
for dist in distances_cent:
    # clip to minimum 1 to prevent infinity values for zeros
    # where this happens node density would also be zero but division of 0 / 0 would cause issues
    # shortest
    far_dist = np.clip(mad_gpd[f"far_{dist}"], 1, np.nanmax(mad_gpd[f"far_{dist}"]))
    mad_gpd[f"closeness_{dist}"] = 1 / far_dist
    mad_gpd[f"close_N1_{dist}"] = mad_gpd[f"density_{dist}"] / far_dist
    mad_gpd[f"close_N1.2_{dist}"] = (mad_gpd[f"density_{dist}"] ** 1.2) / far_dist
    # simplest
    far_dist_ang = np.clip(
        mad_gpd[f"far_{dist}_ang"], 1, np.nanmax(mad_gpd[f"far_{dist}_ang"])
    )
    mad_gpd[f"closeness_{dist}_ang"] = 1 / far_dist_ang
    mad_gpd[f"close_N1_{dist}_ang"] = mad_gpd[f"density_{dist}_ang"] / far_dist_ang
    mad_gpd[f"close_N1.2_{dist}_ang"] = (
        mad_gpd[f"density_{dist}_ang"] ** 1.2
    ) / far_dist_ang


# %%
# generate columns
def generate_cent_columns(cols: list[str]):
    formatted_cols = []
    for col in cols:
        for d in distances_cent:
            formatted_cols.append(col.format(d=d))
    return formatted_cols


# %%
close_cols = generate_cent_columns(
    [
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
    ]
)
# filter columns
mad_gpd_close_filter = mad_gpd[close_cols]
corr_close = mad_gpd_close_filter.corr()
fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
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
cent_cols = generate_cent_columns(
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
    ]
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
    "Landuses 2km",
    "Food/Bev 200m",
    "Food/Bev 500m",
    "Retail 200m",
    "Retail 500m",
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
        "cc_hill_q0_2000_wt",
        "food_bev_200",
        "food_bev_500",
        "retail_200",
        "retail_500",
    ]:
        if is_angular is True:
            if col == "cc_hill_q0_2000_wt":
                col = "cc_hill_q0_2000_ang_wt"
            else:
                col += "_ang"
        corrs[col] = mad_gpd_cent_filter.corrwith(
            mad_gpd[col], method="spearman", numeric_only=True
        )
    # create heatmaps for original variables plotted against correlations
    fig, axes = plt.subplots(
        1, 6, figsize=(12, 8), sharey=True, dpi=200, constrained_layout=True
    )
    if is_angular is False:
        suptitle = "Spearman Rank correlations - landuse distances by shortest paths"
    else:
        suptitle = " Spearman Rank correlations - landuse distances by angular paths"
    fig.suptitle(suptitle)
    for n, (corr, corr_label) in enumerate(zip(list(corrs.values()), corr_labels)):
        heatmap_ax = axes[n]
        if is_angular is True:
            corr_label += " ang. dist."
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
