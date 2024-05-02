""" """

import numpy as np


def rename_cent_cols(df):
    # rename columns for handling / plots
    df = df.rename(
        columns={
            # centralities
            "cc_beta_200": "gravity_200",
            "cc_beta_500": "gravity_500",
            "cc_beta_1000": "gravity_1000",
            "cc_beta_2000": "gravity_2000",
            "cc_beta_5000": "gravity_5000",
            "cc_beta_10000": "gravity_10000",
            "cc_cycles_200": "cycles_200",
            "cc_cycles_500": "cycles_500",
            "cc_cycles_1000": "cycles_1000",
            "cc_cycles_2000": "cycles_2000",
            "cc_cycles_5000": "cycles_5000",
            "cc_cycles_10000": "cycles_10000",
            "cc_density_200": "density_200",
            "cc_density_500": "density_500",
            "cc_density_1000": "density_1000",
            "cc_density_2000": "density_2000",
            "cc_density_5000": "density_5000",
            "cc_density_10000": "density_10000",
            "cc_farness_200": "far_200",
            "cc_farness_500": "far_500",
            "cc_farness_1000": "far_1000",
            "cc_farness_2000": "far_2000",
            "cc_farness_5000": "far_5000",
            "cc_farness_10000": "far_10000",
            "cc_harmonic_200": "harmonic_200",
            "cc_harmonic_500": "harmonic_500",
            "cc_harmonic_1000": "harmonic_1000",
            "cc_harmonic_2000": "harmonic_2000",
            "cc_harmonic_5000": "harmonic_5000",
            "cc_harmonic_10000": "harmonic_10000",
            "cc_hillier_200": "close_N2_200",
            "cc_hillier_500": "close_N2_500",
            "cc_hillier_1000": "close_N2_1000",
            "cc_hillier_2000": "close_N2_2000",
            "cc_hillier_5000": "close_N2_5000",
            "cc_hillier_10000": "close_N2_10000",
            "cc_betweenness_200": "betw_200",
            "cc_betweenness_500": "betw_500",
            "cc_betweenness_1000": "betw_1000",
            "cc_betweenness_2000": "betw_2000",
            "cc_betweenness_5000": "betw_5000",
            "cc_betweenness_10000": "betw_10000",
            "cc_betweenness_beta_200": "betw_wt_200",
            "cc_betweenness_beta_500": "betw_wt_500",
            "cc_betweenness_beta_1000": "betw_wt_1000",
            "cc_betweenness_beta_2000": "betw_wt_2000",
            "cc_betweenness_beta_5000": "betw_wt_5000",
            "cc_betweenness_beta_10000": "betw_wt_10000",
            # segment
            "cc_seg_density_200": "density_200_seg",
            "cc_seg_density_500": "density_500_seg",
            "cc_seg_density_1000": "density_1000_seg",
            "cc_seg_density_2000": "density_2000_seg",
            "cc_seg_density_5000": "density_5000_seg",
            "cc_seg_density_10000": "density_10000_seg",
            "cc_seg_harmonic_200": "harmonic_200_seg",
            "cc_seg_harmonic_500": "harmonic_500_seg",
            "cc_seg_harmonic_1000": "harmonic_1000_seg",
            "cc_seg_harmonic_2000": "harmonic_2000_seg",
            "cc_seg_harmonic_5000": "harmonic_5000_seg",
            "cc_seg_harmonic_10000": "harmonic_10000_seg",
            "cc_seg_beta_200": "gravity_200_seg",
            "cc_seg_beta_500": "gravity_500_seg",
            "cc_seg_beta_1000": "gravity_1000_seg",
            "cc_seg_beta_2000": "gravity_2000_seg",
            "cc_seg_beta_5000": "gravity_5000_seg",
            "cc_seg_beta_10000": "gravity_10000_seg",
            "cc_seg_betweenness_200": "betw_200_seg",
            "cc_seg_betweenness_500": "betw_500_seg",
            "cc_seg_betweenness_1000": "betw_1000_seg",
            "cc_seg_betweenness_2000": "betw_2000_seg",
            "cc_seg_betweenness_5000": "betw_5000_seg",
            "cc_seg_betweenness_10000": "betw_10000_seg",
            # angular
            "cc_density_200_ang": "density_200_ang",
            "cc_density_500_ang": "density_500_ang",
            "cc_density_1000_ang": "density_1000_ang",
            "cc_density_2000_ang": "density_2000_ang",
            "cc_density_5000_ang": "density_5000_ang",
            "cc_density_10000_ang": "density_10000_ang",
            "cc_harmonic_200_ang": "harmonic_200_ang",
            "cc_harmonic_500_ang": "harmonic_500_ang",
            "cc_harmonic_1000_ang": "harmonic_1000_ang",
            "cc_harmonic_2000_ang": "harmonic_2000_ang",
            "cc_harmonic_5000_ang": "harmonic_5000_ang",
            "cc_harmonic_10000_ang": "harmonic_10000_ang",
            "cc_hillier_200_ang": "close_N2_200_ang",
            "cc_hillier_500_ang": "close_N2_500_ang",
            "cc_hillier_1000_ang": "close_N2_1000_ang",
            "cc_hillier_2000_ang": "close_N2_2000_ang",
            "cc_hillier_5000_ang": "close_N2_5000_ang",
            "cc_hillier_10000_ang": "close_N2_10000_ang",
            "cc_farness_200_ang": "far_200_ang",
            "cc_farness_500_ang": "far_500_ang",
            "cc_farness_1000_ang": "far_1000_ang",
            "cc_farness_2000_ang": "far_2000_ang",
            "cc_farness_5000_ang": "far_5000_ang",
            "cc_farness_10000_ang": "far_10000_ang",
            "cc_betweenness_200_ang": "betw_200_ang",
            "cc_betweenness_500_ang": "betw_500_ang",
            "cc_betweenness_1000_ang": "betw_1000_ang",
            "cc_betweenness_2000_ang": "betw_2000_ang",
            "cc_betweenness_5000_ang": "betw_5000_ang",
            "cc_betweenness_10000_ang": "betw_10000_ang",
            # landuses
            "cc_food_bev_100_wt": "food_bev_100",
            "cc_food_bev_200_wt": "food_bev_200",
            "cc_food_bev_500_wt": "food_bev_500",
            "cc_food_bev_1000_wt": "food_bev_1000",
            "cc_food_bev_2000_wt": "food_bev_2000",
            "cc_retail_100_wt": "retail_100",
            "cc_retail_200_wt": "retail_200",
            "cc_retail_500_wt": "retail_500",
            "cc_retail_1000_wt": "retail_1000",
            "cc_retail_2000_wt": "retail_2000",
            "cc_services_100_wt": "services_100",
            "cc_services_200_wt": "services_200",
            "cc_services_500_wt": "services_500",
            "cc_services_1000_wt": "services_1000",
            "cc_services_2000_wt": "services_2000",
            "cc_creat_entert_100_wt": "creat_entert_100",
            "cc_creat_entert_200_wt": "creat_entert_200",
            "cc_creat_entert_500_wt": "creat_entert_500",
            "cc_creat_entert_1000_wt": "creat_entert_1000",
            "cc_creat_entert_2000_wt": "creat_entert_2000",
            "cc_accommod_100_wt": "accommod_100",
            "cc_accommod_200_wt": "accommod_200",
            "cc_accommod_500_wt": "accommod_500",
            "cc_accommod_1000_wt": "accommod_1000",
            "cc_accommod_2000_wt": "accommod_2000",
            # angular
            "cc_food_bev_100_ang_wt": "food_bev_100_ang",
            "cc_food_bev_200_ang_wt": "food_bev_200_ang",
            "cc_food_bev_500_ang_wt": "food_bev_500_ang",
            "cc_food_bev_1000_ang_wt": "food_bev_1000_ang",
            "cc_food_bev_2000_ang_wt": "food_bev_2000_ang",
            "cc_retail_100_ang_wt": "retail_100_ang",
            "cc_retail_200_ang_wt": "retail_200_ang",
            "cc_retail_500_ang_wt": "retail_500_ang",
            "cc_retail_1000_ang_wt": "retail_1000_ang",
            "cc_retail_2000_ang_wt": "retail_2000_ang",
            "cc_services_100_ang_wt": "services_100_ang",
            "cc_services_200_ang_wt": "services_200_ang",
            "cc_services_500_ang_wt": "services_500_ang",
            "cc_services_1000_ang_wt": "services_1000_ang",
            "cc_services_2000_ang_wt": "services_2000_ang",
            "cc_creat_entert_100_ang_wt": "creat_entert_100_ang",
            "cc_creat_entert_200_ang_wt": "creat_entert_200_ang",
            "cc_creat_entert_500_ang_wt": "creat_entert_500_ang",
            "cc_creat_entert_1000_ang_wt": "creat_entert_1000_ang",
            "cc_creat_entert_2000_ang_wt": "creat_entert_2000_ang",
            "cc_accommod_100_ang_wt": "accommod_100_ang",
            "cc_accommod_200_ang_wt": "accommod_200_ang",
            "cc_accommod_500_ang_wt": "accommod_500_ang",
            "cc_accommod_1000_ang_wt": "accommod_1000_ang",
            "cc_accommod_2000_ang_wt": "accommod_2000_ang",
        }
    )
    return df


LU_COLS_SHORTEST = [
    "food_bev_100",
    "food_bev_200",
    "food_bev_500",
    "food_bev_1000",
    "food_bev_2000",
    "retail_100",
    "retail_200",
    "retail_500",
    "retail_1000",
    "retail_2000",
    "services_100",
    "services_200",
    "services_500",
    "services_1000",
    "services_2000",
    "creat_entert_100",
    "creat_entert_200",
    "creat_entert_500",
    "creat_entert_1000",
    "creat_entert_2000",
    "accommod_100",
    "accommod_200",
    "accommod_500",
    "accommod_1000",
    "accommod_2000",
]

LU_COLS_SIMPLEST = [lc + "_ang" for lc in LU_COLS_SHORTEST]


def generate_close_n_cols(df, distances: list[int]):
    # generate the closeness N, N*1.2
    for dist in distances:
        # clip to minimum 1 to prevent infinity values for zeros
        # where this happens node density would also be zero but division of 0 / 0 would cause issues
        # shortest
        far_dist = np.clip(df[f"far_{dist}"], 1, np.nanmax(df[f"far_{dist}"]))
        df[f"closeness_{dist}"] = 1 / far_dist
        df[f"close_N1_{dist}"] = df[f"density_{dist}"] / far_dist
        df[f"close_N1.2_{dist}"] = (df[f"density_{dist}"] ** 1.2) / far_dist
        # simplest
        far_dist_ang = np.clip(
            df[f"far_{dist}_ang"], 1, np.nanmax(df[f"far_{dist}_ang"])
        )
        df[f"closeness_{dist}_ang"] = 1 / far_dist_ang
        df[f"close_N1_{dist}_ang"] = df[f"density_{dist}_ang"] / far_dist_ang
        df[f"close_N1.2_{dist}_ang"] = (df[f"density_{dist}_ang"] ** 1.2) / far_dist_ang
    return df


# generate columns
def generate_cent_columns(cols: list[str], distances: list[int]):
    formatted_cols = []
    for col in cols:
        for d in distances:
            formatted_cols.append(col.format(d=d))
    return formatted_cols
