import os
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
import datetime
from.visualization_constants import *

show_every_nth_time_point_over100bs = 10
show_every_nth_time_point_below100bs = 6


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """
    Source: https://jwalton.info/Matplotlib-latex-PGF/
    Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def attn_weights_heatmap(parcel_attn_weights, ax_attn_weights, cmap, parcel_class, show_every_nth_time_point, show_title=False):
    parcel_attn_weights.index.name = 'Query'
    ax_attn_weights = sns.heatmap(
        parcel_attn_weights,
        cmap=cmap,
        ax=ax_attn_weights,
        yticklabels=False,
        xticklabels=False,
        cbar=False)

    if show_title:
        ax_attn_weights.set_title("Attention Weights for {} parcel".format(parcel_class))

    ax_attn_weights.tick_params(bottom=False, top=False, left=False)

    ax_attn_weights.set_ylabel('')
    ax_attn_weights.set_xlabel('')

    # ax_attn_weights.set_ylabel(r'\textbf{Query}')
    # ax_attn_weights.set_xlabel(r'\textbf{Key}')

    #ax_attn_weights.tick_params(axis="x", rotation=45)

    return ax_attn_weights


def plot_attn_weights(figures_base_path, temporal_attention_weights, plot_title=None, target_classes=None, y_column_name="Attention",
                      y_plot_label="Average attention", plot_width=260, plot_name="attention_weights_over_time",
                      y_err=None):
    plot_data = temporal_attention_weights.copy(deep=True)
    if target_classes is not None:
        plot_data = plot_data.loc[plot_data["Crop type"].isin(target_classes)]

    fig_width = set_size(plot_width)[0]
    fig, axs = plt.subplots(figsize=(fig_width, 2.5))
    if target_classes is not None:
        axs = sns.lineplot(data=plot_data, x="Date", y=y_column_name, hue="Crop type", estimator="mean", ci="sd",
                           ax=axs, palette=CROP_TYPE_COLOR_MAPPING)
    else:
        plot_data = plot_data.groupby("Date").agg(["mean", "std"])["Attention"]
        plot_data.rename(columns={"mean": y_column_name}, inplace=True)
        plot_data["upper"] = plot_data[y_column_name] + plot_data["std"]
        plot_data["lower"] = plot_data[y_column_name] - plot_data["std"]
        axs = sns.lineplot(data=plot_data, x="Date", y=y_column_name, ci=None, ax=axs)
        axs.fill_between(plot_data.index, plot_data.lower, plot_data.upper, alpha=0.5)

    axs.xaxis.set_major_formatter(DATE_FORMATTER)
    axs.tick_params(axis='x', rotation=45)
    if plot_title is not None:
        axs.set_title(plot_title, fontweight="bold")
    axs.set_xlabel(r'Date')
    axs.set_ylabel(r'{}'.format(y_plot_label))

    axs.legend(loc="best")
    fig.tight_layout()
    plt.savefig(os.path.join(figures_base_path, '{}.pdf'.format(plot_name)))

def load_tiff_image(tiff_image_path):
    return rasterio.open(tiff_image_path).read()


def visualize_geotiff_parcels(date,
                              results_path,
                              target_width_pt=165,
                              geotiff_parcels_path="C:/Users/datasets/BavarianCrops/parcel_visualization/{}/geotiff/max_attention/",
                              show_title=False):

    field_parcels_path = geotiff_parcels_path.format(date)

    fig_width=set_size(target_width_pt)[0]
    fig, axs = plt.subplots(figsize=(fig_width, 2), ncols=2, nrows=2)

    tiff_images_paths = [field_parcel for field_parcel in os.listdir(field_parcels_path)
                         if field_parcel.endswith(".tif")]
    for i, tiff_image in enumerate(tiff_images_paths):
            abs_image_path = os.path.join(field_parcels_path, tiff_image)
            image = load_tiff_image(abs_image_path)
            row_idx = int(i / 2)
            col_idx = i % 2
            ax = axs[row_idx][col_idx]
            show(image, ax=ax)
            crop_type = tiff_image.split('_')[0]
            ax.set_title(crop_type, fontsize=6)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    date_output = datetime.datetime(2018, int(date.split('-')[0]), int(date.split('-')[1]))
    if show_title:
        fig.suptitle(date_output.strftime("%B %d"), fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(results_path, 'field_parcels_{}.pdf'.format(date)))


def key_query_allignment(key_and_query_data, ax, parcel_class):
    ax = sns.scatterplot(
        data=key_and_query_data, x="t-SNE Dim. 1", y="t-SNE Dim. 2", hue="CLUSTER", style="TYPE", ax=ax)
    ax.set_title("Keys and Queries Embeddings for {} parcel".format(parcel_class))
    ax.axhline(0.0)
    ax.axvline(0.0)

    return ax


def visualize_attention_weights(
        parcels_data,
        plot_title,
        viz_dimension="spectral_reflectance"):

    assert viz_dimension in ["spectral_reflectance", "key_query"], 'Invalid viz_dimension parameter'

    num_parcels = len(parcels_data)
    fig, axs = plt.subplots(nrows=2, ncols=num_parcels, figsize=(16, 12))
    cmaps = ['Blues', 'Greens', 'Oranges', 'YlOrBr']

    for parcel_idx, parcel_data in enumerate(parcels_data):
        parcel_class = parcel_data[0]
        parcel_attn_weights = parcel_data[1]
        if len(parcels_data) > 1:
            ax_attn_weights = axs[0][parcel_idx]
            ax_viz_dim = axs[1][parcel_idx]
        else:
            ax_attn_weights = axs[0]
            ax_viz_dim = axs[1]


        number_of_obs = parcel_attn_weights.shape[0]
        if number_of_obs > 100:
            show_every_nth_time_point = show_every_nth_time_point_over100bs
        else:
            show_every_nth_time_point = show_every_nth_time_point_below100bs

        ax_attn_weights = attn_weights_heatmap(parcel_attn_weights,
                                               ax_attn_weights,
                                               cmaps[parcel_idx],
                                               parcel_class,
                                               show_every_nth_time_point)

        if viz_dimension == "spectral_reflectance":
            parcel_spectral_index = parcel_data[2].copy()
            corr_matrix = parcel_spectral_index.corr()[["TOTAL TEMPORAL ATTENTION"]]
            corr_matrix[["ATTN_WEIGHT_CORR_ABS"]] = corr_matrix[["TOTAL TEMPORAL ATTENTION"]].abs()
            print(corr_matrix)
            parcel_most_correlated_bands = corr_matrix[["ATTN_WEIGHT_CORR_ABS"]].nlargest(6, "ATTN_WEIGHT_CORR_ABS").index.values[:6]
            spectral_index_to_plot = parcel_spectral_index[parcel_most_correlated_bands]
            ax_viz_dim = sns.lineplot(data=spectral_index_to_plot, ax=ax_viz_dim)
            ax_viz_dim.set_title("Attention vs Most Correlated Bands")
            if parcel_idx == 0:
                ax_viz_dim.set_ylabel("Reflectance")

        else:
            key_and_query_data = parcel_data[3]
            ax_viz_dim = key_query_allignment(key_and_query_data, ax_viz_dim, parcel_class)

    fig.suptitle(plot_title, fontsize=16)
    fig.tight_layout()


def stacked_boxplot(input_data, title, xlabel, ylabel):
    ax = sns.boxplot(x="layer", y="correlation", hue="head", data=input_data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

    plt.clf()
    plt.close()


# def plot_ndvi_of_crop_vs_others(spectral_indices, target_class):
#     fig, axs = plt.subplots(figsize=(7, 4))
#
#     spectral_indices_grassland = test_dataset_spectral_indices.loc[
#         test_dataset_spectral_indices["CLASS"].isin([target_class])]
#     spectral_indices_not_grassland = test_dataset_spectral_indices.loc[
#         ~test_dataset_spectral_indices["CLASS"].isin([target_class])]
#     axs = sns.lineplot(data=spectral_indices_grassland, x="Date", y="NDVI", hue="CLASS", linestyle="dashed", ci="sd",
#                        ax=axs, palette=crop_type_color_mapping)
#     axs = sns.lineplot(data=spectral_indices_not_grassland, x="Date", y="NDVI", linestyle="dashed", ci="sd", ax=axs,
#                        label="Other crops")
#     axs.set_title("NDVI index throughout the year", fontweight="bold")
#
#
# def plot_ndvi_of_target_classes(spectral_indices, target_classes):
#     fig, axs = plt.subplots(figsize=(7, 4))
#
#     spectral_indices_target_classes = test_dataset_spectral_indices.loc[
#         test_dataset_spectral_indices["CLASS"].isin(target_classes)]
#     axs = sns.lineplot(data=spectral_indices_target_classes, x="Date", y="NDVI", hue="CLASS", linestyle="dashed",
#                        ci="sd", ax=axs, palette=crop_type_color_mapping)
#     axs.set_title("NDVI index throughout the year", fontweight="bold")

#################  UNUSED VISUALIZATION FUNCTIONS ##########################
# def plot_attn_weights_key_queries_row_wise:
    # fig, axs = plt.subplots(ncols=2,figsize=(12, 6))
    # ax_attn_weights = sns.heatmap(
    #             parcels_data[0][1],
    #             cmap="Blues",
    #             ax=axs[0],
    #             yticklabels=5,
    #             xticklabels=5,
    #             cbar=True)
    # ax_attn_weights.set_title("Attention Heatmap for a Corn Parcel")
    #
    # parcel_id = str(parcels_data[1][0].split()[-1])
    # total_temp_attn = total_temporal_attention[parcel_id].to_numpy().flatten()
    # print(total_temp_attn.mean())
    # key_and_query_data = parcels_data[1][3]
    # key_and_query_data["MARKER_SIZE"] = total_temp_attn.mean()
    # print(key_and_query_data)
    # ax_viz_dim = sns.scatterplot(
    #     data=key_and_query_data,
    #     x="emb_dim_1",
    #     y="emb_dim_2",
    #     hue = "CLUSTER",
    #     style="TYPE",
    #     s="MARKER_SIZE",
    #     ax = axs[1])
    # ax_viz_dim.set_title("Keys and Queries Allignment for Corn Parcel")
    # ax_viz_dim.axhline(0.0)
    # ax_viz_dim.axvline(0.0)

#
# attn_weights_per_crop_type_corr = calc_attn_weight_corr_per_crop_type(spectral_indices, mean_temporal_attention)
# fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(12, 8))
#
# idx = 0
# for class_name in attn_weights_per_crop_type_corr.keys():
#     row_idx = int(idx / 6)
#
#     attn_weights_bands_corr = attn_weights_per_crop_type_corr[class_name]
#     target_axs = axs[row_idx][idx % 6]
#     target_axs = sns.heatmap(attn_weights_bands_corr[["ATTN_WEIGHT"]], ax=target_axs, cmap=None, annot=True, cbar=None)
#     target_axs.set_title("{}".format(class_name))
#     idx = idx + 1
#
# fig.suptitle("Crop-Type Correlation Between Attention Weights and Spectral Bands")
# fig.tight_layout()

# components_to_consider=20
# attn_weights_np = np.zeros((len(total_temporal_attention.keys()), components_to_consider))
#
# for idx, parcel_temp_attn in enumerate(total_temporal_attention.values()):
#     parcel_temp_attn = parcel_temp_attn.to_numpy().flatten()
#     total_temp_attn = parcel_temp_attn.sum()
#     sorted_temp_attn_idx = np.argsort(-parcel_temp_attn)
#     highest_temp_attn = parcel_temp_attn[sorted_temp_attn_idx]
#     highest_attn_percentage = highest_temp_attn.cumsum()/total_temp_attn
#     attn_weights_np[idx,:]=highest_attn_percentage[:components_to_consider]
#
# average_temp_attn = attn_weights_np.mean(axis=0) * 100
# plot_data={"Num. Attention Keys":[i for i in range(1, components_to_consider + 1)], "Temporal Attention Percentage": average_temp_attn}
# plot_data = pd.DataFrame(plot_data)
# fig, axs = plt.subplots(figsize=(5, 5))
#
# axs = sns.scatterplot(data=plot_data, ax = axs, x="Num. Attention Keys", y="Temporal Attention Percentage")
# axs.set_xticks([0, 5, 10, 15, 20])
# axs.set_title("Cumulative Temporal Attention")

########### PLOTTING THE ATTENTION WEIGHTS VS SPECTRAL SIGNATURES
# for idx, parcel_data in enumerate(parcels_data):
#     spectral_data_for_parcel = parcel_data[2]
#     key_query_data_for_parcel = parcel_data[3]
#     key_data_for_parcel = key_query_data_for_parcel.loc[key_query_data_for_parcel['Embedding'] == "KEY"]
#     key_data_for_parcel = key_data_for_parcel.set_index(spectral_data_for_parcel.index)
#     spectral_data_for_parcel[["KEY_CLUSTER"]] = key_data_for_parcel[["CLUSTER"]]
#     spectral_cluster_diff = spectral_data_for_parcel.groupby(["KEY_CLUSTER"]).aggregate(np.mean).sort_values(
#         by="TOTAL TEMPORAL ATTENTION")
#     spectral_cluster_diff["CLUSTER_LABEL"] = ["NO_ATTN", "HIGH_ATTN"]
#     spectral_data_for_parcel["CLUSTER_LABEL"] = spectral_data_for_parcel["KEY_CLUSTER"].map(
#         lambda x: spectral_cluster_diff.loc[x]["CLUSTER_LABEL"])
#
#     # plot the difference in attention weights
#     ax_parcel_attn_weights = axs[0][idx]
#     ax_parcel_attn_weights = sns.barplot(data=spectral_data_for_parcel[["CLUSTER_LABEL", "TOTAL TEMPORAL ATTENTION"]],
#                                          x="CLUSTER_LABEL",
#                                          y="TOTAL TEMPORAL ATTENTION",
#                                          ci="sd",
#                                          ax=ax_parcel_attn_weights)
#     ax_parcel_attn_weights.set_title("Attention Weights for {}".format(parcel_data[0]))
#
#     # plot the difference in spectral bands
#     spectral_data_for_parcel = spectral_data_for_parcel.drop(["TOTAL TEMPORAL ATTENTION", "KEY_CLUSTER"], axis=1)
#     spectral_bands_plot_data = pd.melt(spectral_data_for_parcel, id_vars="CLUSTER_LABEL", var_name="BAND",
#                                        value_name="SPECTRAL REFLECTANCE")
#     spectral_bands_plot_data = pd.merge(spectral_bands_plot_data, healthy_vegetation_curve, on="BAND")
#     print(spectral_bands_plot_data)
#     ax_parcel_spectral_signature = axs[1][idx]
#     ax_parcel_spectral_signature.set(ylim=(0, 1))
#
#     ax_parcel_spectral_signature = sns.pointplot(
#         data=spectral_bands_plot_data,
#         x="WAVELENGTH(nm)",
#         y="SPECTRAL REFLECTANCE_x",
#         hue="CLUSTER_LABEL",
#         ci="sd",
#         ax=ax_parcel_spectral_signature)
#     ax_parcel_spectral_signature = sns.pointplot(
#         data=healthy_vegetation_curve,
#         x="WAVELENGTH(nm)",
#         y='SPECTRAL REFLECTANCE',
#         hue="SPECTRAL SIGNATURE",
#         palette="Greens_r",
#         linestyles='--',
#         ci="sd",
#         ax=ax_parcel_spectral_signature)
#
#     ax_parcel_spectral_signature.set_title("Average Cluster Spectral Signature")
#     n = 5  # Keeps every 7th label
#     [l.set_visible(False) for (i, l) in enumerate(ax_parcel_spectral_signature.xaxis.get_ticklabels()) if i % n != 0]
#     if idx > 0:
#         ax_parcel_attn_weights.yaxis.set_visible(False)
#         ax_parcel_spectral_signature.yaxis.set_visible(False)
#
# fig.suptitle("Clusters Difference Analysis", fontsize=16)
# fig.tight_layout()
# attn_weights_corn_all_models = attn_weights_corn_all_models.groupby(["Date", "Model"])["Attention"].mean().reset_index()
# fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7,12))
# attn_weights_corn_no_oclcusion = attn_weights_corn_all_models.loc[attn_weights_corn_all_models["Model"].isin(["all_classes"])]
# axs[0] = sns.lineplot(data=attn_weights_corn_no_oclcusion, x="Date", y="Attention", estimator="mean",ci="sd", ax = axs[0], color=color_mapping["corn"])
# axs[0].set_title("Corn temporal attention - full model", fontweight="bold")
# axs[0].legend(["Attention"], loc="upper left")
# axs[0].set_ylabel('Attention', fontsize=10)
# spectral_indices_corn = test_dataset_spectral_indices.loc[test_dataset_spectral_indices["CLASS"].isin(["corn"])]
#
# ax_ndvi_corn = axs[0].twinx()
# ax_ndvi_corn = sns.lineplot(data=spectral_indices_corn, x="Date", y="NDVI",linestyle="dashed", ci=None, ax=ax_ndvi_corn, color="g")
# ax_ndvi_corn.legend(["NDVI corn"],loc="upper right")
#
#
# attn_weights_corn_winter_wheat_occlusion = attn_weights_corn_all_models.loc[attn_weights_corn_all_models["Model"].isin(["winter_wheat_occlusion"])]
# axs[1] = sns.lineplot(data=attn_weights_corn_winter_wheat_occlusion, x="Date", y="Attention", estimator="mean",ci="sd", ax = axs[1], color=color_mapping["corn"] )
# axs[1].legend(["Attention"], loc="upper left")
# axs[1].set_ylabel('Attention', fontsize=10)
# axs[1].set_title("Corn temporal attention - Winter wheat occlusion", fontweight="bold")
# spectral_indices_winter_wheat = test_dataset_spectral_indices.loc[test_dataset_spectral_indices["CLASS"].isin(["winter wheat"])]
#
# ax_ndvi_winter_wheat = axs[1].twinx()
# ax_ndvi_winter_wheat = sns.lineplot(data=spectral_indices_winter_wheat, x="Date", y="NDVI",linestyle="dashed", ci=None, ax=ax_ndvi_winter_wheat, color="g")
# ax_ndvi_winter_wheat.legend(["NDVI Winter wheat"],loc="upper right")
#
# attn_weights_corn_summer_barley_occlusion = attn_weights_corn_all_models.loc[attn_weights_corn_all_models["Model"].isin(["summer_barley_occlusion"])]
# axs[2] = sns.lineplot(data=attn_weights_corn_summer_barley_occlusion, x="Date", y="Attention", estimator="mean",ci="sd", ax = axs[2], color=color_mapping["corn"])
# axs[2].legend(["Attention"], loc="upper left")
# axs[2].set_ylabel('Attention', fontsize=10)
# axs[2].set_title("Corn temporal attention - Summer barley occlusion", fontweight="bold")
# spectral_indices_summer_barley = test_dataset_spectral_indices.loc[test_dataset_spectral_indices["CLASS"].isin(["summer barley"])]
# ax_ndvi_summer_barley = axs[2].twinx()
# ax_ndvi_summer_barley = sns.lineplot(data=spectral_indices_summer_barley, x="Date", y="NDVI",linestyle="dashed", ci=None, ax=ax_ndvi_summer_barley, color="g")
# ax_ndvi_winter_wheat.legend(["NDVI summer barley"],loc="upper right")
# fig.tight_layout()

# def plot_attn_vs_ndvi(relevant_attn_weights, relevant_ndvi_data, attention_column="Attention"):
#     unique_dates = relevant_attn_weights["Date"].unique()
#     fig, axs = plt.subplots(nrows=2, ncols=len(unique_dates), figsize=(13, 7))
#     for date_idx, date in enumerate(unique_dates):
#         ndvi_for_a_date = relevant_ndvi_data.loc[relevant_ndvi_data["Date"] == date]
#         attn_weights_for_a_date = relevant_attn_weights.loc[relevant_attn_weights["Date"] == date]
#         ax_attention = axs[0][date_idx]
#         ax_ndvi = axs[1][date_idx]
#         hue_order = ndvi_for_a_date["Crop type"].unique()
#         ax_attention = sns.barplot(data=attn_weights_for_a_date, x="Date", y=attention_column, hue="Crop type",
#                                    hue_order=hue_order, ci="sd", palette=color_mapping, ax=ax_attention)
#         ax_ndvi = sns.barplot(data=ndvi_for_a_date, x="Date", y="NDVI", hue="Crop type", ci="sd", hue_order=hue_order,
#                               palette=color_mapping, ax=ax_ndvi)
#         ax_attention.legend().set_visible(False)
#         ax_ndvi.set_ylim([0, 1])
#
#         if (date_idx) > 0:
#             ax_attention.set_ylabel("")
#             ax_ndvi.set_ylabel("")
#
#         if (date_idx) != len(unique_dates) - 1:
#             ax_ndvi.legend().set_visible(False)

