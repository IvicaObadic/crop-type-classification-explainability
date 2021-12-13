import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as plticker

show_every_nth_time_point_over100bs = 10
show_every_nth_time_point_below100bs = 6

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
        parcel_attn_weights.index.name = 'Query Obs. Date'
        if len(parcels_data) > 1:
            ax_attn_weights = axs[0][parcel_idx]
            ax_viz_dim = axs[1][parcel_idx]
        else:
            ax_attn_weights = axs[0]
            ax_viz_dim = axs[1]

        ax_attn_weights.set_title("Attention for {}".format(parcel_class))

        number_of_obs = parcel_attn_weights.shape[0]
        if number_of_obs > 100:
            show_every_nth_time_point = show_every_nth_time_point_over100bs
        else:
            show_every_nth_time_point = show_every_nth_time_point_below100bs

        show_color_bar = False
        ax_attn_weights = sns.heatmap(
            parcel_attn_weights,
            cmap=cmaps[parcel_idx],
            ax=ax_attn_weights,
            yticklabels=show_every_nth_time_point,
            xticklabels=show_every_nth_time_point,
            cbar=show_color_bar)
        if parcel_idx > 0:
            ax_attn_weights.set_ylabel('')

        ax_attn_weights.set_xlabel("Key Obs. Date")
        ax_attn_weights.tick_params(bottom=False, top=False, left=False, labelsize=10)

        if viz_dimension == "spectral_reflectance":
            parcel_spectral_index = parcel_data[2].copy()
            parcel_spectral_index = parcel_spectral_index.drop(["TOTAL TEMPORAL ATTENTION"], axis=1)
            corr_matrix = parcel_spectral_index.corr()[["MEAN TEMPORAL ATTENTION"]]
            corr_matrix[["ATTN_WEIGHT_CORR_ABS"]] = corr_matrix[["MEAN TEMPORAL ATTENTION"]].abs()
            parcel_most_correlated_bands = corr_matrix[["ATTN_WEIGHT_CORR_ABS"]].nlargest(6, "ATTN_WEIGHT_CORR_ABS").index.values[:6]
            spectral_index_to_plot = parcel_spectral_index[parcel_most_correlated_bands]
            ax_viz_dim = sns.lineplot(data=spectral_index_to_plot, ax=ax_viz_dim)
            ax_viz_dim.set_title("Attention vs Most Correlated Bands")
            if parcel_idx == 0:
                ax_viz_dim.set_ylabel("Reflectance")

        else:
            key_and_query_data = parcel_data[3]
            ax_viz_dim = sns.scatterplot(
                data=key_and_query_data, x="emb_dim_1", y="emb_dim_2", hue = "CLUSTER", style="TYPE", ax = ax_viz_dim)
            ax_viz_dim.set_title("Keys and Queries Allignment".format(parcel_class))
            ax_viz_dim.axhline(0.0)
            ax_viz_dim.axvline(0.0)

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