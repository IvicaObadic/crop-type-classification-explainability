import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as plticker


def visualize_attention_heads_in_a_layer(
        attn_weights_dim_1,
        attn_weights_dim_2,
        plot_title,
        left_head_label,
        right_head_label,
        class_labels):

    n_heads = attn_weights_dim_1.shape[0]
    fig, axs = plt.subplots(nrows=n_heads, ncols=2, figsize=(16, 10))

    cmaps = ['Blues', 'Greens', 'Oranges', 'Reds']

    for head_idx in range(n_heads):
        attn_weights_dim_1_h_i = attn_weights_dim_1[head_idx]

        attn_weights_dim_2_h_i = attn_weights_dim_2[head_idx]

        ax_dim_1 = axs[head_idx][0]
        ax_dim_2 = axs[head_idx][1]

        ax_dim_1.set_title("Head {}: {}".format(head_idx + 1, left_head_label))
        ax_dim_2.set_title("Head {}: {}".format(head_idx + 1, right_head_label))

        ax_dim_1.set_xlabel(r'$T^{out}$')
        ax_dim_2.set_ylabel(r'$T^{in}$')

        ax_dim_1.set_xlabel(r'$T^{out}$')
        ax_dim_2.set_ylabel(r'$T^{in}$')

        tick_labels = ['2018-01-01', '2018-01-08', '2018-01-15', '2018-01-22', '2018-01-29',
       '2018-02-05', '2018-02-12', '2018-02-19', '2018-02-26', '2018-03-05',
       '2018-03-12', '2018-03-19', '2018-03-26', '2018-04-02', '2018-04-09',
       '2018-04-16', '2018-04-23', '2018-04-30', '2018-05-07', '2018-05-14',
       '2018-05-21', '2018-05-28', '2018-06-04', '2018-06-11', '2018-06-18',
       '2018-06-25', '2018-07-02', '2018-07-09', '2018-07-16', '2018-07-23',
       '2018-07-30', '2018-08-06', '2018-08-13', '2018-08-20', '2018-08-27',
       '2018-09-03', '2018-09-10', '2018-09-17', '2018-09-24', '2018-10-01',
       '2018-10-08', '2018-10-15', '2018-10-22', '2018-10-29', '2018-11-05',
       '2018-11-12', '2018-11-19', '2018-11-26', '2018-12-03', '2018-12-10',
       '2018-12-17', '2018-12-24']

        ax_dim_1 = sns.heatmap(
            attn_weights_dim_1_h_i,
            cmap=cmaps[head_idx],
            ax=ax_dim_1,
            vmin=0,
            vmax=1,
            xticklabels=class_labels,
            yticklabels=False)
        n = 4  # Keeps every 7th label
        #'[l.set_visible(False) for (i, l) in enumerate(ax_dim_1.xaxis.get_ticklabels()) if i % n != 0]
        ax_dim_1.tick_params(bottom=False, top=False, left=False, labelsize=8)
        ax_dim_2 = sns.heatmap(attn_weights_dim_2_h_i,
                               cmap=cmaps[head_idx],
                               ax=ax_dim_2,
                               vmin=0,
                               vmax=1,
                               xticklabels=class_labels,
                               yticklabels=False)
        #[l.set_visible(False) for (i, l) in enumerate(ax_dim_2.xaxis.get_ticklabels()) if i % n != 0]
        ax_dim_2.tick_params(bottom=False, top=False, left=False, labelsize=8)


    fig.suptitle(plot_title, fontsize=12)
    fig.tight_layout()


def visualize_attention_by_layer(
        target_layer,
        attn_weights_dim_1,
        attn_weights_dim_2,
        plot_title,
        left_head_label,
        right_head_label,
        class_labels):
    attn_weights_dim_1_target_layer = attn_weights_dim_1[target_layer]
    attn_weights_dim_2_target_layer = attn_weights_dim_2[target_layer]

    visualize_attention_heads_in_a_layer(
        attn_weights_dim_1_target_layer,
        attn_weights_dim_2_target_layer,
        plot_title,
        left_head_label,
        right_head_label,
        class_labels)


def stacked_boxplot(input_data, title, xlabel, ylabel):
    ax = sns.boxplot(x="layer", y="correlation", hue="head", data=input_data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

    plt.clf()
    plt.close()