import os
import argparse
import pandas as pd
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--perc_important_dates_results_root_dir',
        help='the root folder of the dataset',
        default="/home/results/crop-type-classification-explainability/12_classes/right_padding/obs_acq_date/layers=1,heads=1,emb_dim=128/")

    args, _ = parser.parse_known_args()
    return args


def read_classification_results(classification_results_dir):
    classification_results_path = os.path.join(classification_results_dir, "predictions", "classification_metrics.csv")
    if (os.path.exists(classification_results_path)):
        classification_data = np.genfromtxt(classification_results_path, delimiter=',', skip_header = 1)
        return classification_data[2], classification_data[6]
    return None

def read_confusion_matrix(model_path):
    confusion_matrix_path = os.path.join(model_path, "predictions", "confusion_matrix.csv")
    confusion_matrix = pd.read_csv(confusion_matrix_path)

    confusion_matrix["True class"] = list(confusion_matrix.columns)
    confusion_matrix.set_index("True class", inplace=True)

    confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return confusion_matrix_normalized

def collect_frac_of_dates_accuracy_results(perc_important_dates_results_root_dir):
    accuracy_per_num_dates = []
    for root_perc_result_dir in os.listdir(perc_important_dates_results_root_dir):
        if root_perc_result_dir.endswith("num_dates"):
            num_dates = float(root_perc_result_dir.split("_")[0])
            print(num_dates)
            root_perc_result_path = os.path.join(perc_important_dates_results_root_dir, root_perc_result_dir)
            for root_perc_result_dir_path in os.listdir(root_perc_result_path):
                model_results_dir = os.path.join(root_perc_result_path, root_perc_result_dir_path)
                classification_results = read_classification_results(model_results_dir)
                if classification_results is not None:
                    accuracy_per_num_dates.append((num_dates, classification_results[0], classification_results[1], "Num. dates"))

    accuracy_per_num_dates = pd.DataFrame(data=accuracy_per_num_dates,
                                      columns=["Num. dates", "Class accuracy", "F1 Score","Model type"])
    accuracy_per_num_dates.to_csv(os.path.join(perc_important_dates_results_root_dir, "num_dates_accuracy_results.csv"))
    return accuracy_per_num_dates


# fig, axs = plt.subplots(ncols=1, figsize=(set_size(200)[0], 2.5))
# class_frequency = pd.DataFrame(cm.sum(axis=1))
# print(class_frequency)
# class_frequency_plot = sns.heatmap(class_frequency, annot=True, cbar=None, cmap="Greens", fmt="d", xticklabels=False)
# #class_frequency_plot.set_title("Class Frequency")
# class_frequency_plot.set_xlabel("Count")
# class_frequency_plot.set_ylabel("")
# fig.tight_layout()
# plt.savefig(os.path.join(figures_base_path, 'class_frequency.pdf'), dpi=400)

if __name__ == "__main__":
    args = parse_args()
    collect_frac_of_dates_accuracy_results(args.perc_important_dates_results_root_dir)
