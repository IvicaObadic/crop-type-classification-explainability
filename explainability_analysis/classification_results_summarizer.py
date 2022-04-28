import os
import argparse


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


def collect_frac_of_dates_accuracy_results(perc_important_dates_results_root_dir):
    perc_dates_results = []
    for root_perc_result_dir in os.listdir(perc_important_dates_results_root_dir):
        if root_perc_result_dir.endswith("frac_of_dates"):
            frac_dates = root_perc_result_dir.split("_")[0]
            root_perc_result_path = os.path.join(perc_important_dates_results_root_dir, root_perc_result_dir)
            for root_perc_result_dir_path in os.listdir(root_perc_result_path):
                model_results_dir = os.path.join(root_perc_result_path, root_perc_result_dir_path, "obs_aq_date",
                                                 "layers=1,heads=1,emb_dim=128")
                classification_results = read_classification_results(model_results_dir)
                if classification_results is not None:
                    perc_dates_results.append((int(frac_dates * 100), classification_results[0], classification_results[1], "Percentage of Dates"))

    perc_dates_results = pd.DataFrame(data=perc_dates_results,
                                      columns=["Percent of observations", "Class accuracy", "F1 Score","Model type"])
    perc_dates_results.to_csv(os.path.join(perc_important_dates_results_root_dir, "perc_accuracy_results.csv"))
    return perc_dates_results

if __name__ == "__main__":
    args = parse_args()
    collect_frac_of_dates_accuracy_results(args.perc_important_dates_results_root_dir)