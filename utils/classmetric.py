import os

import numpy as np
import pandas as pd
from sklearn.metrics import *


class ClassMetric(object):

    def __init__(self):
        self.parcel_ids = []
        self.labels = []
        self.predictions = []
        self.batch_losses = []

    def add_batch_stats(self, batch_ids, batch_loss, batch_labels, batch_predictions):
        self.parcel_ids.extend(batch_ids.tolist())
        self.batch_losses.extend([batch_loss])
        self.labels.extend(batch_labels.tolist())
        self.predictions.extend(batch_predictions.tolist())

    def calculate_classification_metrics(self):

        #convert to numpy arrays
        self.batch_losses = np.array(self.batch_losses)
        self.labels = np.array(self.labels)
        self.predictions = np.array(self.predictions)

        accuracy = accuracy_score(self.labels, self.predictions)
        kappa = cohen_kappa_score(self.labels, self.predictions)
        recall = recall_score(self.labels, self.predictions, average="macro", zero_division=0)
        precision = precision_score(self.labels, self.predictions, average="macro", zero_division=0)
        f1 = f1_score(self.labels, self.predictions, average="macro", zero_division=0)

        cm = self.get_confusion_matrix(normalize="true")

        average_class_accuracy = np.diag(cm).mean()

        return dict(
            loss=self.batch_losses.mean(),
            accuracy=accuracy,
            average_class_accuracy=average_class_accuracy,
            kappa=kappa,
            precision=precision,
            recall=recall,
            f1=f1)

    def get_confusion_matrix(self, normalize=None):
        return confusion_matrix(self.labels, self.predictions, normalize=normalize)

    def get_labels(self):
        return self.labels

    def get_predictions(self):
        return self.predictions

    def save_results(self, target_folder, class_names):

        #save the classification metrics
        classification_metrics = pd.DataFrame([self.calculate_classification_metrics()])
        classification_metrics.to_csv(os.path.join(target_folder, "classification_metrics.csv"), index=False)

        print("Classification metrics:")
        print(classification_metrics)
        #save the true and predicted labels for each parcel
        self.parcel_ids = np.array(self.parcel_ids)
        result = np.column_stack((self.parcel_ids, self.labels, self.predictions))
        np.savetxt(
            os.path.join(target_folder, "predicted_vs_true.csv"),
            result,
            fmt='%i',
            delimiter=",",
            header="PARCEL_ID,LABEL,PREDICTION",
            comments="")

        #save the confusion matrix
        cm = self.get_confusion_matrix()
        class_names = ",".join(class_names.tolist())
        np.savetxt(
            os.path.join(target_folder, "confusion_matrix.csv"),
            cm,
            fmt='%i',
            delimiter=",",
            header=class_names,
            comments="")
