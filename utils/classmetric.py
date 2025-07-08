import os

import numpy as np
import pandas as pd
import json
from sklearn.metrics import *


class ClassMetric(object):

    def __init__(self):
        self.parcel_ids = []
        self.labels = []
        self.predictions = []
        self.batch_losses = []

        self.ndvi_r2_score = 0.0
        self.ndvi_rmse_score = 0.0

    def add_batch_stats(self, batch_ids, batch_loss, batch_labels, batch_predictions):
        # self.parcel_ids.extend(batch_ids.tolist())
        # if isinstance(batch_ids, tuple):
        self.parcel_ids.extend(list(batch_ids))
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


class NDVIMetric(object):

    def __init__(self):
        self.parcel_ids = []
        self.labels = []
        self.ndvi_labels = []
        self.ndvi_preds = []

        self.ndvi_r2_score = 0.0
        self.ndvi_rmse_score = 0.0

    def add_batch_stats(self, batch_ids, batch_labels, batch_ndvi_labels, batch_ndvi_preds):
        # self.parcel_ids.extend(batch_ids.tolist())
        # if isinstance(batch_ids, tuple):
        self.parcel_ids.extend(list(batch_ids))
        self.labels.extend(batch_labels.tolist())
        self.ndvi_labels.extend(batch_ndvi_labels.tolist())
        self.ndvi_preds.extend(batch_ndvi_preds.tolist())

    def calculate_ndvi_metrics(self, class_names):

        #convert to numpy arrays
        self.labels = np.array(self.labels)
        self.ndvi_labels = np.array(self.ndvi_labels)
        self.ndvi_preds = np.array(self.ndvi_preds)

        ndvi_r2_score = r2_score(self.ndvi_labels, self.ndvi_preds)
        ndvi_rmse_score = root_mean_squared_error(self.ndvi_labels, self.ndvi_preds)
        dict_1 = dict(ndvi_rmse_score=ndvi_rmse_score, ndvi_r2_score=ndvi_r2_score)

        # get r2 per class
        crop_ndvi = {crop: [] for crop in class_names}
        crop_prediction = {crop: [] for crop in class_names}
        r2_per_class = {crop: [] for crop in class_names}

        for i,label in enumerate(self.labels):
            crop_ndvi[class_names[label]].append(self.ndvi_labels[i])
            crop_prediction[class_names[label]].append(self.ndvi_preds[i])
        
        for crop in class_names:
            if len(crop_ndvi[crop]) > 0:
                r2_crop = r2_score(crop_ndvi[crop], crop_prediction[crop])
                r2_per_class[crop] = crop=r2_crop
            else: 
                r2_per_class[crop] = np.nan

        return dict(tuple(dict_1.items()) + tuple(r2_per_class.items()))  

    def get_labels(self):
        return self.labels  
        
    def get_ndvi_labels(self):
        return self.ndvi_labels

    def get_ndvi_predictions(self):
        return self.ndvi_preds

    def save_results(self, target_folder, class_names):

        #save the classification metrics
        ndvi_metrics = pd.DataFrame([self.calculate_ndvi_metrics(class_names)])
        ndvi_metrics.to_csv(os.path.join(target_folder, "ndvi_metrics.csv"), index=False)

        print("NDVI Decoder metrics:")
        print(ndvi_metrics)
        #save the true and predicted labels for each parcel
        result = {}
        self.parcel_ids = np.array(self.parcel_ids)
        for i, parcel_id in enumerate(self.parcel_ids):
            result[str(parcel_id)] =  [class_names[self.labels[i]], self.ndvi_labels[i].tolist(), self.ndvi_preds[i].tolist()]

        json.dump(result, open(os.path.join(target_folder, "ndvi_predictions.json"), "w"))