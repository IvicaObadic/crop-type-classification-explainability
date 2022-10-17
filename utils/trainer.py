import torch
from utils.classmetric import ClassMetric
from sklearn.metrics import roc_auc_score, auc
from utils.printer import Printer
import sys
import math
import time


import os
import numpy as np
import torch.nn.functional as F
from utils.scheduled_optimizer import ScheduledOptim
import copy

CLASSIFICATION_PHASE_NAME="classification"
EARLINESS_PHASE_NAME="earliness"


class Trainer():
    def __init__(self,
                 model,
                 traindataloader,
                 validdataloader,
                 loss_fn,
                 epochs=100,
                 learning_rate=0.1,
                 store="/tmp",
                 test_every_n_epochs=5,
                 checkpoint_every_n_epochs=20,
                 visdomlogger=None,
                 optimizer=None,
                 show_n_samples=1,
                 overwrite=True,
                 logger=None,
                 **kwargs):

        self.epochs = epochs
        self.batch_size = validdataloader.batch_size
        self.traindataloader = traindataloader
        self.validdataloader = validdataloader
        self.nclasses=traindataloader.dataset.nclasses
        self.store = store
        self.test_every_n_epochs = test_every_n_epochs
        self.logger = logger
        self.show_n_samples = show_n_samples
        self.model = model
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.early_stopping_min_epochs = 10
        self.early_stopping_patience = 3
        self.smooth_period = 3
        self.not_improved_epochs = 0
        self.best_loss = None
        self.loss_fn = loss_fn

        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.classweights = torch.FloatTensor(traindataloader.dataset.classweights)

        if torch.cuda.is_available():
            self.classweights = self.classweights.cuda()

        if visdomlogger is not None:
            self.visdom = visdomlogger
        else:
            self.visdom = None

        # only save checkpoint if not previously resumed from it
        self.resumed_run = False

        self.epoch = 0
        self.start_time = time.time()

        if os.path.exists(self.get_model_name()) and not overwrite:
            print("Resuming from snapshot {}.".format(self.get_model_name()))
            self.resume(self.get_model_name())
            self.resumed_run = True

    def resume(self, filename):
        snapshot = self.model.load(filename)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.epoch = snapshot["epoch"]
        print("resuming optimizer state")
        self.optimizer.load_state_dict(snapshot["optimizer_state_dict"])
        self.logger.resume(snapshot["logged_data"])

    def snapshot(self, model, filename):
        model.save(filename,
                   optimizer_state_dict=self.optimizer.state_dict(),
                   epoch=self.epoch,
                   logged_data=self.logger.get_data())

    def fit(self):
        printer = Printer()

        #Compute the initial results on the validation set
        self.validate_epoch(printer)

        while self.epoch < self.epochs:

            self.new_epoch()  # increments self.epoch

            self.logger.set_mode("train")
            stats = self.train_epoch(self.epoch)
            self.logger.log(stats, self.epoch)
            printer.print(stats, self.epoch, prefix="\n"+self.traindataloader.dataset.partition+": ")

            if self.epoch % self.test_every_n_epochs == 0:
                self.validate_epoch(printer)
                if self.epoch > self.early_stopping_min_epochs and self.check_for_early_stopping():
                    print()
                    print(f"Model did not improve in the last {self.early_stopping_patience} epochs."
                          f"Early termination...")
                    break

            if self.visdom is not None:
                self.visdom.plot_epochs(self.logger.get_data())

            if self.epoch % self.checkpoint_every_n_epochs == 0:
                self.snapshot(self.model, self.get_model_name())
                print("Saving log to {}".format(self.get_log_name()))
                self.logger.get_data().to_csv(self.get_log_name())



        print("Terminating training")
        self.snapshot(self.model, self.get_model_name())
        print("Saving log to {}".format(self.get_log_name()))
        self.logger.get_data().to_csv(self.get_log_name())
        total_training_time = time.time() - self.start_time
        print("Total training time: {}".format(total_training_time))
        return self.logger

    def validate_epoch(self, printer):
        print("Performing validation for epoch {}".format(self.epoch))
        self.logger.set_mode("test")
        stats = self.test_epoch(self.validdataloader)
        self.logger.log(stats, self.epoch)
        printer.print(stats, self.epoch, prefix="\n" + self.validdataloader.dataset.partition + ": ")
        if self.visdom is not None:
            self.visdom_log_test_run(stats)

        validation_loss = stats["loss"]

        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.snapshot(self.model, os.path.join(self.store, "best_model.pth"))


    def check_for_early_stopping(self):
        log = self.logger.get_data()
        log = log.loc[log["mode"] == "test"]

        early_stopping_condition = log["loss"].diff()[-self.smooth_period:].mean() >= 0

        if early_stopping_condition:
            self.not_improved_epochs += 1
            print()
            print(f"model did not improve: {self.not_improved_epochs} of {self.early_stopping_patience} until early stopping...")
            return self.not_improved_epochs >= self.early_stopping_patience
        else:
            self.not_improved_epochs = 0
            return False

    def new_epoch(self):
        self.epoch += 1

    def visdom_log_test_run(self, stats):

        # prevent side effects <- normalization of confusion matrix
        stats = copy.deepcopy(stats)

        if "t_stops" in stats.keys(): self.visdom.plot_boxplot(labels=stats["labels"], t_stops=stats["t_stops"], tmin=0, tmax=self.traindataloader.dataset.samplet)

        # if any prefixed "class_" keys are stored
        if np.array(["class_" in k for k in stats.keys()]).any():
            self.visdom.plot_class_accuracies(stats)

        self.visdom.confusion_matrix(stats["confusion_matrix"], norm=None, title="Confusion Matrix", logscale=None)
        self.visdom.confusion_matrix(stats["confusion_matrix"], norm=0, title="Recall")
        self.visdom.confusion_matrix(stats["confusion_matrix"], norm=1, title="Precision")
        legend = ["class {}".format(c) for c in range(self.nclasses)]
        targets = stats["targets"]
        # either user-specified value or all available values
        n_samples = self.show_n_samples if self.show_n_samples < targets.shape[0] else targets.shape[0]

        for i in range(n_samples):
            classid = targets[i, 0]

            if len(stats["probas"].shape) == 3:
                self.visdom.plot(stats["probas"][:, i, :], name="sample {} P(y) (class={})".format(i, classid),
                                 fillarea=True,
                                 showlegend=True, legend=legend)
            self.visdom.plot(stats["inputs"][i, :, 0], name="sample {} x (class={})".format(i, classid))
            if "pts" in stats.keys(): self.visdom.bar(stats["pts"][i, :], name="sample {} P(t) (class={})".format(i, classid))
            if "deltas" in stats.keys(): self.visdom.bar(stats["deltas"][i, :], name="sample {} deltas (class={})".format(i, classid))
            if "budget" in stats.keys(): self.visdom.bar(stats["budget"][i, :], name="sample {} budget (class={})".format(i, classid))

    def get_model_name(self):
        return os.path.join(self.store, f"model_e{self.epoch}.pth")

    def get_log_name(self):
        return os.path.join(self.store, "log.csv")

    def train_epoch(self, epoch):
        # sets the model to train mode: dropout is applied
        self.model.train()

        # stores the predictions
        training_metrics = ClassMetric()

        for iteration, data in enumerate(self.traindataloader):
            self.optimizer.zero_grad()

            inputs, positions, targets, ids = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                positions = positions.cuda()
                targets = targets.cuda()

            logprobabilities, attn_weights_by_layer = self.model.forward(inputs, positions)

            loss = self.loss_fn(logprobabilities, targets[:, 0])

            loss.backward()
            if isinstance(self.optimizer,ScheduledOptim):
                self.optimizer.step_and_update_lr()
            else:
                self.optimizer.step()

            predictions = self.model.predict(logprobabilities)

            training_metrics.add_batch_stats(
                ids,
                loss.detach().cpu().item(),
                targets.mode(1)[0].detach().cpu(),
                predictions.detach().cpu())

        return training_metrics.calculate_classification_metrics()

    def test_epoch(self, dataloader):
        # sets the model to train mode: no dropout is applied
        self.model.eval()

        # stores the predictions
        test_metrics = ClassMetric()

        with torch.no_grad():
            for iteration, data in enumerate(dataloader):

                inputs, positions, targets, ids = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    positions = positions.cuda()
                    targets = targets.cuda()

                logprobabilities, attn_weights_by_layer = self.model.forward(inputs, positions)

                loss = self.loss_fn(logprobabilities, targets[:, 0])

                prediction = self.model.predict(logprobabilities)

                ## enter numpy world
                predictions = prediction.detach().cpu()
                labels = targets.mode(1)[0].detach().cpu()

                test_metrics.add_batch_stats(ids, loss.detach().cpu(), labels, predictions)

        return test_metrics.calculate_classification_metrics()

