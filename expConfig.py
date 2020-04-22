from utils import *

import torch
import numpy as np
import torch.nn as nn
import math
import torch.optim as optim
import csv
import os
import itertools
import json
import time
import shutil
import gc
from abc import ABCMeta, abstractmethod
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence
from model import SkipCompression
np.random.seed(3)

class ExpConfig():
    """
    Combining data, model, and evaluation metrics.
    Train the specified model on the training data,
    then testing the model on the testing data,
    evaluate results, and write them into logging files
    """

    def __init__(self, d, m, e, config, data_setting, iteration):
        # --- load model pieces ---
        self.dataset = d
        self.model = m
        self.metrics = e
        self.iter = iteration

        # --- unpack hyperparameters ---
        self.BATCH_SIZE = config["training"]["batch_size"]
        self.SCHEDULE_LR = config["training"]["use_scheduler"]
        self._GAMMA = config["training"]["scheduler_param"]
        self.LEARNING_RATE = config["training"]["learning_rate"]
        self.resume = config["training"]["resume"]
        self.optimizer_name = config["training"]["optimizer_name"]
        self._checkpoint = config["training"]["checkpoint"]
        self.N_EPOCHS = config["training"]["n_epochs"]
        self.NUM_WORKERS = config["training"]["num_workers"]
        self.split_props = config["training"]["split_props"]

        # --- CUDA ---
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model.setDevice(device)
        self.model = self.model.to(device)
        self.dataset = self.dataset.to(device)
        # --- dataset setting ---
        self.VAR_LEN = data_setting["VAR_LEN"]
        self.multilabel = data_setting["MULTILABEL"]
        self.seq_length = data_setting["SEQ_LENGTH"]
        self._N_CLASSES = data_setting["N_CLASSES"]

        # --- build directories for logging ---
        self.LOG_PATH = self.setLogPath()
        self.addToPath_(SCHEDULE_LR=self.SCHEDULE_LR,
                        BATCH_SIZE=self.BATCH_SIZE,
                        LEARNING_RATE=self.LEARNING_RATE)
        makedirs(self.LOG_PATH) # Create a directory for saving the results
        print("Writing log file: {}".format(self.LOG_PATH))
        self.saveConfig(config) # Write the current config to that directory
        self.writeFileHeaders() # Add column names to log files (e.g., ["Precision", "Recall"])

        # --- computing the number of trainable parameters ---
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}".format(params))

        # --- retrieve dataloaders ---
        loaders = self.getLoaders(self.dataset)
        self.train_loader = loaders[0]
        self.val_loader = loaders[1]
        self.test_loader = loaders[2]

        # --- resume training ---
        if self.resume:
            self.model = torch.load(self.LOG_PATH + "model.pt")

        # --- set optimizer ---
        self.p_opt = self.getOptimizer(self.model, self.optimizer_name)
        if self.SCHEDULE_LR:
            self.p_sched = optim.lr_scheduler.ExponentialLR(self.p_opt,
                                                            gamma=self._GAMMA)

    def getSplitIndex(self, regen=False):
        """Choosing which examples are used
        for training, development, and testing

        If the indices already exist, load them. Otherwise recreate
        them - a numpy seed has been set already.
        """
        self.N = self.dataset.__len__()
        indices = range(self.N)
        split_points = [int(self.N*i) for i in self.split_props]
        train_ix = np.random.choice(indices,
                                     split_points[0],
                                     replace=False)
        dev_ix = np.random.choice((list(set(indices) - set(train_ix))),
                                   split_points[1],
                                   replace=False)
        test_ix = list(set(indices) - set(train_ix) - set(dev_ix))

        if regen: # Want new indices
            # --- save indices ---
            np.save(self.dataset._load_path + "train_ix.npy", np.array(train_ix))
            np.save(self.dataset._load_path + "dev_ix.npy", np.array(dev_ix))
            np.save(self.dataset._load_path + "test_ix.npy", np.array(test_ix))
        else: # Want to load old indices
            try:
                train_ix = np.load(self.dataset._load_path + "train_ix.npy")
                dev_ix = np.load(self.dataset._load_path + "dev_ix.npy")
                test_ix = np.load(self.dataset._load_path + "test_ix.npy")
            except:
                # --- save indices ---
                np.save(self.dataset._load_path + "train_ix.npy", np.array(train_ix))
                np.save(self.dataset._load_path + "dev_ix.npy", np.array(dev_ix))
                np.save(self.dataset._load_path + "test_ix.npy", np.array(test_ix))
        return train_ix, dev_ix, test_ix

    def getLoaders(self, dataset):
        """define dataloaders"""
        try: # If indices exist in dataset, load them
            self.train_ix = dataset.train_ix
            self.dev_ix = dataset.dev_ix
            self.test_ix = dataset.test_ix
        except: # If not, get random split indices
            self.train_ix, self.dev_ix, self.test_ix = self.getSplitIndex(regen=True)
        train_sampler = SubsetRandomSampler(self.train_ix)
        val_sampler = SubsetRandomSampler(self.val_ix)
        test_sampler = SubsetRandomSampler(self.test_ix)

        train_loader = data.DataLoader(dataset,
                                       batch_size=self.BATCH_SIZE,
                                       sampler=train_sampler,
                                       drop_last=True,
                                       num_workers=self.NUM_WORKERS)
        val_loader = data.DataLoader(dataset,
                                     batch_size=self.BATCH_SIZE,
                                     sampler=val_sampler,
                                     drop_last=True,
                                     num_workers=self.NUM_WORKERS)
        test_loader = data.DataLoader(dataset,
                                      batch_size=self.BATCH_SIZE,
                                      sampler=test_sampler,
                                      drop_last=True,
                                      num_workers=self.NUM_WORKERS)
        return train_loader, val_loader, test_loader

    def getOptimizer(self, model, name):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        # --- get optimizer ---
        if name == "adam":
            optimizer = optim.Adam(trainable_params,
                                   lr=self.LEARNING_RATE,
                                   weight_decay=1e-5)
        elif name == "rmsprop":
            optimizer = optim.RMSprop(trainable_params,
                                      lr=self.LEARNING_RATE,
                                      weight_decay=1e-5)
        else:
            raise NotImplementedError

        return optimizer

    def addToPath_(self, **kwargs):
        """Add an argument to the logging path"""
        for key, val in kwargs.items():
            self.LOG_PATH += "{}-{}/".format(key, val)

    def setLogPath(self):
        """
        Using summaries of different elements of the
        pipeline to create names for logging files

        Returns
        -------
        path : string
            directory in which files for the currect experiment are stored
        """
        path = "./log/DATASET-{}/MODEL-{}".format(self.dataset.name,
                                                  self.model.NAME)
        path = attrToString(self.dataset, path)
        path = attrToString(self.model, path)
        path += "/"
        return path

    def saveConfig(self, config):
        if not os.path.exists(self.LOG_PATH + "config.txt"):
            try:
                with open(self.LOG_PATH + "config.txt", "a+") as file:
                    file.write(json.dumps(config))
            except:
                print("Tried to log config but the file exists already")

    def writeFileHeaders(self):
        if not os.path.exists(self.LOG_PATH+"train_results_{}.csv".format(self.iter)):
            row = ["Loss"]
            for metric in self.metrics: # Add metric names to csv headers
                row.append(metric.name)

            writeCSVRow(row, self.LOG_PATH + "train_results_{}".format(self.iter))
            writeCSVRow(row, self.LOG_PATH + "val_results_{}".format(self.iter))
            writeCSVRow(row, self.LOG_PATH + "test_results_{}".format(self.iter))

    def computeMetrics(self, predictions, labels):# , losses):
        results = []
        for metric in self.metrics:
            m = metric.compute(predictions.copy(), labels.copy())
            results.append(np.round(m, 3))
        return results

    def run(self):
        """Run the training and testing graphs in sequence."""
        for e in range(self.N_EPOCHS):
            # Train model
            start = time.time()
            self.runEpoch(model=self.model,
                          loader=self.train_loader,
                          mode="train",
                          optimizer=self.optimizer,
                          scheduler=self.scheduler)

            # Validate and Test model
            self.runEpoch(self.model, self.val_loader, "val")
            self.runEpoch(self.model, self.test_loader, "test")
            end = time.time()
            print("Epoch {}/{} completed in {} minutes.".format(e+1, self.N_EPOCHS, (end-start)/60.))

    def runEpoch(self, model, loader, mode, optimizer=None, scheduler=None):
        """Given a data loader and model, run through the dataset once."""
        predictions = []
        labels = []
        update_counts = []
        total_loss = 0
        inference_time = 0
        for X, y in loader:
            # Assume: X is of shape (instances x timesteps x variables)
            X = torch.transpose(X, 0, 1) # Sequence first for RNN
            y_hat = model(X) # Give data to model, get prediction
            if optimizer:
                optimizer.zero_grad()
            loss = model.computeLoss()
            total_loss += loss.item()
            if optimizer:
                loss.backward()
                optimizer.step()
            predictions.append(y_hat.detach())
            labels.append(y.detach())
            update_counts.append(update_count.detach())

        if scheduler:
            scheduler.step()
        total_loss = total_loss/len(loader)
        predictions = torch.stack(predictions).squeeze().detach().numpy()
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = torch.stack(labels).squeeze().detach().numpy()
        labels = labels.reshape(-1, 1)

        # ---log results ---
        row = [total_loss]
        metrics = self.computeMetrics(predictions, labels)
        [row.append(metric) for metric in metrics]
        writeCSVRow(row, self.LOG_PATH+"{}_results_{}".format(mode, self.iter), round=True)
