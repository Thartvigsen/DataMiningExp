#from torch.utils.data import Dataset
import os
from torch.utils import data
import numpy as np
import torch
import pandas as pd
from utils import *
import re

class Dataset(data.Dataset):
    """
    This class leans on the PyTorch Dataset class, giving access to easy
    batching and shuffling.

    Methods ending with underscores indicate in-place operations.

    Attributes starting with underscores will not have their values written
    into the log path.

    Example
    -------
    Let's say we want to run a new model on the MNIST dataset. First, we need
    to define a new class in this file called MNIST(). In this new class we
    follow the SimpleSignal() example, creating attribues self.data and
    self.labels along with defining the path to the directory in which we would
    like to store our files (e.g., /home/twhartvigsen/data/MNIST).
    """

    def __init__(self, config):
        self._data_path = "/path/to/my/data/" # Example: /home/twhartvigsen/data/

        # initialize data_setting dictionary to be passed to the model.
        self.data_setting = {}
        self.data_setting["MULTILABEL"] = False
        self.data_setting["VAR_LEN"] = False
        self.data_setting["N_FEATURES"] = 1 # Write over this in new dataset
        self.data_setting["N_CLASSES"] = 2 # Write over this in new dataset

    # --- DATA LOADING -------------------------------------------------------
    def loadData(self, path, regen=False):
        """Check if data exists, if so load it, else create new dataset."""
        if regen: # If forced regeneration
            data, labels = self.generate()
        else:
            try: # Try to load the files
                data = torch.load(path + "data.pt")
                labels = torch.load(path + "labels.pt")
            except: # If no files, generate new files
                print("No files found, creating new dataset")
                makedir(path) # make sure that there's a directory
                data, labels = self.generate()

        # --- save data for next time ---
        arrays = [data, labels]
        tensors = self.arraysToTensors(arrays, "FL")
        data, labels = tensors
        if len(data.shape) < 3:
            data = data.unsqueeze(2)
        outputs = [data, labels]
        names = ["data", "labels"]
        self.saveTensors(outputs, names, self._data_path)
        return data, labels

    # ------------------------------------------------------------------------
    def saveTensors(self, tensors, names, path):
        """Save a list of tensors as .pt files

        Parameters
        ----------
        tensors : list
            A list of pytorch tensors to save
        names : list
            A list of strings, each of which will be used to name
            the data saved with the same index
        """
        for data, name in zip(tensors, names):
            torch.save(data, path+"{}.pt".format(name))

    # ------------------------------------------------------------------------
    def __len__(self):
        """Compute the number of examples in a dataset

        This method is required for use of torch.utils.data.Dataset
        If this method does not apply to a new dataset, rewrite this
        method in the class.
        """
        return len(self.labels)

    # ------------------------------------------------------------------------
    def __getitem__(self, idx):
        """Extract an example from a dataset.

        This method is required for use of torch.utils.data.Dataset

        Parameters
        ----------
        idx : int
            Integer indexing the location of an example.

        Returns
        -------
        X : torch.FloatTensor()
            One example from the dataset.
        y : torch.LongTensor()
            The label associated with the selected example.
        """
        X = self.data[idx]
        y = self.labels[idx]
        return X, y#, idx

    # ------------------------------------------------------------------------
    def toCategorical(self, y, n_classes):
        """1-hot encode a tensor.

        Also known as comnverting a vector to a categorical matrix.

        Paremeters
        ----------
        y : torch.LongTensor()
            1-dimensional vector of integers to be one-hot encoded.
        n_classes : int
            The number of total categories.

        Returns
        -------
        categorical : np.array()
            A one-hot encoded matrix computed from vector y

        """
        categorical = np.eye(n_classes, dtype='uint8')[y]
        return categorical

class SimpleSignal(Dataset):
    def __init__(self, config, dist="uniform"):
        """
        This class provides access to one dataset (in this case called
        "SimpleSignal").

        Key Elements
        ------------
        self.data : torch.tensor (dtype=torch.float)
            Here be data of shape (instances x timesteps x variables)

        self.labels : torch.tensor
            Here be labels of shape (instances x nclasses). For regression,
            nclasses = 1.
        """
        self._n_examples = 500
        self.name = "SimpleSignal"
        # directory in which to store all files relevant to this dataset
        self._load_path = self._data_path + "{}/".format(self.name)
        utils.makedirs(self._load_path)
        super(SimpleSignal, self).__init__(config=config)
        self.data, self.labels = self.loadData(dist)
        self._N_CLASSES = len(np.unique(self.labels))
        self._N_FEATURES = 1
        self.seq_length = 10

        # Log attributes of the dataset to our data_setting dictionary
        self.data_setting["MULTILABEL"] = False
        self.data_setting["VAR_LEN"] = False
        self.data_setting["N_FEATURES"] self._N_FEATURES
        self.data_setting["N_CLASSES"] = self._N_CLASSES

    def generate(self, dist="uniform", pos_signal=1, neg_signal=0):
        """
        The key method in your dataset. Define data and labels here. In this
        example, we create a synthetic dataset from scratch. More commonly,
        this method is used to load data from your dataset folder.
        """
        self.signal_locs = np.random.randint(self.seq_length,
                                             size=int(self._n_examples))
        X = np.zeros((self._n_examples,
                      self.seq_length,
                      self._N_FEATURES))
        y = np.zeros((self._n_examples))

        for i in range(int(self._n_examples)):
            if i < self._n_examples/2:
                X[i, self.signal_locs[i], 0] = pos_signal
                y[i] = 1
            else:
                X[i, self.signal_locs[i], 0] = neg_signal
        data = torch.tensor(np.asarray(X).astype(np.float32), dtype=torch.float)
        labels = torch.tensor(np.array(y).astype(np.int32), dtype=torch.long)
        return data, labels
