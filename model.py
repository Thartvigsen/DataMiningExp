import torch
from torch import nn
from modules import *
import utils
import numpy as np

class Model(nn.Module):
    def __init__(self, config, data_setting):
        super(Model, self).__init__()

        # --- Model hyperparameters ---
        self._BATCH_SIZE = config["training"]["batch_size"]
        self._LOSS_NAME = config["training"]["loss_name"]
        self._N_EPOCHS = config["training"]["n_epochs"]
        self.N_LAYERS = config["model"]["n_layers"]
        self._DROPOUT_PROB = config["model"]["dropout_probability"]
        self._ORTHO_INIT = config["model"]["ortho_init"]
        self._EMBED_DIM = config["model"]["embed_dim"]
        self.HIDDEN_DIM = config["model"]["hidden_dim"]
        self._NOISIN = config["model"]["noisin"]
        self._CELL_TYPE = config["model"]["cell_type"]

        # --- data setting ---
        self._MULTILABEL = data_setting["MULTILABEL"]
        self._N_FEATURES = data_setting["N_FEATURES"]
        self._N_CLASSES = data_setting["N_CLASSES"]
        self._VAR_LEN = data_setting["VAR_LEN"]
        self._outrow = []

        # --- define the projection nonlinearity ---
        # self.out_nonlin is supposed to make it easier to run a collection of
        # both regression tasks and classification tasks concurrently since
        # they each require their own final projection method that can be
        # defined along with the dataset.
        if self._MULTILABEL:
            self.out_nonlin = nn.Sigmoid()
        else:
            self.out_nonlin = nn.Softmax(dim=1)

    def setDevice(self, device):
        self.device = device

    def initHidden(self, bsz):
        """Initialize hidden states"""
        if self._CELL_TYPE == "LSTM":
            h = (torch.zeros(self.N_LAYERS,
                             bsz,
                             self.HIDDEN_DIM),
                 torch.zeros(self.N_LAYERS,
                             bsz,
                             self.HIDDEN_DIM))
        else:
            h = torch.zeros(self.N_LAYERS,
                            bsz,
                            self.HIDDEN_DIM)
        return h

    def getCriterion(self, name):
        """PyTorch implementations of Loss functions"""
        if name == "mse":
            criterion = torch.nn.MSELoss(size_average=True, reduction="mean")
        elif name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=True,
                                             ignore_index=-100, reduction="mean")
        elif name == "bce":
            criterion = torch.nn.BCELoss(weight=None, size_average=True,
                                    reduction="mean")
        else:
            raise NotImplementedError
        return criterion

    def computeLoss(self, pred, label):
        """
        Basic loss computation, this can be written over in individual
        models to use custom loss functions.
        """
        criterion = self.getCriterion(self._LOSS_NAME)
        loss = criterion.forward(pred, label)
        return loss

    def initHidden(self, bsz):
        """Initialize hidden weights."""
        if self.CELL_TYPE == "LSTM":
            h = (torch.zeros(self.N_LAYERS,
                             bsz,
                             self.HIDDEN_DIM),
                 torch.zeros(self.N_LAYERS,
                             self._BATCH_SIZE,
                             self.HIDDEN_DIM))
        else:
            h = torch.zeros(self.N_LAYERS,
                            bsz,
                            self.HIDDEN_DIM)
        if self._CUDA:
            h = h.cuda()
        return h

    def modifyGradients(self):
        """
        Sometimes you may want to change the raw gradient values.
        In this framework, we assume that you ALWAYS want to change them, but
        by default this modification does nothing. Redefine this method in your
        model if you actually want to modify your gradients.
        """
        pass

class RNN(Model):
    def __init__(self, config, data_setting):
        """
        Example model implementation.

        Parameters
        ----------
        config : dict
            A dictionary that contains the experimental configuration (number
            of dimensions in the hidden state, batch size, etc.). This
            dictionary gets passed into the parent Model() and initialized.

        data_setting : dict
            A dictionairy containing relevant settings from the dataset (number
            of classes to predict, whether or not the time series are of
            variable-length, how many variables the data have, etc.)
        """
        super(RNN, self).__init__(config=config,
                                  data_setting=data_setting)
        self.NAME = "RNN" # Name of the model for creating the log files

        # --- Mappings ---
        self.RNNCell = nn.GRU(self._N_FEATURES, self.HIDDEN_DIM, self.N_LAYERS)
        self.predict = nn.Linear(self.HIDDEN_DIM, self._N_CLASSES)

        # Example: using a sub-network by loading a module. Delete this if you
        # don't want to use modules!
        self.MyModule = SampleModule(self._N_FEATURES, self._N_CLASSES)
    
    def forward(self, X):
        """
        The main method of your model, completely mapping the input data to the
        output predictions. In this example, the RNN outputs a classification
        using only the final hidden state (out[-1]).
        """
        T, B, V = X.shape # Assume timesteps x batch x variables input
        hidden = self.initHidden(B)
        out, hidden = self.RNNCell(X, hidden)
        y_hat = self.out_nonlin(self.predict(out[-1]).squeeze())
        return y_hat
