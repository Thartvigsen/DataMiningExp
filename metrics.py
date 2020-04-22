from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from abc import abstractmethod
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_similarity_score

class metric():
    def __init__(self):
        pass

    @abstractmethod
    def compute(self):
        pass

class AUC(metric):
    def __init__(self):
        self.name = "AUC"

    def compute(self, pred, true):
        return roc_auc_score(true, pred, average="micro")

class F1(metric):
    def __init__(self):
        self.name = "F1"

    def compute(self, pred, true):
        return f1_score(true, pred, average="micro")