from abc import ABCMeta, abstractmethod
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import hamming_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_similarity_score
import numpy as np
import torch

class metric():
    def __init__(self):
        self.values = []

    @abstractmethod
    def compute(self, pred, true):
        # Catch metrics failing due to particular prediction forms
        try:
            return self.metric(pred, true)
        except:
             return np.zeros_like(pred)

class Accuracy(metric):
    def __init__(self):
        self.name = "Accuracy"

    def compute(self, pred, true):
        if len(pred.shape) > 1:
            pred = np.argmax(pred, 1)
        return accuracy_score(true, pred)

class MLLAUC(metric):
    def __init__(self):
        self.name = "AUC"

    def compute(self, pred, true):
        m = []
        for i in range(len(pred)):
            print(true[i])
            print(pred[i])
            m.append(roc_auc_score(true[i], pred[i], average=None))
        return np.mean(m)

class Subset01Loss(metric):
    def __init__(self):
        self.name = "Subset01Loss"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        mistakes = 0
        for i in range(len(pred)):
            mistakes += (dummy_pred[i] != true[i]).any()
        return mistakes/len(pred)

class AUC_instance(metric):
    def __init__(self):
        self.name = "AUC_instance"

    def compute(self, pred, true):
        auc = 0
        for i in range(len(pred)):
            try:
                auc += roc_auc_score(true[i], pred[i], average="samples")
            except:
                auc += 1 # If label is [0, 0] or [1, 1] any threshold works...
        auc /= len(pred)
        return auc
#        for i in range(len(pred)):
#            print(pred[i])
#            print(true[i])
#            print(roc_auc_score(true[i], pred[i], average="samples"))
#        print(true[:5].shape)
#        print(pred[:5].shape)
        #return roc_auc_score(true, pred, average="samples")

class RankingLoss(metric):
    def __init__(self):
        self.name = "RankingLoss"

    def pairwise_sub(self, first_tensor, second_tensor):
        column = first_tensor.unsqueeze(2)
        row = second_tensor.unsqueeze(1)
        return column - row

    def pairwise_and(self, first_tensor, second_tensor):
        column = first_tensor.unsqueeze(2)
        row = second_tensor.unsqueeze(1)
        return column & row                                                                                                               

    def compute(self, y_hat, y):
        y_hat = torch.tensor(y_hat)
        y = torch.tensor(y)

        shape = y.shape
        y_in = torch.eq(y.float(), torch.ones(shape))
        y_out = torch.ne(y.float(), torch.ones(shape))

        # get indices to check   
        truth_matrix = self.pairwise_and(y_in, y_out).float()

        # calculate all exp'd differences
        sub_matrix = self.pairwise_sub(y_hat, y_hat)
        exp_matrix = torch.exp(-sub_matrix)

        # check which differences to consider and sum them
        sparse_matrix = exp_matrix * truth_matrix
        sums = 1 + torch.sum(sparse_matrix, dim=(1, 2))
        
        # get normalizing terms and apply them
#        normalizers = getNormalizers(y_in, y_out)
#        results = torch.log(sums)/normalizers
#        return results.mean(0) # Mean over the batch (?)

        return torch.log(sums).mean().numpy() # Average over batch

class AUC_macro(metric):
    def __init__(self):
        self.name = "AUC_macro"
        self.metric = roc_auc_score

    def compute(self, pred, true):
        try:
            return roc_auc_score(true, pred, average="macro")
        except:
            return np.zeros_like(pred)

class AUC_micro(metric):
    def __init__(self):
        self.name = "AUC_micro"

    def compute(self, pred, true):
        return roc_auc_score(true, pred, average="micro")

class F1_macro(metric):
    def __init__(self):
        self.name = "F1_macro"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        return f1_score(true, dummy_pred, average="macro")

class F1_micro(metric):
    def __init__(self):
        self.name = "F1_micro"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        return f1_score(true, dummy_pred, average="micro")

class Precision_micro(metric):
    def __init__(self):
        self.name = "Precision_micro"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        return precision_score(true, dummy_pred, average="micro")

class Precision_macro(metric):
    def __init__(self):
        self.name = "Precision_macro"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        return precision_score(true, dummy_pred, average="macro")

class Recall_micro(metric):
    def __init__(self):
        self.name = "Recall_micro"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        score = recall_score(true, dummy_pred, average="micro")
        return score

class Recall_macro(metric):
    def __init__(self):
        self.name = "Recall_macro"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        score = recall_score(true, dummy_pred, average="macro")
        return score

class Jaccard(metric):
    def __init__(self):
        self.name = "Jaccard"

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        return jaccard_similarity_score(true, dummy_pred)

class HammingLoss(metric):
    def __init__(self):
        self.name = "HammingLoss"

#    def compute(self, pred, true):
#        pred[pred >= threshold] = 1
#        pred[pred < threshold] = 0
#        return (pred == true).sum()/pred.size()

    def compute(self, pred, true, threshold=0.5):
        dummy_pred = np.zeros_like(pred)
        dummy_pred[pred >= threshold] = 1
        return hamming_loss(true, dummy_pred)

class AveragePrecision_macro(metric):
    def __init__(self):
        self.name = "AveragePrecision_macro"

    def compute(self, pred, true):
        return average_precision_score(true, pred, average="macro")

class AveragePrecision_micro(metric):
    def __init__(self):
        self.name = "AveragePrecision_micro"

    def compute(self, pred, true):
        return average_precision_score(true, pred, average="micro")

class HingeLoss(metric):
    def __init__(self):
        self.name = "HingeLoss"

    def compute(self, pred, true):
        return hinge_loss(true, pred)

class Custom(metric):
    def __init__(self):
        self.name = "custom"

    def compute(self, pred, true):
        pred = np.array(pred)
        true = np.array(true)
        diff = 1. - np.abs(pred - true)
        acc = np.sum(diff, axis=1)/true.shape[1]
        acc = np.mean(acc, axis=0)
        return acc

class MAP(metric):
    def __init__(self):
        self.name = "mAP"

    def compute(self, pred, true):
        return np.mean(average_precision_score(true, pred, average=None))
