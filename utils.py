import torch
import numpy as np
import csv
import os.path
import os

class MainWriter(object):
    def __init__(self):
        self.header = ("from expConfig import *\n"
                       "from model import *\n"
                       "from dataset import *\n"
                       "from metric import *\n"
                       "from utils import ConfigReader\n"
                       "import argparse\n\n"
        
                       "parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)\n"
                       "parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')\n"
                       "args = parser.parse_args()\n\n"
        
                       "# parse parameters\n"
                       "t = args.taskid\n\n"
        
                       "c_reader = ConfigReader()\n\n")
        
        # --- set experimental configs ---
        self.datasets = [
            """SimpleSignal(config)""",
        ]

        self.metrics = """[Accuracy()]"""

        self.models = [
            """RNN(config, d.data_setting)""",
        ]

        self.n_iterations = 5

    def write(self):
        t = 0
        for d in self.datasets:
            for model in self.models:
                for i in range(self.n_iterations):
                    text = ("if t == {0}:\n"
                            "    # --- Iteration: {1} ---\n"
                            "    config = c_reader.read(t)\n"
                            "    d = {2}\n"
                            "    m = {3}\n"
                            "    e = {4}\n"
                            "    p = ExpConfig(dataset=d,\n"
                            "                  model=m,\n"
                            "                  metric=e,\n"
                            "                  config=config,\n"
                            "                  data_setting=d.data_setting,\n"
                            "                  iteration=t%{5},\n"
                            "    p.run()\n\n".format(t,
                                                     i+1,
                                                     d,
                                                     model,
                                                     self.metrics,
                                                     self.n_iterations))
                    self.header += text
                    t += 1

        with open("main.py", "w") as f:
            f.write(self.header)
            f.close()

def attrToString(obj, prefix,
                exclude_list=["NAME", "name", "desc", "training", "bsz",
                              "data", "labels", "signal_locs", "round",
                              "train", "test", "train_labels", "test_labels",
                              "data_setting", "y_train", "y_test", "seq_length"]):
    """Convert the attributes of an object into a unique string of
    path for result log and model checkpoint saving. The private
    attributes (starting with '_', e.g., '_attr') and the attributes
    in the `exclude_list` will be excluded from the string.
    Args:
        obj: the object to extract the attribute values prefix: the
        prefix of the string (e.g., MODEL, DATASET) exclude_list:
        the list of attributes to be exclude/ignored in the
        string Returns: a unique string of path with the
        attribute-value pairs of the input object
    """
    out_dir = prefix #+"-"#+obj.name
    for k, v in obj.__dict__.items():
        if not k.startswith('_') and k not in exclude_list:
            out_dir += "/{}-{}".format(k, ",".join([str(i) for i in v]) if type(v) == list else v)
    return out_dir

def writeCSVRow(row, name, path="./", round=False):
    """
    Given a row, rely on the filename variable to write
    a new row of experimental results into a log file

    New Idea: Write a header for the csv so that I can
    clearly understand what is going on in the file

    Parameters
    ----------
    row : list
        A list of variables to be logged
    name : str
        The name of the file
    path : str
        The location to store the file
    """

    if round:
        row = [np.round(i, 2) for i in row]
    f = path + name + ".csv"
    with open(f, "a+") as csvfile:
        filewriter = csv.writer(csvfile, delimiter=",")
        filewriter.writerow(row)

def exponentialDecay(N):
    tau = 1
    #tau = 1.5
    #tmax = 10
    #tmax = 2
    tmax = 7
    t = np.linspace(0, tmax, N)
    y = torch.tensor(np.exp(-t/tau), dtype=torch.float)
    return y#/5.

class ConfigReader(object):
    def __init__(self):
        self.path = "./configs/"

    def read(self, t):
        """Read config file t as a dictionary.
        If the requested file does not exist, load the base
        configuration file instead.
        """
        if os.path.isfile(self.path+"input_{}.txt".format(t)):
            s = open(self.path+"input_{}.txt".format(t), "r").read()
        else:
            print("Loading base config file.")
            s = open(self.path+"base_config.txt", "r").read()
        return eval(s)

def hardSigma(a, x):
    temp = torch.div(torch.add(torch.mul(x, a), 1), 2.0)
    output = torch.clamp(temp, min=0, max=1)
    return output

def printParams(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def makedir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except:
        pass
