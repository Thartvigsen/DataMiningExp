from expConfig import *
from model import *
from dataset import *
from metric import *
from utils import ConfigReader
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--taskid', type=int, default=0, help='the experiment task to run')
args = parser.parse_args()

# parse parameters
t = args.taskid

c_reader = ConfigReader()

if t == 0:
    # --- Iteration: 1 ---
    config = c_reader.read(t)
    d = SimpleSignal(config)
    m = RNN(config, d.data_setting)
    e = [Accuracy()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5,
    p.run()

if t == 1:
    # --- Iteration: 2 ---
    config = c_reader.read(t)
    d = SimpleSignal(config)
    m = RNN(config, d.data_setting)
    e = [Accuracy()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5,
    p.run()

if t == 2:
    # --- Iteration: 3 ---
    config = c_reader.read(t)
    d = SimpleSignal(config)
    m = RNN(config, d.data_setting)
    e = [Accuracy()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5,
    p.run()

if t == 3:
    # --- Iteration: 4 ---
    config = c_reader.read(t)
    d = SimpleSignal(config)
    m = RNN(config, d.data_setting)
    e = [Accuracy()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5,
    p.run()

if t == 4:
    # --- Iteration: 5 ---
    config = c_reader.read(t)
    d = SimpleSignal(config)
    m = RNN(config, d.data_setting)
    e = [Accuracy()]
    p = ExpConfig(dataset=d,
                  model=m,
                  metric=e,
                  config=config,
                  data_setting=d.data_setting,
                  iteration=t%5,
    p.run()

