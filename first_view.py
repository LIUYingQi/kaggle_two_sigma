import numpy as np
import pandas as pd
import cPickle as pkl
from matplotlib import pyplot
import seaborn

import requests

train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")

f = open('pkl_files/label_1.pkl','r')
a = pkl.load(f)
print a

f = open('pkl_files/label_2.pkl','r')
b = pkl.load(f)
print b