import numpy as np
import pandas as pd
import cPickle as pkl
from matplotlib import pyplot
import seaborn
train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")
print train_df
train_df = train_df.sort_index()
print train_df.index
print train_df
print train_df["listing_id"]
# print train_df.columns

# a = train_df["latitude"].values
# b = train_df["longitude"].values
# pyplot.figure()
# pyplot.scatter(a,b)
# pyplot.show()

#
# print test_df.ix[test_df["latitude"]==0]
# print test_df.ix[test_df["longitude"]==0]
# f = open("res_ha.pkl",'r')
# info = pkl.load(f)
# info = info.values
# print info.shape
# for i in range(2000):
#     print str(i+1)+ "   " +str(info[i,0])