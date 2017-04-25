import pandas as pd
import cPickle as pkl
import numpy as np

f = open('pkl_files/label_1.pkl','rb')
label1 = pkl.load(f)
f.close()

f = open('pkl_files/label_2.pkl','rb')
label2 = pkl.load(f)
f.close()

print label1.dtype
print label2.dtype

print np.mean(label1 - label2)