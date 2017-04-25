import cPickle as pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from matplotlib import pyplot
from scipy import stats


info = np.loadtxt('train_stacknet.csv',delimiter=',')
info = info[:,1:]
label = info[:,0]
label = label.astype(int)
print info.shape
print info

for i in range(150):
    print np.min(info[:,i]),np.mean(info[:,i]),np.max(info[:,i])