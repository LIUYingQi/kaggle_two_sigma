import numpy as np
import pandas as pd
import cPickle as pkl
from matplotlib import pyplot
import seaborn
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import cPickle as pickle

pres1 = np.loadtxt('stacking_results/baseline_xgboost_1.csv')
print pres1
pres2 = np.loadtxt('stacking_results/baseline_xgboost_2.csv')
print pres2
pres3 = np.loadtxt('stacking_results/baseline_RF_1.csv')
print pres3
pres4 = np.loadtxt('stacking_results/baseline_RF_2.csv')
print pres4


f = open('pkl_files/label_1.pkl','r')
label = pkl.load(f)
print label

train_set = np.hstack((pres1,pres2,pres3,pres4))
print train_set

print log_loss(label,pres1)
print log_loss(label,pres2)
print log_loss(label,pres3)
print log_loss(label,pres4)

test_df_f = open("pkl_files/test_df_1.pkl",'rb')
test_df = pickle.load(test_df_f)
test_df_f.close()
print len(test_df)

clf = MLPClassifier(alpha=0.5)
clf.fit(train_set,label)



spliter = StratifiedKFold(5)
skf = spliter.split(train_set,label)
for i, (train, test) in enumerate(skf):
    print i
    clf.fit(train_set[train],label[train])
    ans = clf.predict_proba(train_set[test])
    print log_loss(label[test],ans)





#################
###  problem is that info leakage in train / sub set
### so that here give u a big leason     only when LB and CV are coherent proves u r right





# res1 = pd.read_csv('results/baseline_xgboost_1.csv')
# pres1 = np.zeros((len(test_df),3),dtype=float)
# pres1[:,0] = res1['low'].values
# pres1[:,1] = res1['medium'].values
# pres1[:,2] = res1['high'].values
#
# res2 = pd.read_csv('results/baseline_xgboost_2.csv')
# pres2 = np.zeros((len(test_df),3),dtype=float)
# pres2[:,0] = res2['low'].values
# pres2[:,1] = res2['medium'].values
# pres2[:,2] = res2['high'].values
#
# res3 = pd.read_csv('results/baseline_RF_1.csv')
# pres3 = np.zeros((len(test_df),3),dtype=float)
# pres3[:,0] = res3['low'].values
# pres3[:,1] = res3['medium'].values
# pres3[:,2] = res3['high'].values
#
# res4 = pd.read_csv('results/baseline_RF_2.csv')
# pres4 = np.zeros((len(test_df),3),dtype=float)
# pres4[:,0] = res4['low'].values
# pres4[:,1] = res4['medium'].values
# pres4[:,2] = res4['high'].values
#
# submission_set = np.hstack((pres1,pres2,pres3,pres4))
# print submission_set
#
#
# preds = clf.predict_proba(submission_set)
# print preds
# sub = pd.DataFrame(data={'listing_id': test_df['listing_id'].ravel()})
# sub['low'] = preds[:, 0]
# sub['medium'] = preds[:, 1]
# sub['high'] = preds[:, 2]
# sub.to_csv("baseline_xgboost_1_submission.csv", index=False, header=True)
