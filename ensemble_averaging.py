import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
results = pd.read_csv('results.txt')
print results
baseline = results['baseline'].values
score = results['score'].values

score = score[:2]
weight = 1/score**3
print weight
weight = weight/sum(weight)
print weight

f_baseline_xgboost_1 = baseline[0]+'.csv'
f_baseline_xgboost_2 = baseline[1]+'.csv'
# f_baseline_RF_1 = baseline[2]+'.csv'
# f_baseline_RF_2 = baseline[3]+'.csv'

print f_baseline_xgboost_1
print f_baseline_xgboost_2
# print f_baseline_RF_1
# print f_baseline_RF_2

xgboost_1 = pd.read_csv('results/'+f_baseline_xgboost_1)
listing_id = xgboost_1['listing_id'].values
xgboost_1['low'] = xgboost_1['low']*weight[0]
xgboost_1['medium'] = xgboost_1['medium']*weight[0]
xgboost_1['high'] = xgboost_1['high']*weight[0]
xgboost_1 = xgboost_1.loc[:,['low','medium','high']].values

xgboost_2 = pd.read_csv('results/'+f_baseline_xgboost_2)
xgboost_2['low'] = xgboost_2['low']*weight[1]
xgboost_2['medium'] = xgboost_2['medium']*weight[1]
xgboost_2['high'] = xgboost_2['high']*weight[1]
xgboost_2 = xgboost_2.loc[:,['low','medium','high']].values

# RF_1 = pd.read_csv('results/'+f_baseline_RF_1)
# RF_1['low'] = RF_1['low']*weight[2]
# RF_1['medium'] = RF_1['medium']*weight[2]
# RF_1['high'] = RF_1['high']*weight[2]
# RF_1 = RF_1.loc[:,['low','medium','high']].values
#
# RF_2 = pd.read_csv('results/'+f_baseline_RF_2)
# RF_2['low'] = RF_2['low']*weight[3]
# RF_2['medium'] = RF_2['medium']*weight[3]
# RF_2['high'] = RF_2['high']*weight[3]
# RF_2 = RF_2.loc[:,['low','medium','high']].values

print xgboost_1
print xgboost_2
# print RF_1
# print RF_2
#
# res = xgboost_1+xgboost_2
# print res
# sub = pd.DataFrame(data={'listing_id': listing_id})
# sub['low'] = res[:, 0]
# sub['medium'] = res[:, 1]
# sub['high'] = res[:, 2]
# sub.to_csv("submission1.csv", index=False, header=True)

a = np.max(xgboost_1 - xgboost_2,1)
a.sort()
plt.figure()
plt.plot(a)
plt.show()