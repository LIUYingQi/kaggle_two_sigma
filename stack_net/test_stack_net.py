import pandas as pd
import numpy as np

info = np.loadtxt('test_stacknet.csv',delimiter=',')
listing_id = info[:,0]
listing_id = np.array(listing_id,dtype=int)
print listing_id

preds = np.loadtxt('sigma_stack_pred.csv',delimiter=',')
print preds

print '+++++++++++++++++++++++++++++++++++++++++++'
sub = pd.DataFrame(data={'listing_id': listing_id})
sub['high'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['low'] = preds[:, 2]
# sub.to_csv("submission.csv", index=False, header=True)

print '++++++++++++++++++++++++++++++++++++++++++++'
is_it_lit = pd.read_csv('baseline_xgboost_2.csv')
sub = pd.merge()