import pandas as pd
import numpy as np

info = pd.read_csv('baseline_xgboost_1_submission.csv')
info = info.values
listing_id = info[:,0]
listing_id = np.array(listing_id,dtype=int)
print listing_id

preds = np.loadtxt('sigma_stack_pred.csv',delimiter=',')
print preds

print '+++++++++++++++++++++++++++++++++++++++++++'
sub = pd.DataFrame(data={'listing_id': listing_id})
# here watch out order
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
print sub
sub.to_csv("submission.csv", index=False, header=True)

print '++++++++++++++++++++++++++++++++++++++++++++'
is_it_lit = pd.read_csv('it_is_lit.csv')
is_it_lit_2 = pd.read_csv('is_it_lit_branden.csv')
print is_it_lit
sub = pd.concat([sub,is_it_lit,is_it_lit_2],axis=0,ignore_index=True)
print sub
sub = pd.DataFrame(sub)
submit = sub.groupby(by='listing_id').mean()
print submit
submit.to_csv("submission.csv", index=True, header=True)
