import pandas as pd
import numpy as np

info = np.loadtxt('my_test_stacknet.csv',delimiter=',')
listing_id = info[:,0]
listing_id = np.array(listing_id,dtype=int)
print listing_id

preds = np.loadtxt('sigma_xgboost2_stack_pred.csv',delimiter=',')
print preds

print '+++++++++++++++++++++++++++++++++++++++++++'
sub = pd.DataFrame(data={'listing_id': listing_id})
# here watch out order
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
print sub
sub.to_csv("baseline_xgboost_1_submission.csv", index=False, header=True)

# print '++++++++++++++++++++++++++++++++++++++++++++'
# is_it_lit = pd.read_csv('baseline_xgboost_1.csv')
# print is_it_lit
# sub = pd.concat([sub,is_it_lit],axis=0,ignore_index=True)
# print sub
# sub = pd.DataFrame(sub)
# submit = sub.groupby(by='listing_id').mean()
# print submit
# submit.to_csv("baseline_xgboost_1_submission.csv", index=True, header=True)
