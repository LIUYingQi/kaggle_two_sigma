
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np

print "fitting..."
train_df_f = open("pkl_files/train_df_2.pkl","rb")
train_df = pickle.load(train_df_f)
test_df_f = open("pkl_files/test_df_2.pkl",'rb')
test_df = pickle.load(test_df_f)
y_f = open("pkl_files/label_2.pkl",'rb')
y = pickle.load(y_f)
train_df_f.close()
test_df_f.close()
y_f.close()
print train_df.describe()
print test_df.describe()
print len(y)

# CV
###################################################################3

####################################################################

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=400)
clf.fit(X_train,y_train)
preds = clf.predict_proba(X_test)
res = log_loss(y_test,preds)
print res

print("Fitted")
preds = clf.predict_proba(test_df)
print preds
sub = pd.DataFrame(data={'listing_id': test_df['listing_id'].ravel()})
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
sub.to_csv("results/baseline_RF_2.csv", index=False, header=True)
