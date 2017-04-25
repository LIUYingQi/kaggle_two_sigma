
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

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




image_date = pd.read_csv("input/listing_image_time.csv")

# rename columns so you can join tables later on
image_date.columns = ["listing_id", "time_stamp"]

# reassign the only one timestamp from April, all others from Oct/Nov
image_date.loc[80240,"time_stamp"] = 1478129766

image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
image_date["img_date_month"]            = image_date["img_date"].dt.month
image_date["img_date_week"]             = image_date["img_date"].dt.week
image_date["img_date_day"]              = image_date["img_date"].dt.day
image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
image_date["img_date_hour"]             = image_date["img_date"].dt.hour
image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

train_df = pd.merge(train_df, image_date, on="listing_id", how="left")
test_df = pd.merge(test_df, image_date, on="listing_id", how="left")

del train_df['img_date']
del test_df['img_date']


print train_df.describe()
print test_df.describe()
print len(y)

# CV
###################################################################3

####################################################################

X_train, X_test, y_train, y_test = train_test_split(train_df, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=400,max_depth=15)
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
sub.to_csv("baseline_RF_2_submission.csv", index=False, header=True)


# generating stacking file
####################################################################
train_df = train_df.values
test_df = test_df.values

clf = RandomForestClassifier(n_estimators=400)

stacking_values = np.zeros((len(y),3),dtype=float)

spliter = StratifiedKFold()
skf = spliter.split(train_df,y)
for i,(train,test) in enumerate(skf):
    print "Fold", i
    X_train = train_df[train]
    y_train = y[train]
    X_test = train_df[test]
    y_test = y[test]
    clf.fit(X_train, y_train)
    y_submission = clf.predict_proba(X_test)
    stacking_values[test] = y_submission

np.savetxt('baseline_RF_2_stacking.csv',stacking_values,fmt='%f')
