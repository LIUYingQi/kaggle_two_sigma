import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import random
from math import exp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import xgboost as xgb
import cPickle as pickle

random.seed(321)
np.random.seed(321)

X_train = pd.read_json("input/train.json")
X_test = pd.read_json("input/test.json")

interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
X_train['interest_level'] = X_train['interest_level'].apply(lambda x: interest_level_map[x])
X_test['interest_level'] = -1

# add features
feature_transform = CountVectorizer(stop_words='english', max_features=150)
X_train['features'] = X_train["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
X_test['features'] = X_test["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
feature_transform.fit(list(X_train['features']) + list(X_test['features']))

train_size = len(X_train)
low_count = len(X_train[X_train['interest_level'] == 0])
medium_count = len(X_train[X_train['interest_level'] == 1])
high_count = len(X_train[X_train['interest_level'] == 2])

def find_objects_with_only_one_record(feature_name):
    temp = pd.concat([X_train[feature_name].reset_index(),
                      X_test[feature_name].reset_index()])
    temp = temp.groupby(feature_name, as_index=False).count()
    return temp[temp['index'] == 1]

managers_with_one_lot = find_objects_with_only_one_record('manager_id')
buildings_with_one_lot = find_objects_with_only_one_record('building_id')
addresses_with_one_lot = find_objects_with_only_one_record('display_address')

lambda_val = None
k = 5.0
f = 1.0
r_k = 0.01
g = 1.0

def categorical_average(variable, y, pred_0, feature_name):
    def calculate_average(sub1, sub2):
        s = pd.DataFrame(data={
            variable: sub1.groupby(variable, as_index=False).count()[variable],
            'sumy': sub1.groupby(variable, as_index=False).sum()['y'],
            'avgY': sub1.groupby(variable, as_index=False).mean()['y'],
            'cnt': sub1.groupby(variable, as_index=False).count()['y']
        })

        tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable)
        del tmp['index']
        tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
        tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

        def compute_beta(row):
            cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
            return 1.0 / (g + exp((cnt - k) / f))

        if lambda_val is not None:
            tmp['beta'] = lambda_val
        else:
            tmp['beta'] = tmp.apply(compute_beta, axis=1)

        tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred_0'],axis=1)

        tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred_0']
        tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred_0']
        tmp['random'] = np.random.uniform(size=len(tmp))
        tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] * (1 + (row['random'] - 0.5) * r_k),axis=1)

        return tmp['adj_avg'].ravel()

    # cv for training set
    k_fold = StratifiedKFold(5)
    X_train[feature_name] = -999
    for (train_index, cv_index) in k_fold.split(np.zeros(len(X_train)),X_train['interest_level'].ravel()):
        sub = pd.DataFrame(data={variable: X_train[variable],'y': X_train[y],'pred_0': X_train[pred_0]})

        sub1 = sub.iloc[train_index]
        sub2 = sub.iloc[cv_index]

        X_train.loc[cv_index, feature_name] = calculate_average(sub1, sub2)

    # for test set
    sub1 = pd.DataFrame(data={variable: X_train[variable],'y': X_train[y],'pred_0': X_train[pred_0]})
    sub2 = pd.DataFrame(data={variable: X_test[variable],'y': X_test[y],'pred_0': X_test[pred_0]})
    X_test.loc[:, feature_name] = calculate_average(sub1, sub2)


def transform_data(X):
    # add features
    feat_sparse = feature_transform.transform(X["features"])
    vocabulary = feature_transform.vocabulary_
    del X['features']
    X1 = pd.DataFrame([pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0])])
    X1.columns = list(sorted(vocabulary.keys()))
    X = pd.concat([X.reset_index(), X1.reset_index()], axis=1)
    del X['index']

    X["num_photos"] = X["photos"].apply(len)
    X['created'] = pd.to_datetime(X["created"])
    X["num_description_words"] = X["description"].apply(lambda x: len(x.split(" ")))
    X['price_per_bed'] = X['price'] / (X['bedrooms']+0.5)
    X['price_per_bath'] = X['price'] / (X['bathrooms']+0.5)
    X['price_per_room'] = X['price'] / ((X['bathrooms'] + X['bedrooms'])+0.5)

    X['low'] = 0
    X.loc[X['interest_level'] == 0, 'low'] = 1
    X['medium'] = 0
    X.loc[X['interest_level'] == 1, 'medium'] = 1
    X['high'] = 0
    X.loc[X['interest_level'] == 2, 'high'] = 1

    X['display_address'] = X['display_address'].apply(lambda x: x.lower().strip())
    X['street_address'] = X['street_address'].apply(lambda x: x.lower().strip())

    X['pred0_low'] = low_count * 1.0 / train_size
    X['pred0_medium'] = medium_count * 1.0 / train_size
    X['pred0_high'] = high_count * 1.0 / train_size

    X.loc[X['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()),
          'manager_id'] = "-1"
    X.loc[X['building_id'].isin(buildings_with_one_lot['building_id'].ravel()),
          'building_id'] = "-1"
    X.loc[X['display_address'].isin(addresses_with_one_lot['display_address'].ravel()),
          'display_address'] = "-1"
                                                                                  ################  this one!
    return X


def normalize_high_cordiality_data():
    high_cardinality = ["building_id", "manager_id"]
    for c in high_cardinality:
        categorical_average(c, "medium", "pred0_medium", c + "_mean_medium")
        categorical_average(c, "high", "pred0_high", c + "_mean_high")
        categorical_average(c, "low", "pred0_low", c + "_mean_low")


def transform_categorical_data():
    categorical = ['building_id', 'manager_id',
                   'display_address', 'street_address']

    for f in categorical:
        encoder = LabelEncoder()
        encoder.fit(list(X_train[f]) + list(X_test[f]))
        X_train[f] = encoder.transform(X_train[f].ravel())
        X_test[f] = encoder.transform(X_test[f].ravel())


def remove_columns(X):
    columns = ["photos", "pred0_high", "pred0_low", "pred0_medium",
               "description", "low", "medium", "high",
               "interest_level", "created"]
    for c in columns:
        del X[c]


print("Starting transformations")
X_train = transform_data(X_train)
X_test = transform_data(X_test)
y = X_train['interest_level'].ravel()

print y

print("Normalizing high cordiality data...")
normalize_high_cordiality_data()
transform_categorical_data()

remove_columns(X_train)
remove_columns(X_test)

dtypes,columns = X_test.dtypes,X_test.columns

print X_train.values
print X_test.values
for item in zip(dtypes,columns):
    print item
train_df_mean = np.mean(X_train.values,axis=0)
test_df_mean = np.mean(X_test.values,axis=0)
print train_df_mean
print test_df_mean
print train_df_mean/test_df_mean
print y
index = np.where(np.abs((train_df_mean/test_df_mean)-1)>0.1)[0]
print index
count = 0
for item in zip(dtypes,columns):
    if count in index:
        print item
    count+=1

print X_train


def find_strange_latitude(lat):
    if lat<40.5 or lat>40.9:
        return 40.74875
    else:
        return lat

def find_strange_longitude(lon):
    if lon>-73.6 or lon<-74.6:
        return -73.968
    else:
        return lon

X_train["latitude"] = X_train["latitude"].apply(find_strange_latitude)
X_train["longitude"] = X_train["longitude"].apply(find_strange_longitude)

X_train["latitude"] = X_train["latitude"].apply(find_strange_latitude)
X_train["longitude"] = X_train["longitude"].apply(find_strange_longitude)

test_dtypes,test_columns = X_test.dtypes,X_test.columns
train_dtypes,train_columns = X_train.dtypes,X_train.columns

for i in range(len(train_dtypes)):
    print (train_columns[i],train_dtypes[i])
    print (test_columns[i],test_dtypes[i])

print X_train.describe()
print X_test.describe()
print y

print X_train.loc[:,['listing_id','building_id_mean_medium','building_id_mean_high','manager_id_mean_medium','manager_id_mean_high']]

f = open('train_id_proba_features.pkl','w')
pickle.dump(X_train.loc[:,['listing_id','building_id_mean_low','building_id_mean_medium','building_id_mean_high','manager_id_mean_low','manager_id_mean_medium','manager_id_mean_high']],f)
f.close()

f = open('test_id_proba_features.pkl','w')
pickle.dump(X_test.loc[:,['listing_id','building_id_mean_low','building_id_mean_medium','building_id_mean_high','manager_id_mean_low','manager_id_mean_medium','manager_id_mean_high']],f)
f.close()


# print("Start fitting...")
#
# param = {}
# param['objective'] = 'multi:softprob'
# param['eta'] = 0.02
# param['max_depth'] = 4
# param['silent'] = 1
# param['num_class'] = 3
# param['eval_metric'] = "mlogloss"
# param['min_child_weight'] = 1
# param['subsample'] = 0.7
# param['colsample_bytree'] = 0.7
# param['seed'] = 321
# param['nthread'] = 8
# num_rounds = 2000
#
# xgtrain = xgb.DMatrix(X_train, label=y)
# clf = xgb.XGBClassifier()
#
# print("Fitted")
#
# def prepare_submission(model):
#     xgtest = xgb.DMatrix(X_test)
#     preds = model.predict(xgtest)
#     sub = pd.DataFrame(data={'listing_id': X_test['listing_id'].ravel()})
#     sub['low'] = preds[:, 0]
#     sub['medium'] = preds[:, 1]
#     sub['high'] = preds[:, 2]
#     sub.to_csv("baseline_xgboost_1_submission.csv", index=False, header=True)
# prepare_submission(clf)