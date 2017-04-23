# this file is to do feature engineering

import numpy as np
import pandas as pd
import cPickle as pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

# print "reading data......."
# train_df = pd.read_json("input/train.json")
# test_df = pd.read_json(open("input/test.json", "r"))

###############################################################################
# nature features
# num_feature : bathrooms  bedrooms   price
# computer vision : photos
# location feature : latitude longitude  street_address  display_address
# time feature : created
# NLP : features description
# ID_feature : listing_id  manager_id  building_id
##############################################################################
#
# print "generating train set"
#
# low_count = len(train_df[train_df['interest_level'] == "low"])
# medium_count = len(train_df[train_df['interest_level'] == "medium"])
# high_count = len(train_df[train_df['interest_level'] == "high"])
# total_num = len(train_df)
#
# priori_low = low_count/(total_num+0.001)
# priori_medium = medium_count/(total_num+0.001)
# priori_high = high_count/(total_num+0.001)
#
# # naive feature engineering
# ###################################################################################
# print "naive features"
# train_df["num_photos"] = train_df["photos"].apply(len)
# train_df["num_features"] = train_df["features"].apply(len)
# train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
# train_df["price/bedrooms"] = train_df["price"]/(train_df["bedrooms"]+0.05)
# train_df["price/bathrooms"] = train_df["price"]/(train_df["bathrooms"]+0.05)
# train_df["price/rooms"] = train_df["price"]/(train_df["bedrooms"]+train_df["bathrooms"]+0.05)
# train_df["bed/bath"] = train_df["bedrooms"]/(train_df["bathrooms"]+0.05)
# train_df["bed-bath"] = train_df["bedrooms"]-train_df["bathrooms"]
# train_df["num_rooms"] = train_df["bedrooms"]+train_df["bathrooms"]
#
# # time features
# ###################################################################################
# print "time features"
# train_df["created"] = pd.to_datetime(train_df["created"])
# train_df["created_month"] = train_df["created"].dt.month
# train_df["created_day"] = train_df["created"].dt.day
# train_df["time_created"] = (12-train_df["created_month"])*30+30-train_df["created_day"]
#
# # location features
# ####################################################################################
# print "location features"
#
# f = open('train_df_real_location.pkl','r')
# train_df_location = pickle.load(f)
# f.close()
#
# del train_df['latitude']
# del train_df['longitude']
#
# train_df = pd.merge(train_df,train_df_location,on='listing_id')
#
# train_df['latitude'] = train_df['latitude'].astype('float')
# train_df['longitude'] = train_df['longitude'].astype('float')
#
# NY_center_lat = 40.751814
# NY_center_lon = -73.987632
#
# NY_manhaton_down_lat = 40.721229
# NY_manhaton_down_lon = -73.998666
#
# UN_lat = 40.74875
# UN_lon = -73.96825
#
# long_island_lat = 40.744136
# long_island_lon = -73.948159
#
# queens_lat = 40.726694
# queens_lon = -73.808733
#
# train_df["distance_to_center"]=((train_df["latitude"]-NY_center_lat)**2+(train_df["longitude"]-NY_center_lon)**2)*1000
# train_df["distance_to_manhaton_down"]=((train_df["latitude"]-NY_manhaton_down_lat)**2+(train_df["longitude"]-NY_manhaton_down_lon)**2)*1000
# train_df["distance_to_UN"]=((train_df["latitude"]-UN_lat)**2+(train_df["longitude"]-UN_lon)**2)*1000
# train_df["distance_to_long_island"]=((train_df["latitude"]-long_island_lat)**2+(train_df["longitude"]-long_island_lon)**2)*1000
# train_df["distance_to_queens"]=((train_df["latitude"]-queens_lat)**2+(train_df["longitude"]-queens_lon)**2)*1000
#
# # description NLP analysis
# ######################################################################################
# print "NLP features"
#
# # features (change duplicate features, then one-hot)
# # load dict prepared
# features_dict = open("features_dict.pkl",'rb')
# features_dict = pickle.load(features_dict)
# # map values to key in dict
# train_df["features"] = train_df["features"].astype('str').str.replace("[","")
# train_df["features"] = train_df["features"].str.replace("u","")
# train_df["features"] = train_df["features"].str.replace("'","")
# train_df["features"] = train_df["features"].str.replace("\"","")
# train_df["features"] = train_df["features"].str.replace("!","")
# train_df["features"] = train_df["features"].str.replace("]","")
# train_df["features"] = train_df["features"].str.split(",")
# train_df["features"] = train_df[["features"]].apply(lambda line: [list(map(str.strip, map(str.lower, x))) for x in line])
#
# def clean(x):
#     x = x.replace("-", " ")
#     x = x.replace("twenty four hour", "24")
#     x = x.replace("24/7", "24")
#     x = x.replace("24hr", "24")
#     x = x.replace("fll-time", "24")
#     x = x.replace("24-hour", "24")
#     x = x.replace("24hour", "24")
#     x = x.replace("24 hour", "24")
#     x = x.replace("ft", "24")
#     x = x.replace("apt. ", "")
#     x = x.replace("actal", "actual")
#     x = x.replace("common", "cm")
#     x = x.replace("concierge", "doorman")
#     x = x.replace("bicycle", "bike")
#     x = x.replace("private", "pv")
#     x = x.replace("deco", "dc")
#     x = x.replace("decorative", "dc")
#     x = x.replace("onsite", "os")
#     x = x.replace("on-site", "os")
#     x = x.replace("outdoor", "od")
#     x = x.replace("ss appliances", "stainless")
#     x = x.replace("garantor", "guarantor")
#     x = x.replace("high speed", "hp")
#     x = x.replace("high-speed", "hp")
#     x = x.replace("hi", "high")
#     x = x.replace("eatin", "eat in")
#     x = x.replace("pet", "pets")
#     x = x.replace("indoor", "id")
#     x = x.replace("redced", "reduced")
#     x = x.replace("indoor", "id")
#     key = x[:5].strip()
#     return key
#
# train_df["features"] = train_df["features"].apply(lambda line : [clean(x) for x in line])
# # find if in key set
# keys = features_dict.keys()
# values = train_df.loc[:,["features"]].values
# len_values = len(values)
# for item in keys:
#     result_list=[]
#     for row in range(len_values):
#         result_list.append(str(item) in values[row][0])
#     train_df[str(item)] = pd.Series(np.array(result_list,dtype=bool),index=train_df.index)
#     train_df[str(item)] = train_df[str(item)].astype('int')
# del train_df["features"]
#
# # ID features
# ###################################################################################
# print "ID features"
#
# def find_objects_with_only_one_record(feature_name):
#     temp = pd.concat([train_df[feature_name].reset_index(),
#                       test_df[feature_name].reset_index()])
#     temp = temp.groupby(feature_name, as_index=False).count()
#     return temp[temp['index'] == 1]
#
# managers_with_one_lot = find_objects_with_only_one_record('manager_id')
# buildings_with_one_lot = find_objects_with_only_one_record('building_id')
# addresses_with_one_lot = find_objects_with_only_one_record('display_address')
#
# train_df["single_manager_id"] = 0
# train_df.loc[train_df['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()),'single_manager_id'] = 1
#
# f = open('train_id_proba_features.pkl','r')
# id_features = pickle.load(f)
# f.close()
# train_df = pd.merge(train_df,id_features,how='left',on='listing_id')
#
# # delete useless features
# def remove_columns(X):
#     columns = ["photos","description","created","street_address","display_address","manager_id","building_id"]
#     for c in columns:
#         del X[c]
# remove_columns(train_df)
#
# print train_df
# print train_df.dtypes
#
# #############################################################################
# # test_set
# #############################################################################
#
# print "generating test set"
#
# # naive feature engineering
# ###################################################################################
# print "naive features"
# test_df["num_photos"] = test_df["photos"].apply(len)
# test_df["num_features"] = test_df["features"].apply(len)
# test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
# test_df["price/bedrooms"] = test_df["price"]/(test_df["bedrooms"]+0.05)
# test_df["price/bathrooms"] = test_df["price"]/(test_df["bathrooms"]+0.05)
# test_df["price/rooms"] = test_df["price"]/(test_df["bedrooms"]+test_df["bathrooms"]+0.05)
# test_df["bed/bath"] = test_df["bedrooms"]/(test_df["bathrooms"]+0.05)
# test_df["bed-bath"] = test_df["bedrooms"]-test_df["bathrooms"]
# test_df["num_rooms"] = test_df["bedrooms"]+test_df["bathrooms"]
#
# # time features
# ###################################################################################
# print "time features"
# test_df["created"] = pd.to_datetime(test_df["created"])
# test_df["created_month"] = test_df["created"].dt.month
# test_df["created_day"] = test_df["created"].dt.day
# test_df["time_created"] = (12-test_df["created_month"])*30+30-test_df["created_day"]
#
# # location features
# ####################################################################################
# print "location features"
#
# f = open('test_df_real_location.pkl','r')
# test_df_location = pickle.load(f)
# f.close()
#
# del test_df['latitude']
# del test_df['longitude']
#
# test_df = pd.merge(test_df,test_df_location,on='listing_id')
#
# test_df['latitude'] = test_df['latitude'].astype('float')
# test_df['longitude'] = test_df['longitude'].astype('float')
#
# test_df["distance_to_center"]=((test_df["latitude"]-NY_center_lat)**2+(test_df["longitude"]-NY_center_lon)**2)*1000
# test_df["distance_to_manhaton_down"]=((test_df["latitude"]-NY_manhaton_down_lat)**2+(test_df["longitude"]-NY_manhaton_down_lon)**2)*1000
# test_df["distance_to_UN"]=((test_df["latitude"]-UN_lat)**2+(test_df["longitude"]-UN_lon)**2)*1000
# test_df["distance_to_long_island"]=((test_df["latitude"]-long_island_lat)**2+(test_df["longitude"]-long_island_lon)**2)*1000
# test_df["distance_to_queens"]=((test_df["latitude"]-queens_lat)**2+(test_df["longitude"]-queens_lon)**2)*1000
#
# # description NLP analysis
# ######################################################################################
# print "NLP features"
#
# # features (change duplicate features, then one-hot)
# # map values to key in dict
# test_df["features"] = test_df["features"].astype('str').str.replace("[","")
# test_df["features"] = test_df["features"].str.replace("u","")
# test_df["features"] = test_df["features"].str.replace("'","")
# test_df["features"] = test_df["features"].str.replace("\"","")
# test_df["features"] = test_df["features"].str.replace("!","")
# test_df["features"] = test_df["features"].str.replace("]","")
# test_df["features"] = test_df["features"].str.split(",")
# test_df["features"] = test_df[["features"]].apply(lambda line: [list(map(str.strip, map(str.lower, x))) for x in line])
# test_df["features"] = test_df["features"].apply(lambda line : [clean(x) for x in line])
# # find if in key set
# values = test_df.loc[:,["features"]].values
# len_values = len(values)
# for item in keys:
#     result_list=[]
#     for row in range(len_values):
#         result_list.append(str(item) in values[row][0])
#     test_df[str(item)] = pd.Series(np.array(result_list,dtype=bool),index=test_df.index)
#     test_df[str(item)] = test_df[str(item)].astype('int')
# del test_df["features"]
#
# # ID features
# ###################################################################################
# print "ID features"
#
# test_df["single_manager_id"] = 0
# test_df.loc[test_df['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()),'single_manager_id'] = 1
#
# f = open('test_id_proba_features.pkl','r')
# id_features = pickle.load(f)
# f.close()
# test_df = pd.merge(test_df,id_features,on='listing_id')
#
# # delete useless features
# remove_columns(test_df)
# print test_df
#
# #############################################################################
# #label
# #############################################################################
#
# interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
# train_df['interest_level'] = train_df['interest_level'].apply(lambda x: interest_level_map[x])
#
# y = train_df['interest_level'].ravel()
# del train_df["interest_level"]
# print y[:30]
#
# #############################################################################
# # check train set / label / test set
# #############################################################################
#
# # test_dtypes,test_columns = test_df.dtypes,test_df.columns
# # train_dtypes,train_columns = train_df.dtypes,train_df.columns
# #
# # print train_df.shape
# # print test_df.shape
# # print len(y)
# # print train_df
# # for i in range(len(train_dtypes)):
# #     print (train_columns[i],train_dtypes[i])
# #     print (test_columns[i],test_dtypes[i])
# #
# # # print train_df
# # # print test_df
# #
# # print train_df
# # print test_df
# #
# # train_df_mean = np.mean(train_df.values,axis=0)
# # test_df_mean = np.mean(test_df.values,axis=0)
# # print train_df_mean
# # print test_df_mean
# # print train_df_mean/test_df_mean
#
# ###########################################################################
# # serialization
# ##############################################################################
#
# print 'serialization'
#
# print train_df.describe()
# print test_df.describe()
#
# f_train_df = open("train_df_1.pkl",'wb')
# pickle.dump(train_df,f_train_df)
# f_train_df.close()
#
# f_test_df = open("test_df_1.pkl",'wb')
# pickle.dump(test_df,f_test_df)
# f_test_df.close()
#
# f_y = open("label_1.pkl",'wb')
# pickle.dump(y,f_y)
# f_y.close()

##############################################################################
# train
##############################################################################

print "fitting..."
train_df_f = open("pkl_files/train_df_1.pkl","rb")
train_df = pickle.load(train_df_f)
test_df_f = open("pkl_files/test_df_1.pkl",'rb')
test_df = pickle.load(test_df_f)
y_f = open("pkl_files/label_1.pkl",'rb')
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

print train_df.dtypes
print test_df.dtypes
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
clf.fit(train_df,y)
preds = clf.predict_proba(test_df)
print preds
sub = pd.DataFrame(data={'listing_id': test_df['listing_id'].ravel()})
sub['low'] = preds[:, 0]
sub['medium'] = preds[:, 1]
sub['high'] = preds[:, 2]
sub.to_csv("baseline_RF_1_submission.csv", index=False, header=True)

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

np.savetxt('baseline_RF_1_stacking.csv',stacking_values,fmt='%f')
