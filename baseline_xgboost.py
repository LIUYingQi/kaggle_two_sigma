# this file is to do feature engineering

import numpy as np
import pandas as pd
import cPickle as pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier

# train_df = pd.read_json(open("input/train.json", "r"))
#
# # print train_df.describe()
# # print train_df.columns
#
# #################################################################################
# # nature features
# # num_feature : bathrooms  bedrooms   price
# # computer vision : photos
# # location feature : latitude longitude  street_address  display_address
# # time feature : created
# # NLP : features description
# # ID_feature : listing_id  manager_id  building_id
# ################################################################################
#
# print "generating train set"
#
# low_count = len(train_df[train_df['interest_level'] == "low"])
# medium_count = len(train_df[train_df['interest_level'] == "medium"])
# high_count = len(train_df[train_df['interest_level'] == "high"])
# total_num = low_count + medium_count + high_count
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
# train_df["price/bedrooms"] = train_df["price"]/(train_df["bedrooms"]+0.5)
# train_df["price/bathrooms"] = train_df["price"]/(train_df["bathrooms"]+0.5)
# train_df["price/rooms"] = train_df["price"]/(train_df["bedrooms"]+train_df["bathrooms"]+0.5)
# train_df["bed/bath"] = train_df["bedrooms"]/(train_df["bathrooms"]+0.5)
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
# train_df = train_df[train_df.latitude!=0.0]
# train_df = train_df[train_df.longitude!=0.0]
#
# def find_strange_latitude(lat):
#     if lat<40.5 or lat>40.9:
#         return 40.74875
#     else:
#         return lat
#
# def find_strange_longitude(lon):
#     if lon>-73.6 or lon<-74.6:
#         return -73.968
#     else:
#         return lon
#
# train_df["latitude"] = train_df["latitude"].apply(find_strange_latitude)
# train_df["longitude"] = train_df["longitude"].apply(find_strange_longitude)
#
# train_df["distance_to_manhaton_down"]=((train_df["latitude"]-NY_manhaton_down_lat)**2+(train_df["longitude"]-NY_manhaton_down_lon)**2)*1000
# train_df["distance_to_UN"]=((train_df["latitude"]-UN_lat)**2+(train_df["longitude"]-UN_lon)**2)*1000
# train_df["distance_to_long_island"]=((train_df["latitude"]-long_island_lat)**2+(train_df["longitude"]-long_island_lon)**2)*1000
# train_df["distance_to_queens"]=((train_df["latitude"]-queens_lat)**2+(train_df["longitude"]-queens_lon)**2)*1000
#
# del train_df["latitude"]
# del train_df["longitude"]
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
# # manager id
# train_df["manager_id"] = train_df["manager_id"].astype(str)
# train_df["interest_level"] = train_df["interest_level"].astype(str)
#
# manager_sell_nums = train_df.groupby("manager_id").count()["num_rooms"].to_dict()
# manager_interest_level = train_df.groupby(["manager_id","interest_level"]).count()["num_rooms"].to_dict()
#
# train_df["probability_low_manager"] = 0
# train_df["probability_medium_manager"] = 0
# train_df["probability_high_manager"] = 0
#
# def posterior_pro_manager (id,level):
#     if id in manager_sell_nums and manager_sell_nums[id]>9 and (id,level) in manager_interest_level:
#         return manager_interest_level[(id,level)]/(manager_sell_nums[id]+0.001)
#     else:
#         if level=="low":
#             return priori_low
#         if level=="medium":
#             return priori_medium
#         else:
#             return priori_high
#
# train_df["probability_high_manager"] = train_df["manager_id"].apply(lambda x: posterior_pro_manager(x,"high"))
# train_df["probability_medium_manager"] = train_df["manager_id"].apply(lambda x: posterior_pro_manager(x,"medium"))
# train_df["probability_low_manager"] = train_df["manager_id"].apply(lambda x: posterior_pro_manager(x,"low"))
#
# del train_df["manager_id"]
#
# # building id
# train_df["building_id"] = train_df["building_id"].astype(str)
#
# building_sell_nums = train_df.groupby("building_id").count()["num_rooms"].to_dict()
# building_interest_level = train_df.groupby(["building_id","interest_level"]).count()["num_rooms"].to_dict()
#
# train_df["probability_low_building"] = 0
# train_df["probability_medium_building"] = 0
# train_df["probability_high_building"] = 0
#
# def posterior_pro_building (id,level):
#     if id in building_sell_nums and id!="0" and (id,level) in building_interest_level:
#         return building_interest_level[(id,level)]/(building_sell_nums[id]+0.001)
#     else:
#         if level=="low":
#             return priori_low
#         if level=="medium":
#             return priori_medium
#         else:
#             return priori_high
#
# train_df["probability_high_building"] = train_df["building_id"].apply(lambda x: posterior_pro_building(x,"high"))
# train_df["probability_medium_building"] = train_df["building_id"].apply(lambda x: posterior_pro_building(x,"medium"))
# train_df["probability_low_building"] = train_df["building_id"].apply(lambda x: posterior_pro_building(x,"low"))
# del train_df["building_id"]
#
# # delete useless features
# def remove_columns(X):
#     columns = ["photos","description","created","street_address","display_address"]
#     for c in columns:
#         del X[c]
# remove_columns(train_df)
# print train_df
#
# #############################################################################
# #label
# #############################################################################
#
# # label Y
# interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
# train_df['interest_level'] = train_df['interest_level'].apply(lambda x: interest_level_map[x])
# y = train_df['interest_level'].ravel()
#
# del train_df["interest_level"]
# print y[:30]
# # print train_df.dtypes
#
# #############################################################################
# # test_set
# #############################################################################
#
# test_df = pd.read_json(open("input/test.json", "r"))
# print "generating test set"
#
# # naive feature engineering
# ###################################################################################
# print "naive features"
# test_df["num_photos"] = test_df["photos"].apply(len)
# test_df["num_features"] = test_df["features"].apply(len)
# test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))
# test_df["price/bedrooms"] = test_df["price"]/(test_df["bedrooms"]+0.5)
# test_df["price/bathrooms"] = test_df["price"]/(test_df["bathrooms"]+0.5)
# test_df["price/rooms"] = test_df["price"]/(test_df["bedrooms"]+test_df["bathrooms"]+0.5)
# test_df["bed/bath"] = test_df["bedrooms"]/(test_df["bathrooms"]+0.5)
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
# test_df["latitude"] = test_df["latitude"].apply(find_strange_latitude)
# test_df["longitude"] = test_df["longitude"].apply(find_strange_longitude)
# test_df["distance_to_manhaton_down"]=((test_df["latitude"]-NY_manhaton_down_lat)**2+(test_df["longitude"]-NY_manhaton_down_lon)**2)*1000
# test_df["distance_to_UN"]=((test_df["latitude"]-UN_lat)**2+(test_df["longitude"]-UN_lon)**2)*1000
# test_df["distance_to_long_island"]=((test_df["latitude"]-long_island_lat)**2+(test_df["longitude"]-long_island_lon)**2)*1000
# test_df["distance_to_queens"]=((test_df["latitude"]-queens_lat)**2+(test_df["longitude"]-queens_lon)**2)*1000
#
# del test_df["latitude"]
# del test_df["longitude"]
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
# # manager id
# test_df["manager_id"] = test_df["manager_id"].astype(str)
#
# test_df["probability_low_manager"] = 0
# test_df["probability_medium_manager"] = 0
# test_df["probability_high_manager"] = 0
#
# test_df["probability_high_manager"] = test_df["manager_id"].apply(lambda x: posterior_pro_manager(x,"high"))
# test_df["probability_medium_manager"] = test_df["manager_id"].apply(lambda x: posterior_pro_manager(x,"medium"))
# test_df["probability_low_manager"] = test_df["manager_id"].apply(lambda x: posterior_pro_manager(x,"low"))
#
# del test_df["manager_id"]
#
# # building id
# test_df["building_id"] = test_df["building_id"].astype(str)
#
# test_df["probability_low_building"] = 0
# test_df["probability_medium_building"] = 0
# test_df["probability_high_building"] = 0
#
# test_df["probability_high_building"] = test_df["building_id"].apply(lambda x: posterior_pro_building(x,"high"))
# test_df["probability_medium_building"] = test_df["building_id"].apply(lambda x: posterior_pro_building(x,"medium"))
# test_df["probability_low_building"] = test_df["building_id"].apply(lambda x: posterior_pro_building(x,"low"))
# del test_df["building_id"]
#
# # delete useless features
# remove_columns(test_df)
# print test_df
#
# #############################################################################
# # check train set / label / test set
# #############################################################################
#
# test_dtypes,test_columns = test_df.dtypes,test_df.columns
# train_dtypes,train_columns = train_df.dtypes,train_df.columns
#
# for i in range(len(train_dtypes)):
#     print (train_columns[i],train_dtypes[i])
#     print (test_columns[i],test_dtypes[i])
#
# print train_df
# # print test_df
#
# for i in y:
#     print i
#
# # print train_df.values
# # print test_df.values
# #
# # train_df_mean = np.mean(train_df.values,axis=0)
# # test_df_mean = np.mean(test_df.values,axis=0)
# # print train_df_mean
# # print test_df_mean
# # print train_df_mean/test_df_mean
# # print y
# # index = np.where(np.abs((train_df_mean/test_df_mean)-1)>0.1)[0]
# # print index
# # count = 0
# # for item in zip(dtypes,columns):
# #     if count in index:
# #         print item
# #     count+=1
#
# ###########################################################################
# # serialization
# ##############################################################################
# f_train_df = open("train_df.pkl",'wb')
# pickle.dump(train_df,f_train_df)
# f_train_df.close()
#
# f_test_df = open("test_df.pkl",'wb')
# pickle.dump(test_df,f_test_df)
# f_test_df.close()
#
# f_y = open("label.pkl",'wb')
# pickle.dump(y,f_y)
# f_y.close()

##############################################################################
# train
##############################################################################

print "fitting..."
train_df_f = open("train_df.pkl","rb")
train_df = pickle.load(train_df_f)
test_df_f = open("test_df.pkl",'rb')
test_df = pickle.load(test_df_f)
y_f = open("label.pkl",'rb')
y = pickle.load(y_f)
train_df_f.close()
test_df_f.close()
y_f.close()

print train_df

# # CV
# param = {}
# param['objective'] = 'multi:softprob'
# param['eta'] = 0.02
# param['max_depth'] = 9
# param['min_child_weight'] = 8
# param['gamma'] = 0.2
# param['silent'] = 1
# param['num_class'] = 3
# param['eval_metric'] = "mlogloss"
# param['min_child_weight'] = 1
# param['subsample'] = 0.7
# param['colsample_bytree'] = 0.7
# param['seed'] = 20
# num_rounds = 2500
# #
# xgtrain = xgb.DMatrix(train_df, label=y)
#
# ##   CV
# res = xgb.cv(param, xgtrain, num_rounds,nfold=5,metrics='mlogloss')
# print res
# f = open("res.pkl","w")
# pickle.dump(res,f)
#
#
# # grid search
# # cv_params = {'gamma': [0,0.1,0.2,0.3,0.4,0.5]}
# # ind_params = {'learning_rate': 0.1, 'n_estimators': 55, 'seed':10, 'subsample': 0.8, 'colsample_bytree': 0.8,'max_depth':9,'min_child_weight':8,'gamma':0.2,
# #              'objective': 'multi:softprob'}
# # optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),
# #                             cv_params,
# #                              scoring = 'log_loss', cv = 5, n_jobs = -1)
# # optimized_GBM.fit(train_df.values,y)
# # print optimized_GBM.cv_results_
# # print optimized_GBM.best_estimator_
#
#
# clf = xgb.train(param, xgtrain, num_rounds)
#
# print("Fitted")
#
# def prepare_submission(model):
#     xgtest = xgb.DMatrix(test_df)
#     preds = model.predict(xgtest)
#     print preds
#     sub = pd.DataFrame(data={'listing_id': test_df['listing_id'].ravel()})
#     sub['low'] = preds[:, 0]
#     sub['medium'] = preds[:, 1]
#     sub['high'] = preds[:, 2]
#     sub.to_csv("submission.csv", index=False, header=True)
#
# prepare_submission(clf)