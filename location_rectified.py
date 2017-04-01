import numpy as np
import pandas as pd
import cPickle as pkl
from matplotlib import pyplot
import seaborn

import requests

train_df = pd.read_json("input/train.json")
test_df = pd.read_json("input/test.json")


train_df = train_df.loc[:,['listing_id','latitude','longitude','display_address']].values
print train_df.shape
count = 0
for row in train_df:
    if (row[1]<40.5) | (row[1]>40.9) | (row[2]<-74.6) | (row[2]>-73.6):
        print str(row[3])
        try:
            str_find = 'https://maps.googleapis.com/maps/api/geocode/json?address='+str(row[3])+'+NewYork city+USA'
            response = requests.get(str_find)
            resp_json_payload = response.json()
            train_df[count, 1] = resp_json_payload['results'][0]['geometry']['location']['lat']
            train_df[count, 2] = resp_json_payload['results'][0]['geometry']['location']['lng']
        except:
            train_df[count,1]=40.75
            train_df[count,2]=-74.1
    count+=1
print "++++++++++++++++++++++++++++++++++"
for row in train_df:
    if (row[1] < 40.5) | (row[1] > 40.9) | (row[2] < -74.6) | (row[2] > -73.6):
        print str(row[3])

train_df_real_location = pd.DataFrame(train_df[:,:3],columns=['listing_id','latitude','longitude'])
print train_df_real_location['latitude'].max()
print train_df_real_location['latitude'].min()



test_df = test_df.loc[:,['listing_id','latitude','longitude','display_address']].values
print test_df.shape
count = 0
for row in test_df:
    if (row[1]<40.5) | (row[1]>40.9) | (row[2]<-74.6) | (row[2]>-73.6):
        print str(row[3])
        try:
            str_find = 'https://maps.googleapis.com/maps/api/geocode/json?address='+str(row[3])+'+NewYork city+USA'
            response = requests.get(str_find)
            resp_json_payload = response.json()
            test_df[count, 1] = resp_json_payload['results'][0]['geometry']['location']['lat']
            test_df[count, 2] = resp_json_payload['results'][0]['geometry']['location']['lng']
        except:
            test_df[count,1]=40.75
            test_df[count,2]=-74.1
    count+=1
print "++++++++++++++++++++++++++++++++++"
for row in test_df:
    if (row[1] < 40.5) | (row[1] > 40.9) | (row[2] < -74.6) | (row[2] > -73.6):
        print str(row[3])

test_df_real_location = pd.DataFrame(test_df[:,:3],columns=['listing_id','latitude','longitude'])
print test_df_real_location['latitude'].max()
print test_df_real_location['latitude'].min()

f = open('train_df_real_location.pkl','w')
pkl.dump(train_df_real_location,f)
f.close()


f = open('test_df_real_location.pkl','w')
pkl.dump(test_df_real_location,f)
f.close()

