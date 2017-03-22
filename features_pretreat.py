# this file is to do feature engineering

import numpy as np
import pandas as pd
from collections import Counter,defaultdict
import cPickle as pkl

train_df = pd.read_json(open("input/train.json", "r"))
test_df = pd.read_json(open("input/test.json", "r"))
train_df = pd.concat([train_df, test_df], 0)

# find all not rare features
train_df["features"] = train_df["features"].astype('str').str.replace("[","")
train_df["features"] = train_df["features"].str.replace("u","")
train_df["features"] = train_df["features"].str.replace("'","")
train_df["features"] = train_df["features"].str.replace("\"","")
train_df["features"] = train_df["features"].str.replace("!","")
train_df["features"] = train_df["features"].str.replace("]","")
train_df["features"] = train_df["features"].str.split(",")
train_df["features"] = train_df[["features"]].apply(lambda line: [list(map(str.strip, map(str.lower, x))) for x in line])
features = train_df["features"]
feature_counts = Counter()
for feature in features:
    feature_counts.update(feature)
feature_list = sorted([k for (k,v) in feature_counts.items() if v > 10])  # 5 as threshold for rare feature
print feature_list

# merge duplicated features into a dict
def clean(x):
    x = x.replace("-", " ")
    x = x.replace("twenty four hour", "24")
    x = x.replace("24/7", "24")
    x = x.replace("24hr", "24")
    x = x.replace("fll-time", "24")
    x = x.replace("24-hour", "24")
    x = x.replace("24hour", "24")
    x = x.replace("24 hour", "24")
    x = x.replace("ft", "24")
    x = x.replace("apt. ", "")
    x = x.replace("actal", "actual")
    x = x.replace("common", "cm")
    x = x.replace("concierge", "doorman")
    x = x.replace("bicycle", "bike")
    x = x.replace("private", "pv")
    x = x.replace("deco", "dc")
    x = x.replace("decorative", "dc")
    x = x.replace("onsite", "os")
    x = x.replace("on-site", "os")
    x = x.replace("outdoor", "od")
    x = x.replace("ss appliances", "stainless")
    x = x.replace("garantor", "guarantor")
    x = x.replace("high speed", "hp")
    x = x.replace("high-speed", "hp")
    x = x.replace("hi", "high")
    x = x.replace("eatin", "eat in")
    x = x.replace("pet", "pets")
    x = x.replace("indoor", "id")
    x = x.replace("redced", "reduced")
    x = x.replace("indoor", "id")
    return x

key2original = defaultdict(list)
for f in feature_list:
    cleaned = clean(f)
    key = cleaned[:5].strip()
    key2original[key].append(f)
del key2original[""]

output = open('features_dict.pkl', 'wb')
print key2original.keys()
pkl.dump(key2original,output)

