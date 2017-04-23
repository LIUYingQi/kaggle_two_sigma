import cPickle as pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

if __name__ == '__main__':

    f = open('pkl_files/train_df_1.pkl','rb')
    trainset = pickle.load(f)
    f.close()
    f = open('pkl_files/test_df_1.pkl','rb')
    testset = pickle.load(f)
    f.close()
    f = open('pkl_files/label_1.pkl','rb')
    label = pickle.load(f)
    f.close()

    print trainset.shape
    print testset.shape

    label = np.reshape(label,(label.size,1))
    testset_listing_id = testset['listing_id'].values
    testset_listing_id = np.reshape(testset_listing_id,(testset_listing_id.size,1))
    trainset = trainset.values
    testset = testset.values
    stander = StandardScaler()
    total_set = np.vstack((trainset,testset))
    stander.fit(total_set)
    trainset = stander.transform(trainset)
    testset = stander.transform(testset)

    xgb_stacking = np.loadtxt('baseline_xgboost_1_stacking.csv')
    xgb_test = pd.read_csv('baseline_xgboost_1_submission.csv')
    xgb_test = xgb_test.values[:,1:]
    print xgb_stacking
    print xgb_test

    RF_stacking = np.loadtxt('baseline_RF_1_stacking.csv')
    RF_test = pd.read_csv('baseline_RF_1_submission.csv')
    RF_test = RF_test.values[:,1:]
    print RF_stacking
    print RF_test

    trainset = np.hstack((label,trainset,RF_stacking,xgb_stacking))
    testset = np.hstack((testset_listing_id,testset,RF_test,xgb_test))

    np.savetxt('train_stacknet.csv',trainset,delimiter=',',fmt='%.5f')
    np.savetxt('test_stacknet.csv',testset,delimiter=',',fmt='%.5f')