import cPickle as pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from matplotlib import pyplot
from scipy import stats

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

    # trainset = np.hstack((label,trainset,RF_stacking,xgb_stacking))
    trainset = np.hstack((trainset,RF_stacking,xgb_stacking))
    # testset = np.hstack((testset_listing_id,testset,RF_test,xgb_test))
    testset = np.hstack((testset,RF_test,xgb_test))

    print trainset.shape
    print testset.shape
    print label.shape

    # np.savetxt('train_stacknet.csv',trainset,delimiter=',',fmt='%.5f')
    # np.savetxt('test_stacknet.csv',testset,delimiter=',',fmt='%.5f')

    # print label==0
    #
    # index1 = (label==0).flatten()
    # index2 = (label==1).flatten()
    # index3 = (label==2).flatten()
    #
    # trainset_0 = trainset[index1]
    # trainset_1 = trainset[index2]
    # trainset_2 = trainset[index3]
    #
    # for i in range(25):
    #     print stats.describe(trainset_0[:,i*10:(i+1)*10])
    #     print stats.describe(trainset_1[:,i*10:(i+1)*10])
    #     print stats.describe(trainset_2[:,i*10:(i+1)*10])

    # for i in range(250):
    #     pyplot.figure()
    #     sns.distplot(trainset_0[:,i],kde=False,rug=True)
    #     sns.distplot(trainset_1[:,i],kde=False,rug=True)
    #     sns.distplot(trainset_2[:,i],kde=False,rug=True)
    #     pyplot.show()

    # clf = GradientBoostingClassifier(n_estimators=300)
    # clf = clf.fit(trainset, label)
    # importance = clf.feature_importances_
    # np.savetxt('feature_importance.csv',importance,fmt='%f')
    #
    # model = SelectFromModel(clf, prefit=True)
    # trainset = model.transform(trainset)
    # print trainset.shape

    importance = np.loadtxt('feature_importance.csv')
    feature_selected = importance > 0.0001       # can change here
    print np.mean(feature_selected)
    trainset = trainset[:,feature_selected]
    testset = testset[:,feature_selected]

    print trainset.shape
    print testset.shape

    xgb_stacking = np.loadtxt('baseline_xgboost_2_stacking.csv')
    xgb_test = pd.read_csv('baseline_xgboost_2_submission.csv')
    xgb_test = xgb_test.values[:,1:]
    print xgb_stacking
    print xgb_test

    RF_stacking = np.loadtxt('baseline_RF_2_stacking.csv')
    RF_test = pd.read_csv('baseline_RF_2_submission.csv')
    RF_test = RF_test.values[:,1:]
    print RF_stacking
    print RF_test

    trainset = np.hstack((label,trainset,RF_stacking,xgb_stacking))
    testset = np.hstack((testset_listing_id,testset,RF_test,xgb_test))

    print trainset.shape
    print testset.shape

    np.savetxt('train_stacknet.csv',trainset,delimiter=',',fmt='%.5f')
    np.savetxt('test_stacknet.csv',testset,delimiter=',',fmt='%.5f')