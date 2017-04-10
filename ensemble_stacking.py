import cPickle as pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler



if __name__ == '__main__':
    # train = np.loadtxt('stack_net/train_stacknet.csv',delimiter=',')
    # print train.shape
    # test = np.loadtxt('stack_net/test_stacknet.csv',delimiter=',')
    # print test.shape


    f = open('pkl_files/train_df_2.pkl','rb')
    trainset = pickle.load(f)
    f.close()
    f = open('pkl_files/test_df_2.pkl','rb')
    testset = pickle.load(f)
    f.close()
    f = open('pkl_files/label_2.pkl','rb')
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

    trainset = np.hstack((label,trainset))
    testset = np.hstack((testset_listing_id,testset))

    np.savetxt('xgboost_2_train_stacknet.csv',trainset,delimiter=',',fmt='%.5f')
    np.savetxt('xgboost_2_test_stacknet.csv',testset,delimiter=',',fmt='%.5f')