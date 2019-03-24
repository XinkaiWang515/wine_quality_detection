import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from data import r_X_train,r_y_train


def feature_selection(X_train, y_train, fea_num):
    # skb = SelectKBest(chi2, k=7)
    # fit = skb.fit(r_X_train, r_y_train)
    # print(fit.scores_)
    # print(fit.pvalues_)

    rfe = RFE(LogisticRegression(multi_class='multinomial', solver='newton-cg'),fea_num)
    rfe_sle = rfe.fit(X_train, y_train)
    # print("Num Features: %d"% fit.n_features_)
    # print("Selected Features: %s"% fit.support_)
    # print("Feature Ranking: %s"% fit.ranking_)
    features = []
    for idx,select in enumerate(rfe_sle.ranking_):
        if select==1:
            features.append(idx)


    # pca = PCA(n_components=7)
    # fit_pca = pca.fit(scal_X_train)
    # print('\n')
    # print(fit_pca.explained_variance_ratio_)


    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    imp_sc = model.feature_importances_
    # print(imp_sc)
    etc_sel_fea = []
    for idx,n in enumerate(imp_sc):
        if n>=sorted(imp_sc)[11-fea_num]:
            etc_sel_fea.append(idx)
    # print("importance rank: %s"% rank)
    return (rfe_sle,features,imp_sc,etc_sel_fea)

if __name__=="__main__":
    fit,features,imp_sc,etc_sel_fea = feature_selection(r_X_train,r_y_train,8)
    print("Num Features: %d" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
    print('\n')
    print("feature importance scores: %s"% imp_sc)
    print('\n')
    print("rfe with logistic regression selected features: %s"% features)
    print("feature importance of extra trees classifier selected features: %s"% etc_sel_fea)