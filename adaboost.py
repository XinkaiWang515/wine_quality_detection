from data import r_X_train, r_y_train, r_X_test, r_y_test, r_X_val, scal_X_train, scal_X_val, scal_X_test
# from data_adjust import r_X_train, r_y_train, r_X_test, r_y_test, r_X_val, scal_X_train, scal_X_val, scal_X_test
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
import numpy as np
from feature_sel import feature_selection
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from adaboost_tune import ada_para_tune
from smote import X_res,y_res
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def adaboost(X_train,y_train,X_val,X_test,y_test):
    ada_best_param, val_bal_acc, val_accu = ada_para_tune(X_train,y_train,X_val)
    print(ada_best_param)

    features = feature_selection(X_train,y_train,ada_best_param['fea_num'])[-1]
    sel_X_train=X_train.iloc[:,features]
    sel_X_test=X_test.iloc[:,features]
    adab = AdaBoostClassifier(base_estimator=ada_best_param['base_estimator'],n_estimators=ada_best_param['n_estimators'],
                       learning_rate=ada_best_param['learning_rate'],random_state=1)
    # adab = AdaBoostClassifier(base_estimator=ada_best_param['base_estimator'],
    #                           n_estimators=ada_best_param['n_estimators'],
    #                           learning_rate=ada_best_param['learning_rate'], algorithm=ada_best_param['algorithm'],
    #                           random_state=1)
    adab.fit(sel_X_train,y_train)
    train_pred = adab.predict(sel_X_train)
    tr_bal_accu = balanced_accuracy_score(y_train, train_pred)
    tr_accu = accuracy_score(y_train, train_pred)

    y_pred = adab.predict(sel_X_test)
    feature_n = ada_best_param['fea_num']
    bal_accu = balanced_accuracy_score(y_test,y_pred)
    f1_sc = f1_score(y_test,y_pred,average='weighted')
    accu = accuracy_score(y_test,y_pred)

    return feature_n,bal_accu,f1_sc,accu,tr_bal_accu, tr_accu

if __name__ == "__main__":
    feature_n, bal_accu, f1_sc, accu, bal_acc, tr_accu = adaboost(r_X_train,r_y_train,r_X_val,r_X_test,r_y_test)
    print('selected feature numbers: %d'% feature_n)
    print('adaboost test balanced accuracy score:', bal_accu)
    print('adaboost test f1 score:', f1_sc)
    print('adaboost test accuracy:', accu)
    print('adaboost training balanced accuracy:', bal_acc)
    print('adaboost training accuracy:', tr_accu)

    # std_feature_n,std_bal_accu,std_f1_sc,std_accu,std_bal_acc,std_tr_accu = adaboost(scal_X_train,r_y_train,scal_X_val,scal_X_test,r_y_test)
    # print('\n')
    # print('after standardizing features')
    # print('selected feature numbers: %d' % std_feature_n)
    # print('adaboost balanced accuracy score:', std_bal_accu)
    # print('adaboost f1 score:', std_f1_sc)
    # print('adaboost accuracy:', std_accu)
    # print('adaboost training balanced accuracy:', std_bal_acc)
    # print('adaboost training accuracy:', std_tr_accu)
    #
    # sm_feature_n, sm_bal_accu, sm_f1_sc, sm_accu,sm_bal_acc,sm_tr_accu = adaboost(r_X_train, r_y_train, r_X_val, r_X_test, r_y_test)
    # print('\n')
    # print('after downsampling by OSS and then use SMOTE')
    # print('selected feature numbers: %d' % sm_feature_n)
    # print('adaboost balanced accuracy score:', sm_bal_accu)
    # print('adaboost f1 score:', sm_f1_sc)
    # print('adaboost accuracy:', sm_accu)
    # print('adaboost training balanced accuracy:', sm_bal_acc)
    # print('adaboost training accuracy:', sm_tr_accu)