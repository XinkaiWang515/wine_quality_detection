from __future__ import division
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, f1_score
import xgboost as xgb
# from data import r_X_train, r_y_train, r_X_training, r_y_training, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from data_adjust import r_X_train, r_y_train, r_X_training, r_y_training, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from xgboost.sklearn import XGBClassifier
from smote import X_res,y_res
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from feature_sel import feature_selection
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def xgb_classfier(X_train,y_train,X_test,y_test):
    xgbc = XGBClassifier(objective='multi:softmax',learning_rate=0.1,n_estimators=300,max_depth=6,
                        silent=1,nthread=4,num_class=6,gamma=0,min_child_weight=1,subsample=0.8,colsample_bytree=0.75)
    xgbc.fit(X_train,y_train)

    tr_y_pred = xgbc.predict(X_train)
    tr_acc = accuracy_score(y_train, tr_y_pred)
    tr_bal_acc = balanced_accuracy_score(y_train, tr_y_pred)

    y_pred = xgbc.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    bal_acc = balanced_accuracy_score(y_test,y_pred)
    f1_sc = f1_score(r_y_test, y_pred, average='weighted')

    return bal_acc, f1_sc, acc, tr_bal_acc, tr_acc

if __name__ == "__main__":
    bal_accu, f1_sc, accu, bal_acc, tr_accu = xgb_classfier(r_X_train, r_y_train, r_X_test,
                                                                       r_y_test)
    print('random forest test balanced accuracy score:', bal_accu)
    print('random forest test f1 score:', f1_sc)
    print('random forest test accuracy:', accu)
    print('random forest training balanced accuracy:', bal_acc)
    print('random forest training accuracy:', tr_accu)

    std_bal_accu, std_f1_sc, std_accu, std_bal_acc, std_tr_accu = xgb_classfier(scal_X_train, r_y_train,scal_X_test,r_y_test)
    print('\n')
    print('after standardizing features')
    print('random forest test balanced accuracy score:', std_bal_accu)
    print('random forest test f1 score:', std_f1_sc)
    print('random forest test accuracy:', std_accu)
    print('random forest training balanced accuracy:', std_bal_acc)
    print('random forest training accuracy:', std_tr_accu)

    sm_bal_accu, sm_f1_sc, sm_accu, sm_bal_acc, sm_tr_accu = xgb_classfier(X_res, y_res, r_X_test, r_y_test)
    print('\n')
    print('after downsampling by OSS and then use SMOTE')
    print('random forest test balanced accuracy score:', sm_bal_accu)
    print('random forest test f1 score:', sm_f1_sc)
    print('random forest test accuracy:', sm_accu)
    print('random forest training balanced accuracy:', sm_bal_acc)
    print('random forest training accuracy:', sm_tr_accu)