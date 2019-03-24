from __future__ import division
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer
import xgboost as xgb
# from data import r_X_train, r_y_train, r_X_val, r_y_val, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from data_adjust import r_X_train, r_y_train, r_X_val, r_y_val, r_X_test, r_y_test, scal_X_train, scal_X_val, scal_X_test
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from feature_sel import feature_selection

# print((r_y_train-3).head(50))

xg_train = xgb.DMatrix(r_X_train, label=(r_y_train-3))
xg_val = xgb.DMatrix(r_X_val, label=(r_y_val-3))
xgb_param = {'objective': ['multi:softmax'],
            'learning_rate': [0.1],
            'n_estimators': [250,300,350],
            'max_depth': [5,6],
            'silent': [1],
            'nthread': [4],
            'num_class':[6],
            'gamma': [0],
            'min_child_weight': [1,3],
            'subsample': [0.8],
            'colsample_bytree': [0.75],
            # 'reg_alpha':[0.005,1e-2,0.05]
            }

watchlist = [(xg_train, 'train'), (xg_val, 'test')]
num_round = 5
# bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
# pred = bst.predict(xg_val)
# acc = accuracy_score(r_y_val-3, pred)
# bal_acc = balanced_accuracy_score(r_y_val-3, pred)
# print('Test accuracy using softmax = {}'.format(acc))
# print('Test balanced accuracy using softmax = {}'.format(bal_acc))

# xgbc = XGBClassifier(random_state=0)
# scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
#
# clf_grid = GridSearchCV(estimator=xgbc,param_grid=xgb_param,scoring=scoring,cv=5,refit='Accuracy',n_jobs=4)
# clf_grid.fit(r_X_train,r_y_train)
# print(clf_grid.best_params_)
# print(clf_grid.best_score_)
#
# val_pre = clf_grid.predict(r_X_val)
# acc = accuracy_score(r_y_val, val_pre)
# bal_acc = balanced_accuracy_score(r_y_val, val_pre)
# print('accuracy',acc)
# print('balanced accuracy',bal_acc)

def xgb_para_tune(X_train,y_train,X_val):
    fea_num = [8,9,10,11]
    max_bal_acc = 0

    xgbc = XGBClassifier(random_state=0)
    scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
    for n in fea_num:
        etc_feat = feature_selection(X_train, y_train, n)[-1]
        sel_X_train = X_train.iloc[:, etc_feat]
        sel_X_val = X_val.iloc[:, etc_feat]
        # xgbc = XGBClassifier(random_state=0)
        # scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
        clf_grid = GridSearchCV(estimator=xgbc,param_grid=xgb_param,scoring=scoring,cv=5,refit='Accuracy',n_jobs=4)
        clf_grid.fit(sel_X_train,y_train)
        print(clf_grid.best_params_)
        print(clf_grid.best_score_)

        val_pre = clf_grid.predict(sel_X_val)
        acc = accuracy_score(r_y_val, val_pre)
        bal_acc = balanced_accuracy_score(r_y_val, val_pre)
        print('accuracy',acc)
        print('balanced accuracy',bal_acc)

if __name__=='__main__':
    xgb_para_tune(r_X_train,r_y_train,r_X_val)