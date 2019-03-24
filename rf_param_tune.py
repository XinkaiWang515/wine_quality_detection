# from data import r_X_train, r_y_train, r_X_val, r_y_val, scal_X_train, scal_X_val
from data_adjust import r_X_train, r_y_train, r_X_val, r_y_val, scal_X_train, scal_X_val
from smote import X_res,y_res
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
import numpy as np
from feature_sel import feature_selection
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

n_estimators = [x for x in range(200,1100,100)]
max_depth = [x for x in range(10,70,10)]
min_samples_split = [2,5,10]
min_samples_leaf = [1,2,4]
max_features = ['auto', 'sqrt']
class_weight = ['balanced']

# rand_search_param = {'n_estimators':n_estimators,
#                     'max_features': max_features,
#                     'max_depth': max_depth,
#                     'min_samples_split': min_samples_split,
#                     'min_samples_leaf': min_samples_leaf,
#                     'class_weight':class_weight
#                     }
#

def rf_para_tune(X_train,y_train,X_val):
    scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
    grid_param = {'n_estimators': [400, 500, 600],
                  'max_features': ['auto'],
                  'max_depth': [20],
                  'min_samples_split': [4,5],
                  'min_samples_leaf': [4],
                  'class_weight': class_weight
                  }
    fea_num = [8, 9, 10, 11]
    max_bal_acc = 0
    max_accu = 0
    # for n in fea_num:
    #     etc_feat = feature_selection(X_train, y_train, n)[-1]
    #     sel_X_train = X_train.iloc[:, etc_feat]
    #     sel_X_val = X_val.iloc[:, etc_feat]
    #     rf = RandomForestClassifier(random_state=0)
    #
    #     clf_grid = GridSearchCV(estimator=rf,param_grid=grid_param,cv=5,n_jobs=-1)
    #     clf_grid.fit(sel_X_train,y_train)
    #     y_pred = clf_grid.predict(sel_X_val)
    #     # print(clf_grid.best_params_)
    #     tr_accu = accuracy_score(r_y_val, y_pred)
    #     if tr_accu>max_accu:
    #         max_accu = tr_accu
    #         rf_best_param = clf_grid.best_params_
    #         rf_best_param['fea_num'] = n
    #         bal_acc = balanced_accuracy_score(r_y_val, y_pred)
    #
    # return rf_best_param,bal_acc, max_accu

    for n in fea_num:
        etc_feat = feature_selection(X_train, y_train, n)[-1]
        sel_X_train = X_train.iloc[:, etc_feat]
        sel_X_val = X_val.iloc[:, etc_feat]
        rf = RandomForestClassifier(random_state=0)

        # clf = RandomizedSearchCV(estimator=rf,param_distributions=rand_search_param,scoring=scoring,n_iter=10,cv=5,refit='Accuracy')
        # clf.fit(r_X_train,r_y_train)
        # y_pred = clf.predict(r_X_val)
        # print(clf.best_params_)
        # print(balanced_accuracy_score(r_y_val,y_pred))

        clf_grid = GridSearchCV(estimator=rf,param_grid=grid_param,scoring=scoring,cv=5,refit='Accuracy',n_jobs=4)
        clf_grid.fit(sel_X_train,y_train)
        y_pred = clf_grid.predict(sel_X_val)
        # print(clf_grid.best_params_)
        bal_acc = balanced_accuracy_score(r_y_val, y_pred)
        if bal_acc>max_bal_acc:
            max_bal_acc = bal_acc
            rf_best_param = clf_grid.best_params_
            rf_best_param['fea_num'] = n
            tr_accu = accuracy_score(r_y_val, y_pred)

    return rf_best_param,max_bal_acc, tr_accu

if __name__=="__main__":
    rf_best_param, bal_acc, tr_accu = rf_para_tune(X_res,y_res,r_X_val)
    print(rf_best_param)
    print(bal_acc)
