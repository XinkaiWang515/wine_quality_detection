# from data import r_X_training, r_y_training, r_X_train, r_y_train, r_X_val, r_y_val
from data_adjust import r_X_training, r_y_training, r_X_train, r_y_train, r_X_val, r_y_val
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
import numpy as np
from feature_sel import feature_selection
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings

svc_param = {'kernel': ['rbf','sigmoid'],
             'C': [0.01,0.1,1,10],
             'gamma': ['auto','scale']}

# svc = SVC(probability=False)

def svc_para_tune(X_train,y_train,X_val):
    scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
    fea_num = [8, 9, 10, 11]
    max_bal_acc = 0

    for n in fea_num:
        etc_feat = feature_selection(X_train, y_train, n)[-1]
        sel_X_train = X_train.iloc[:, etc_feat]
        sel_X_val = X_val.iloc[:, etc_feat]
        svc = SVC(probability=False)
        clf_grid = GridSearchCV(estimator=svc,param_grid=svc_param,scoring=scoring,cv=5,refit='Accuracy',n_jobs=4)
        clf_grid.fit(sel_X_train,r_y_train)
        print(clf_grid.best_params_)
        print(clf_grid.best_score_)

        val_pre = clf_grid.predict(sel_X_val)
        acc = accuracy_score(r_y_val, val_pre)
        bal_acc = balanced_accuracy_score(r_y_val, val_pre)
        print('accuracy',acc)
        print('balanced accuracy',bal_acc)


if __name__=='__main__':
    svc_para_tune(r_X_train,r_y_train,r_X_val)