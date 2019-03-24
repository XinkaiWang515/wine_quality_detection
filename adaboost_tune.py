from data import r_X_training, r_y_training, r_X_train, r_y_train, r_X_val, r_y_val
# from data_adjust import r_X_training, r_y_training, r_X_train, r_y_train, r_X_val, r_y_val
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

dt = DecisionTreeClassifier(max_features='sqrt',
                            max_depth=10,min_samples_leaf=8,
                            min_samples_split=10,class_weight='balanced')
adab_param = {'base_estimator': [DecisionTreeClassifier(max_depth=5)],
                'n_estimators': [300,500,600],
                'learning_rate': [0.01,0.05,0.1],
                'algorithm': ['SAMME','SAMME.R']
                }
def ada_para_tune(X_train,y_train,X_val):
    # fea_num = [11]
    fea_num = [8,9,10,11]
    max_bal_acc = 0
    max_accu = 0
    # for n in fea_num:
    #     etc_feat = feature_selection(X_train, y_train, n)[-1]
    #     sel_X_train = X_train.iloc[:, etc_feat]
    #     sel_X_val = X_val.iloc[:, etc_feat]
    #     ada = AdaBoostClassifier(random_state=0)
    #     scoring = {'Accuracy':make_scorer(balanced_accuracy_score)}
    #     clf = GridSearchCV(estimator=ada,param_grid=adab_param,cv=5,n_jobs=-1)
    #     clf.fit(sel_X_train,r_y_train)
    #     y_pred = clf.predict(sel_X_val)
    #     tr_accu = accuracy_score(r_y_val, y_pred)
    #     if tr_accu>max_accu:
    #         max_accu = tr_accu
    #         ada_best_param = clf.best_params_
    #         ada_best_param['fea_num'] = n
    #         bal_acc_ada = balanced_accuracy_score(r_y_val, y_pred)
    #
    # return ada_best_param,bal_acc_ada, max_accu

    scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
    for n in fea_num:
        etc_feat = feature_selection(X_train, y_train, n)[-1]
        sel_X_train = X_train.iloc[:, etc_feat]
        sel_X_val = X_val.iloc[:, etc_feat]
        ada = AdaBoostClassifier(random_state=1)
        clf = GridSearchCV(estimator=ada,param_grid=adab_param,scoring=scoring,cv=5,refit='Accuracy',n_jobs=4)
        clf.fit(sel_X_train,y_train)
        # tr_pred = clf.predict(sel_X_train)
        # print('train accuracy:',accuracy_score(y_train,tr_pred))
        # print(clf.best_score_)

        y_pred = clf.predict(sel_X_val)
        bal_acc_ada = balanced_accuracy_score(r_y_val,y_pred)
        if bal_acc_ada>max_bal_acc:
            max_bal_acc = bal_acc_ada
            ada_best_param = clf.best_params_
            ada_best_param['fea_num'] = n
            tr_accu = accuracy_score(r_y_val,y_pred)

    return ada_best_param,max_bal_acc, tr_accu

if __name__=="__main__":
    ada_best_param, bal_acc, tr_accu = ada_para_tune(r_X_train,r_y_train,r_X_val)
    print(bal_acc)
    print(ada_best_param)