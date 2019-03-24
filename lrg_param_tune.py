from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, make_scorer
import numpy as np
from data import r_X_train, r_y_train, r_X_val, r_y_val, scal_X_train, scal_X_val
# from data_adjust import r_X_train, r_y_train, r_X_val, r_y_val, scal_X_train, scal_X_val
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from feature_sel import feature_selection
from smote import X_res,y_res

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def lrg_para_tune(X_train,y_train,X_val):
    fea_nums = [8,9,10,11]
    max_bal_accu = 0
    max_accu = 0
    c_range = np.logspace(-3, 3, 7)
    params = {'C': c_range, 'multi_class': ['multinomial'], 'solver': ['newton-cg']}
    scoring = {'Accuracy': make_scorer(balanced_accuracy_score)}
    for n in fea_nums:
        features = feature_selection(X_train,y_train,n)[1]
        sel_X_train=X_train.iloc[:,features]
        sel_X_val=X_val.iloc[:,features]

        # etc_feat = feature_selection(r_X_train, r_y_train, n)[-1]
        # sel_X_train = r_X_train.iloc[:, etc_feat]
        # sel_X_val = r_X_val.iloc[:, etc_feat]

        # clf = GridSearchCV(estimator=LogisticRegression(class_weight='balanced'), param_grid=params, cv=5)
        #
        # clf.fit(sel_X_train,y_train)
        # y_pred = clf.predict(sel_X_val)
        # accu = accuracy_score(r_y_val, y_pred)
        # # bal_accu = balanced_accuracy_score(r_y_val,y_pred)
        # if accu>max_accu:
        #     max_accu = accu
        #     bal_accu = balanced_accuracy_score(r_y_val, y_pred)
        #     best_param = {'solver':clf.best_estimator_.solver, 'C':clf.best_estimator_.C, 'fea_num':n}
        #
        # return best_param, max_accu, bal_accu

        clf = GridSearchCV(estimator=LogisticRegression(class_weight='balanced'), param_grid=params, scoring=scoring,
                           cv=5, refit='Accuracy')

        clf.fit(sel_X_train, y_train)
        # if n<11:
        #     print('for data with selected %d features:'% n)
        # elif n==11:
        #     print('for data with all %d features:' % n)
        # print('Best score:', clf.best_score_)
        # print('Best solver:', clf.best_estimator_.solver)
        # print('Best C:',clf.best_estimator_.C)
        y_pred = clf.predict(sel_X_val)
        bal_accu = balanced_accuracy_score(r_y_val, y_pred)
        if bal_accu > max_bal_accu:
            max_bal_accu = bal_accu
            accu = accuracy_score(r_y_val, y_pred)
            best_param = {'solver': clf.best_estimator_.solver, 'C': clf.best_estimator_.C, 'fea_num': n}

    return best_param,accu,max_bal_accu

if __name__=="__main__":
    best_param, accu, bal_accu = lrg_para_tune(scal_X_train,r_y_train,scal_X_val)
    print(best_param)
    print('accuracy:', accu)
    print('balanced accuracy:', bal_accu)

    # best_param,accu,max_bal_accu = lrg_para_tune(scal_X_train,r_y_train,scal_X_val)
    # print(best_param)
    # print('accuracy:',accu)
    # print('balanced accuracy:',max_bal_accu)


# print('Best score for data:', clf.best_score_)
# print('Best solver:', clf.best_estimator_.solver)
# print('Best C:',clf.best_estimator_.C)
# y_pred = clf.predict(r_X_val)
# print('balanced accuracy score is:',balanced_accuracy_score(r_y_val,y_pred))
# print('f1 score is:',f1_score(r_y_val,y_pred,average='weighted'))


# acc = []
# for i in c_range:
#     lrg = LogisticRegression(multi_class='multinomial', solver='newton-cg', C=i, random_state=0)
#     lrg.fit(r_X_train,r_y_train)
#     y_pred = lrg.predict(r_X_val)
#     acc_score = balanced_accuracy_score(r_y_val,y_pred)
#     acc.append(acc_score)
# print(acc)
