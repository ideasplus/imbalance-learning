import time
import copy
import prettytable
from collections import Counter
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from imblearn.datasets import fetch_datasets
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from SMOTE import *
from BorderLine_SMOTE import *
from ADASYN import *
from SafelevelSMOTE import *


class DummySampler(object):

    @staticmethod
    def sample(X, y):
        return X, y

    def fit(self, X, y):
        return self

    def fit_sample(self, X, y):
        return self.sample(X, y)


def test():
    dic = {'recall': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
           'precision': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
           'f1': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
           'auc': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
           'gmean': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []}}

    results = prettytable.PrettyTable(["Classifier", "Precision", 'Recall', 'AUC', 'F-measure', 'G-mean'])

    # 加载数据
    dataset = fetch_datasets()['satimage']
    X = dataset.data
    y = dataset.target
    print(Counter(y))
    # 随机种子，保证每次实验结果相同
    np.random.seed(42)
    # -------------------------------------------CART----------------------------------------------------
    # 起始时间
    start_time = time.time()
    # 交叉验证CART
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # initialize CART
        cart = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        # 归一化
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 训练
        cart.fit(X_train_minmax, y[train])
        # 测试
        predict = cart.predict(X_test_minmax)
        probability = cart.predict_proba(X_test_minmax)

        cart_auc = metrics.roc_auc_score(y[test], probability[:, 1])
        cart_precision = metrics.precision_score(y[test], predict)
        cart_recall = metrics.recall_score(y[test], predict)
        if cart_precision == 0:
            cart_f1 = 0
        else:
            cart_f1 = 2 * (cart_precision * cart_recall) / (cart_precision + cart_recall)
        cart_gmean = geometric_mean_score(y[test], predict)
        dic['precision']['CART'].append(cart_precision)
        dic['recall']['CART'].append(cart_recall)
        dic['f1']['CART'].append(cart_f1)
        dic['auc']['CART'].append(cart_auc)
        dic['gmean']['CART'].append(cart_gmean)
    print('CART building id transforming took %fs!' % (time.time() - start_time))

    # ---------------------------------------------------SMOTE----------------------------------------------------------
    # 起始时间
    start_time = time.time()
    # 交叉验证
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # preprocess
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # initialize sampler
        sb = SMOTE(N=100, k_neighbors=5, random_state=42)
        # sampling
        X_res, y_res = sb.fit_sample(X_train_minmax, y[train])
        # initialize classifier
        model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        # model = svm.SVC(class_weight={1: 20})
        model.fit(X_res, y_res)
        predict = model.predict(X_test_minmax)
        probability = model.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision']['SMOTE'].append(precision)
        dic['recall']['SMOTE'].append(recall)
        dic['f1']['SMOTE'].append(f1)
        dic['auc']['SMOTE'].append(auc)
        dic['gmean']['SMOTE'].append(gmean)

    print('SMOTE building id transforming took %fs!' % (time.time() - start_time))

    # ---------------------------------------------Borderline-SMOTE1----------------------------------------------------
    # 起始时间
    start_time = time.time()
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 初始化采样器
        sb = BorderSMOTE(N=100, m_neighbors=30, k_neighbors=5, random_state=42, kind='borderline1')
        # 采样
        X_res, y_res = sb.fit_sample(X_train_minmax, y[train])

        model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        model.fit(X_res, y_res)
        predict = model.predict(X_test_minmax)
        probability = model.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision']['Border1'].append(precision)
        dic['recall']['Border1'].append(recall)
        dic['f1']['Border1'].append(f1)
        dic['auc']['Border1'].append(auc)
        dic['gmean']['Border1'].append(gmean)

    print('BorderSmote1 building id transforming took %fs!' % (time.time() - start_time))

    # ---------------------------------------------Borderline-SMOTE2----------------------------------------------------
    # 起始时间
    start_time = time.time()
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 初始化采样器
        sb = BorderSMOTE(N=100, m_neighbors=30, k_neighbors=5, random_state=42, kind='borderline2')
        # 采样
        X_res, y_res = sb.fit_sample(X_train_minmax, y[train])

        model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        model.fit(X_res, y_res)
        predict = model.predict(X_test_minmax)
        probability = model.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision']['Border2'].append(precision)
        dic['recall']['Border2'].append(recall)
        dic['f1']['Border2'].append(f1)
        dic['auc']['Border2'].append(auc)
        dic['gmean']['Border2'].append(gmean)

    print('BorderSmote2 building id transforming took %fs!' % (time.time() - start_time))

    # ---------------------------------------------ADASYN---------------------------------------------------------------
    # 起始时间
    start_time = time.time()
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 训练
        sb = ADASYN(bata=0.1, k_neighbors=5, random_state=42)
        # 预测
        X_res, y_res = sb.fit_sample(X_train_minmax, y[train])

        model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        model.fit(X_res, y_res)
        predict = model.predict(X_test_minmax)
        probability = model.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision']['ADASYN'].append(precision)
        dic['recall']['ADASYN'].append(recall)
        dic['f1']['ADASYN'].append(f1)
        dic['auc']['ADASYN'].append(auc)
        dic['gmean']['ADASYN'].append(gmean)

    print('ADASYN building id transforming took %fs!' % (time.time() - start_time))

    # ------------------------------------------------Safe-Level-SMOTE----------------------------------------------
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 训练
        sb = SafeLevelSMOTE(N=100, k_neighbors=5, random_state=42)
        # 预测
        X_res, y_res = sb.fit_sample(X_train_minmax, y[train])

        model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        model.fit(X_res, y_res)
        predict = model.predict(X_test_minmax)
        probability = model.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision']['Safe-level'].append(precision)
        dic['recall']['Safe-level'].append(recall)
        dic['f1']['Safe-level'].append(f1)
        dic['auc']['Safe-level'].append(auc)
        dic['gmean']['Safe-level'].append(gmean)

    print('Safe-level building id transforming took %fs!' % (time.time() - start_time))

    # display
    results.add_row(['CART',
                     np.mean(np.array(dic['precision']['CART'])),
                     np.mean(np.array(dic['recall']['CART'])),
                     np.mean(np.array(dic['auc']['CART'])),
                     np.mean(np.array(dic['f1']['CART'])),
                     np.mean(np.array(dic['gmean']['CART']))])
    results.add_row(['SMOTE',
                     np.mean(np.array(dic['precision']['SMOTE'])),
                     np.mean(np.array(dic['recall']['SMOTE'])),
                     np.mean(np.array(dic['auc']['SMOTE'])),
                     np.mean(np.array(dic['f1']['SMOTE'])),
                     np.mean(np.array(dic['gmean']['SMOTE']))])
    results.add_row(['Border1',
                     np.mean(np.array(dic['precision']['Border1'])),
                     np.mean(np.array(dic['recall']['Border1'])),
                     np.mean(np.array(dic['auc']['Border1'])),
                     np.mean(np.array(dic['f1']['Border1'])),
                     np.mean(np.array(dic['gmean']['Border1']))])
    results.add_row(['Border2',
                     np.mean(np.array(dic['precision']['Border2'])),
                     np.mean(np.array(dic['recall']['Border2'])),
                     np.mean(np.array(dic['auc']['Border2'])),
                     np.mean(np.array(dic['f1']['Border2'])),
                     np.mean(np.array(dic['gmean']['Border2']))])
    results.add_row(['ADASYN',
                     np.mean(np.array(dic['precision']['ADASYN'])),
                     np.mean(np.array(dic['recall']['ADASYN'])),
                     np.mean(np.array(dic['auc']['ADASYN'])),
                     np.mean(np.array(dic['f1']['ADASYN'])),
                     np.mean(np.array(dic['gmean']['ADASYN']))])
    results.add_row(['Safe-level',
                     np.mean(np.array(dic['precision']['Safe-level'])),
                     np.mean(np.array(dic['recall']['Safe-level'])),
                     np.mean(np.array(dic['auc']['Safe-level'])),
                     np.mean(np.array(dic['f1']['Safe-level'])),
                     np.mean(np.array(dic['gmean']['Safe-level']))])
    print(results)


if __name__ == '__main__':
    test()
