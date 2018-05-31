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


# dic = {'recall': {'CART': [], 'SMOTE': {100:[1, 0, ]}, 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
#        'precision': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
#        'f1': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
#        'auc': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []},
#        'gmean': {'CART': [], 'SMOTE': [], 'Border1': [], 'Border2': [], 'ADASYN': [], 'Safe-level': []}}
dic = {}

results = prettytable.PrettyTable(["Classifier", "Ratio", "Precision", 'Recall', 'F-measure', 'AUC', 'G-mean'])


def templet(sampler_name, sample_ratio):
    """
    模板方法
    :param sampler_name: 采样算法名
    :param sample_ratio: 采样比例
    :return:
    """
    dataset = fetch_datasets()['satimage']
    X = dataset.data
    y = dataset.target
    # 起始时间
    start_time = time.time()
    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        sb = None
        if sampler_name == 'CART':
            sb = DummySampler()
        elif sampler_name == 'SMOTE':
            sb = SMOTE(N=sample_ratio, k_neighbors=5, random_state=42)
        elif sampler_name == 'Border1':
            sb = BorderSMOTE(N=sample_ratio, m_neighbors=9, k_neighbors=5, random_state=42, kind='borderline1')
        elif sampler_name == 'Border2':
            sb = BorderSMOTE(N=sample_ratio, m_neighbors=9, k_neighbors=5, random_state=42, kind='borderline2')
        elif sampler_name == 'ADASYN':
            sb = ADASYN(bata=sample_ratio, k_neighbors=5, random_state=42)
        elif sampler_name == 'Safe-level':
            sb = SafeLevelSMOTE(N=sample_ratio, k_neighbors=5, random_state=42)
        else:
            pass
        X_res, y_res = sb.fit_sample(X_train_minmax, y[train])  # 采样
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
        # write2dic
        fill_dic('precision', sampler_name, sample_ratio, precision)
        fill_dic('recall', sampler_name, sample_ratio, recall)
        fill_dic('f1', sampler_name, sample_ratio, f1)
        fill_dic('auc', sampler_name, sample_ratio, auc)
        fill_dic('gmean', sampler_name, sample_ratio, gmean)
    print('%s %.1f building id transforming took %fs!' % (sampler_name, sample_ratio, time.time() - start_time))


def fill_dic(measure_name, sampler_name, sample_ratio, value):
    """
    填充字典
    :param measure_name: 评价指标
    :param sampler_name: 采样算法
    :param sample_ratio: 采样比例
    :param value: 指标
    :return:
    """
    if measure_name in dic.keys():
        if sampler_name in dic[measure_name].keys():
            if sample_ratio in dic[measure_name][sampler_name].keys():
                dic[measure_name][sampler_name][sample_ratio].append(value)
            else:
                dic[measure_name][sampler_name][sample_ratio] = [value]
        else:
            dic[measure_name][sampler_name] = {sample_ratio: [value]}
    else:
        dic[measure_name] = {sampler_name: {sample_ratio: [value]}}


def write2row(sampler_name, sample_ratio):
    """
    :param sampler_name: 采样方法
    :param sample_ratio: 采样比例
    :return:
    """
    results.add_row([sampler_name, sample_ratio,
                     np.mean(np.array(dic['precision'][sampler_name][sample_ratio])),
                     np.mean(np.array(dic['recall'][sampler_name][sample_ratio])),
                     np.mean(np.array(dic['f1'][sampler_name][sample_ratio])),
                     np.mean(np.array(dic['auc'][sampler_name][sample_ratio])),
                     np.mean(np.array(dic['gmean'][sampler_name][sample_ratio]))])


record = {'CART': [0], 'SMOTE': [100, 200, 300, 400, 500],
          'Border1': [100, 200, 300, 400, 500], 'Border2': [100, 200, 300, 400, 500],
          'ADASYN': [0.1, 0.3, 0.5, 0.7, 0.9], 'Safe-level': [100, 200, 300, 400, 500]}

# 输出
for key, value in record.items():
    for ratio in value:
        templet(key, ratio)
        write2row(key, ratio)

print(results)


