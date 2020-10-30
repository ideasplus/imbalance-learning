import os
import csv
import time
import logging
import warnings
import numpy as np
import prettytable
import pandas as pd
from scipy import interp
from collections import Counter

from sklearn import tree
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

from imblearn.datasets import fetch_datasets
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import geometric_mean_score
from imblearn.ensemble import BalancedBaggingClassifier

from EasyEnsemble import *
from BalanceCascade import *
from SMOTEBoost import *
from RUSBoost import *

dic = {}
results = prettytable.PrettyTable(["Classifier", "Precision", 'Recall', 'F-measure', 'AUC', 'G-mean'])


def model(boosting_name, data_name, classifier_name, cv_name, mode):
    """
    模板方法
    :param boosting_name: 集成学习的方法
    :param data_name: 数据集名称
    :param classifier_name: 使用的基分类器
    :param cv_name: 交叉验证模式
    :param mode: 采样模式
    :return:
    """
    # 加载数据
    if data_name in fetch_datasets().keys():
        dataset = fetch_datasets()[data_name]
        X = dataset.data
        y = dataset.target
        print(Counter(y))
    else:
        # 加载自定义数据
        df = pd.read_csv('../imbalanced_data/%s.csv' % data_name, header=None)
        array = df.values.astype(float)
        X = array[:, 0:array.shape[1] - 1]
        y = array[:, -1]
        print(Counter(y))
    base = None
    if classifier_name == 'CART':
        base = tree.DecisionTreeClassifier(max_depth=8, random_state=42, min_samples_split=10)
    elif classifier_name == 'svm':
        base = svm.SVC()
    else:
        pass
    # 起始时间
    start_time = time.time()
    cv = None
    if cv_name == 'StratifiedKFold':
        cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    elif cv_name == 'RepeatedStratifiedKFold':
        cv = RepeatedStratifiedKFold(n_repeats=10, n_splits=10, random_state=42)
    else:
        pass
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)  # 插值点(保证每一折的fpr和tpr相同)
    aucs = []
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        classifier = None
        if boosting_name == 'CART':
            classifier = base
        elif boosting_name == 'Bagging':
            classifier = BaggingClassifier(base_estimator=base, n_estimators=40)
        elif boosting_name == 'BalancedBagging':
            classifier = BalancedBaggingClassifier(base_estimator=base,
                                                   ratio='auto',
                                                   replacement=True,
                                                   random_state=42)
        elif boosting_name == 'Adaboost':
            classifier = AdaBoostClassifier(base_estimator=base, n_estimators=40)
        elif boosting_name == 'Random Forest':
            classifier = RandomForestClassifier(max_depth=8, min_samples_split=10, n_estimators=40, random_state=42)
        elif boosting_name == 'EasyEnsemble':
            model_under(boosting_name, X_train_minmax, y[train], X_test_minmax, y[test])
            continue
        elif boosting_name == 'BalanceCascade':
            model_under(boosting_name, X_train_minmax, y[train], X_test_minmax, y[test])
            continue
        elif boosting_name == 'SMOTEBoost':
            classifier = SMOTEBoost(rate=100, n_estimators=40, weak_estimator=base,
                                    random_state=42, class_dist=False)
        elif boosting_name == 'RUSBoost':
            classifier = RUSBoost(ratio=50, n_estimators=40, weak_estimator=base,
                                  random_state=42, class_dist=False)
        else:
            pass
        classifier.fit(X_train_minmax, y[train])  # 采样
        predict = classifier.predict(X_test_minmax)
        probability = classifier.predict_proba(X_test_minmax)[:, 1]
        # 指标计算
        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        accuracy = metrics.accuracy_score(y[test], predict)
        # -------------step6.计算每一折的ROC曲线和PR曲线上的点 -------------
        fpr, tpr, thresholds = metrics.roc_curve(y[test], probability)
        # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0  # 为什么？
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        # write2dic
        fill_dic('precision', boosting_name, precision)
        fill_dic('recall', boosting_name, recall)
        fill_dic('f1', boosting_name, f1)
        fill_dic('auc', boosting_name, auc)
        fill_dic('gmean', boosting_name, gmean)

    if boosting_name != 'EasyEnsemble' and boosting_name != 'BalanceCascade':
        # 将frp和tpr写入文件
        # 在mean_fpr100个点，每个点处插值插值多次取平均
        mean_tpr /= cv.get_n_splits()
        # 坐标最后一个点为（1,1）
        mean_tpr[-1] = 1.0
        # 计算平均AUC值
        mean_auc = metrics.auc(mean_fpr, mean_tpr)

        # 将平均fpr和tpr拼接起来存入文件
        filename = './ROC/{data_name}/{mode}/{base_classifier}/{sampler}.csv'. \
            format(data_name=data_name, mode=mode, base_classifier=classifier_name, sampler=boosting_name)
        # 将文件路径分割出来
        file_dir = os.path.split(filename)[0]
        # 判断文件路径是否存在，如果不存在，则创建，此处是创建多级目录
        if not os.path.isdir(file_dir):
            os.makedirs(file_dir)
        # # 然后再判断文件是否存在，如果不存在，则创建
        # if not os.path.exists(filename):
        #     os.system(r'touch %s' % filename)
        # 将结果拼合起来
        all = np.c_[mean_fpr, mean_tpr]
        np.savetxt(filename, all, delimiter=',', fmt='%f')

    print('%s building id transforming took %fs!' % (boosting_name, time.time() - start_time))


def model_under(sampler_name, X_train, y_train, X_test, y_test):
    sb = None
    if sampler_name == 'EasyEnsemble':
        sb = EasyEnsemble(T=4, rounds=10)  # 初始化分类器
    elif sampler_name == 'BalanceCascade':
        sb = BalanceCascade(T=4, rounds=10)
    else:
        pass
    for i in range(1):
        sb.fit(X_train, y_train)  # 训练

        auc = sb.CalculateAUC(X_test, y_test)  # 测试
        predict = None
        if sampler_name == 'EasyEnsemble':
            predict = sb.predict(X_test)
        elif sampler_name == 'BalanceCascade':
            predict = sb.predict(X_test, sb.ensemble)
        precision, recall, f1, gmean = sb.ImbalanceEvaluate(y_test, predict, 1, 0)
        accuracy = metrics.accuracy_score(y_test, predict)
        # write2dic
        fill_dic('precision', sampler_name, precision)
        fill_dic('recall', sampler_name, recall)
        fill_dic('f1', sampler_name, f1)
        fill_dic('auc', sampler_name, auc)
        fill_dic('gmean', sampler_name, gmean)


def fill_dic(measure_name, sampler_name, value):
    """
    填充字典
    :param measure_name: 评价指标
    :param sampler_name: 采样算法
    :param value: 指标
    :return:
    """
    if measure_name in dic.keys():
        if sampler_name in dic[measure_name].keys():
            dic[measure_name][sampler_name].append(value)
        else:
            dic[measure_name][sampler_name] = [value]
    else:
        dic[measure_name] = {sampler_name: [value]}


def write2row(sampler_name):
    """
    :param sampler_name: 采样方法
    :return:
    """
    results.add_row([sampler_name,
                     np.mean(np.array(dic['precision'][sampler_name])),
                     np.mean(np.array(dic['recall'][sampler_name])),
                     np.mean(np.array(dic['f1'][sampler_name])),
                     np.mean(np.array(dic['auc'][sampler_name])),
                     np.mean(np.array(dic['gmean'][sampler_name]))])


def write2file(configure, data_name, classifier_name):
    """
    将分类指标写入csv文件，方便记录
    :param configure: 字典，使用的采样器及比例
    :param data_name: 数据集的名称
    :param classifier_name：基分类器名称
    :return:
    """
    file_header = ["Classifier", "Precision", 'Recall', 'F-measure', 'AUC', 'G-mean']
    # 路径
    filename = './metrics/{name}/{classifier}_boosting.csv'.format(name=data_name, classifier=classifier_name)
    # 将文件路径分割出来
    file_dir = os.path.split(filename)[0]
    # 判断文件路径是否存在，如果不存在，则创建，此处是创建多级目录
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    # 写入数据
    csvFile = open(filename, "w", newline='')
    writer = csv.writer(csvFile)
    # 写入第一行
    writer.writerow(file_header)

    for sampler_name in configure:
        # 写入指标
        tmp = [sampler_name,
               np.mean(np.array(dic['precision'][sampler_name])),
               np.mean(np.array(dic['recall'][sampler_name])),
               np.mean(np.array(dic['f1'][sampler_name])),
               np.mean(np.array(dic['auc'][sampler_name])),
               np.mean(np.array(dic['gmean'][sampler_name]))]
        writer.writerow(tmp)
    csvFile.close()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    record = ['Bagging', 'BalancedBagging', 'Random Forest', 'Adaboost',
              'EasyEnsemble', 'BalanceCascade', 'RUSBoost', 'SMOTEBoost']
    record_tmp = ['RUSBoost']
    # 随机种子，保证每次实验结果相同
    np.random.seed(42)
    for method in record:
        model(boosting_name=method, data_name='satimage', classifier_name='CART',
              cv_name='RepeatedStratifiedKFold', mode='boosting')
        write2row(method)
    print(results)
    # 后台输出
    write2file(record, 'satimage', 'CART')
