import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


class SafeLevelSMOTE(object):

    def __init__(self, N, k_neighbors, random_state):
        """
        :param N: Amount of SMOTE N%
        :param k_neighbors:Number of nearest neighbors k for SAFE
        :param random_state:
        """
        self.N = N
        self.k_neighbors = k_neighbors
        self.random_state = check_random_state(random_state)

    def fit_sample(self, X, y):
        """
        Safe-level SMOTE
        :param X: full train data
        :param y: label
        :return: synthetic samples
        """
        # determine the minority class label
        stats_c_ = Counter(y)
        minority_target = min(stats_c_, key=stats_c_.get)
        majority_target = max(stats_c_, key=stats_c_.get)
        # divide train data into positive and negative data
        pos_data = X[y == minority_target]
        neg_data = X[y == majority_target]

        self.N = int(self.N / 100)  # The amount of SMOTE is assumed to be integral multiples of 100

        # Find k nearest neighbors based on the Euclidean distance in full train data
        nn_m = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_m.fit(X)
        # Find k nearest neighbors based on the Euclidean distance in positive data
        nn_k = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_k.fit(pos_data)

        syn = []
        for bout in range(self.N):
            for i in range(len(pos_data)):
                gap = 0
                # compute k nearest neighbours for p in positive data
                pos_index = nn_k.kneighbors(pos_data[i].reshape(1, -1), return_distance=False)[:, 1:]
                # randomly select one from the k nearest neighbours call it n
                n_index = self.random_state.choice(pos_index[0])
                # calculate the safe level for p in train set
                train_index_p = nn_m.kneighbors(pos_data[i].reshape(1, -1), return_distance=False)[:, 1:]
                safe_level_p = np.sum((y[train_index_p] == minority_target).astype(int), axis=1)
                # calculate the safe level for n in train set
                train_index_n = nn_m.kneighbors(pos_data[n_index].reshape(1, -1), return_distance=False)[:, 1:]
                safe_level_n = np.sum((y[train_index_n] == minority_target).astype(int), axis=1)
                if safe_level_n != 0:
                    safe_level_ratio = safe_level_p / safe_level_n
                else:
                    safe_level_ratio = np.inf
                # the first case
                # p and n are noises
                if safe_level_ratio == np.inf and safe_level_p == 0:
                    pass
                # the second case
                # n is noise
                elif safe_level_ratio == np.inf and safe_level_p != 0:
                    gap = 0
                # the third case
                # p is as safe as n
                elif safe_level_ratio == 1:
                    gap = self.random_state.uniform()
                # the fourth case
                # the safe level of p is greater than n
                elif safe_level_ratio > 1:
                    gap = self.random_state.uniform(0, 1 / safe_level_ratio)
                elif safe_level_ratio < 1:
                    gap = self.random_state.uniform(1 - safe_level_ratio, 1)
                dif = pos_data[n_index] - pos_data[i]
                syn.append(pos_data[i] + gap * dif)
        X_new = np.array(syn)
        y_new = np.array([minority_target] * len(X_new))
        # combine
        X_resampled = np.vstack((X, X_new))
        y_resampled = np.hstack((y, y_new))
        return X_resampled, y_resampled


if __name__ == '__main__':
    import time
    import prettytable
    from collections import Counter
    from sklearn import tree
    from sklearn import metrics
    from sklearn import preprocessing
    from imblearn.datasets import fetch_datasets
    from imblearn.metrics import geometric_mean_score
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

    start_time = time.time()
    dataset = fetch_datasets()['satimage']
    X = dataset.data
    y = dataset.target
    print(Counter(y))

    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    dic = {'recall': [], 'precision': [], 'f1': [], 'auc': [], 'gmean': []}
    results = prettytable.PrettyTable(["Classifier", "Precision", 'Recall', 'F-measure', 'AUC', 'G-mean'])
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
        print(recall)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision'].append(precision)
        dic['recall'].append(recall)
        dic['f1'].append(f1)
        dic['auc'].append(auc)
        dic['gmean'].append(gmean)

    results.add_row(['BorderSmote',
                     np.mean(np.array(dic['precision'])),
                     np.mean(np.array(dic['recall'])),
                     np.mean(np.array(dic['f1'])),
                     np.mean(np.array(dic['auc'])),
                     np.mean(np.array(dic['gmean']))])
    print(results)
    print('BorderSmote building id transforming took %fs!' % (time.time() - start_time))
