import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors


class SMOTE(object):

    def __init__(self, N, k_neighbors=5, random_state=None):
        """
        初始化
        :param N: Amount of SMOTE N%
        :param k_neighbors: Number of nearest neighbors k
        :param random_state: seed
        """
        self.N = N
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        np.random.seed(seed=self.random_state)

    def fit_sample(self, X, y):
        """
        Train model based on input data.
        :param X: array-like, shape = [n__samples, n_features]
        :return: X_res, y_res
        """
        # determine the minority class label
        stats_c_ = Counter(y)
        minority_target = min(stats_c_, key=stats_c_.get)
        majority_target = max(stats_c_, key=stats_c_.get)
        # divide train data into positive and negative data
        pos_data = X[y == minority_target]
        neg_data = X[y == majority_target]
        # shape
        T, n_features = pos_data.shape
        # If N is less than 100%, randomize the minority class samples as only a random percent of them will be SMOTEd
        if self.N < 100:
            T = round((self.N / 100) * T)
            self.N = 100
            shuffle_index = np.random.permutation(len(pos_data))
            pos_data = pos_data[shuffle_index]

        self.N = int(self.N / 100)  # The amount of SMOTE is assumed to be integral multiples of 100

        # KNN model
        nn_k = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_k.fit(pos_data)

        SMOTEd = np.zeros(shape=(self.N * T, n_features))
        newindex = 0
        for i in range(T):
            N_tmp = self.N
            knn_index = nn_k.kneighbors(pos_data[i].reshape(1, -1), return_distance=False)[:, 1:]  # find k neighbors
            while N_tmp != 0:
                # choose a random number between 1 and k
                nn_index = np.random.choice(knn_index[0])
                diff = pos_data[nn_index] - pos_data[i]
                gap = np.random.random()
                SMOTEd[newindex] = pos_data[i] + gap * diff
                newindex = newindex + 1
                N_tmp = N_tmp - 1
        y_new = np.array([minority_target] * len(SMOTEd))
        # combine
        X_resampled = np.vstack((X, SMOTEd))
        y_resampled = np.hstack((y, y_new))
        return X_resampled, y_resampled


if __name__ == '__main__':
    import time
    import prettytable
    from collections import Counter
    from sklearn import svm
    from sklearn import tree
    from sklearn import metrics
    from sklearn import preprocessing
    from imblearn.datasets import fetch_datasets
    from imblearn.metrics import geometric_mean_score
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

    start_time = time.time()
    dataset = fetch_datasets()['oil']
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
        sb = SMOTE(N=100, k_neighbors=5, random_state=42)
        # 预测
        # X_res, y_res = sb.fit_sample(X_train_minmax, y[train])

        # model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
        model = svm.SVC(class_weight={1: 20})
        model.fit(X_train_minmax, y[train])
        predict = model.predict(X_test_minmax)
        # probability = model.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], predict)
        recall = metrics.recall_score(y[test], predict)
        if precision == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        # auc = metrics.roc_auc_score(y[test], probability)
        gmean = geometric_mean_score(y[test], predict)
        dic['precision'].append(precision)
        dic['recall'].append(recall)
        dic['f1'].append(f1)
        dic['auc'].append(1)
        dic['gmean'].append(gmean)

    results.add_row(['BorderSmote',
                     np.mean(np.array(dic['precision'])),
                     np.mean(np.array(dic['recall'])),
                     np.mean(np.array(dic['f1'])),
                     np.mean(np.array(dic['auc'])),
                     np.mean(np.array(dic['gmean']))])
    print(results)
    print('BorderSmote building id transforming took %fs!' % (time.time() - start_time))
