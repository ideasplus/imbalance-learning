import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


class BorderSMOTE(object):

    def __init__(self, N, k_neighbors, m_neighbors, random_state, kind):
        """
        :param N: Amount of SMOTE N%
        :param k_neighbors: Number of nearest neighbors k for SMOTE
        :param m_neighbors: Number of nearest neighbors m for filter positive samples
        :param random_state:
        :param kind:
        """
        self.N = N
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.random_state = random_state
        np.random.seed(seed=self.random_state)
        self.kind = kind

    def _make_samples(self, ratio, danger, target):
        # shape
        T, n_features = danger.shape
        # KNN model for smote
        nn_k = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_k.fit(target)
        # synthetic samples
        SMOTEd = np.zeros(shape=(self.N * T, n_features))
        newindex = 0
        for i in range(T):
            N_tmp = ratio
            knn_index = nn_k.kneighbors(danger[i].reshape(1, -1), return_distance=False)[:, 1:]  # find k neighbors
            while N_tmp != 0:
                # choose a random number between 1 and k
                nn_index = np.random.choice(knn_index[0])
                diff = target[nn_index] - danger[i]
                gap = np.random.random()
                SMOTEd[newindex] = danger[i] + gap * diff
                newindex = newindex + 1
                N_tmp = N_tmp - 1
        return SMOTEd

    def fit_sample(self, X, y):
        # determine the minority class label
        stats_c_ = Counter(y)
        minority_target = min(stats_c_, key=stats_c_.get)
        majority_target = max(stats_c_, key=stats_c_.get)
        # divide train data into positive and negative data
        pos_data = X[y == minority_target]
        neg_data = X[y == majority_target]

        # Step1:we calculate minority m nearest neighbors from the whole training set T.
        nn_m = NearestNeighbors(n_neighbors=self.m_neighbors + 1)
        nn_m.fit(X)
        index_m = nn_m.kneighbors(pos_data, return_distance=False)[:, 1:]
        # Step2:we find DANGER set where m/2 =<n_maj< m.
        mm_label = (y[index_m] != minority_target).astype(int)
        n_maj = np.sum(mm_label, axis=1)
        danger_index = np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2,
                                      n_maj < nn_m.n_neighbors - 1)
        danger = pos_data[danger_index]
        # print(len(danger))

        # If N is less than 100%, randomize the minority class samples as only a random percent of them will be SMOTEd
        if self.N < 100:
            T = round((self.N / 100) * len(danger))
            self.N = 100
            shuffle_index = np.random.permutation(len(danger))
            danger = danger[shuffle_index]
        self.N = int(self.N / 100)  # The amount of SMOTE is assumed to be integral multiples of 100

        if self.kind == 'borderline1':
            X_syn = self._make_samples(self.N, danger, pos_data)
            y_syn = np.array([minority_target] * len(X_syn))
            # combine
            X_resampled = np.vstack((X, X_syn))
            y_resampled = np.hstack((y, y_syn))
            return X_resampled, y_resampled
        elif self.kind == 'borderline2':
            random_state = check_random_state(self.random_state)
            fractions = random_state.beta(10, 10)
            pos_gen = round(self.N * fractions)
            neg_gen = self.N - pos_gen
            # SMOTE for minority
            SMOTEd_min = self._make_samples(pos_gen, danger, pos_data)
            # SMOTE for majority
            SMOTEd_maj = self._make_samples(neg_gen, danger, neg_data)
            X_syn = np.vstack([SMOTEd_min, SMOTEd_maj])
            y_syn = np.array([minority_target] * len(X_syn))
            # combine
            X_resampled = np.vstack((X, X_syn))
            y_resampled = np.hstack((y, y_syn))
            return X_resampled, y_resampled
        else:
            pass


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
    dataset = fetch_datasets()['oil']
    X = dataset.data
    y = dataset.target
    # print(Counter(y))

    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    dic = {'recall': [], 'precision': [], 'f1': [], 'auc': [], 'gmean': []}
    results = prettytable.PrettyTable(["Classifier", "Precision", 'Recall', 'F-measure', 'AUC', 'G-mean'])
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 训练
        sb = BorderSMOTE(N=100, m_neighbors=9, k_neighbors=5, random_state=42, kind='borderline1')
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
