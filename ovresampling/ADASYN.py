import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state


class ADASYN(object):

    def __init__(self, bata, k_neighbors, random_state):
        """
        :param bata: a parameter used to specify the desired balance level after generation of synthetic data
        :param k_neighbors: Number of nearest neighbours to used to construct synthetic samples.
        :param random_state:
        """

        self.bata = bata
        self.k_neighbors = k_neighbors
        self.random_state = check_random_state(random_state)

    def fit_sample(self, X, y):
        """
        ADASYN
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
        # calculate the number of synthetic data examples that need to be generate for minority class
        gen = round((len(neg_data) - len(pos_data)) * self.bata)
        # Find k nearest neighbors based on the Euclidean distance in full train data
        nn_m = NearestNeighbors(n_neighbors=self.k_neighbors + 1)
        nn_m.fit(X)
        mm_index = nn_m.kneighbors(pos_data, return_distance=False)[:, 1:]
        # calculate the ratio
        ratio_nn = (np.sum(y[mm_index] != minority_target, axis=1) / (nn_m.n_neighbors - 1))
        # Normalize
        ratio_nn /= np.sum(ratio_nn)
        # Calculate the number of synthetic data examples that need to be generate for each minority example
        n_samples_generate = np.rint(ratio_nn * gen).astype(int)

        # the nearest neighbors need to be fitted only on the current class
        # to find the class NN to generate new samples
        nn_m.fit(pos_data)
        _, nn_index = nn_m.kneighbors(pos_data)
        ADASYNed = np.zeros(shape=(np.sum(n_samples_generate), X.shape[1]))
        newindex = 0
        for x_i, x_i_nn, num_sample_i in zip(pos_data, nn_index, n_samples_generate):
            nn_zs = self.random_state.randint(1, high=nn_m.n_neighbors, size=num_sample_i)
            steps = self.random_state.uniform(size=len(nn_zs))
            for step, nn_z in zip(steps, nn_zs):
                ADASYNed[newindex] = (x_i + step * (pos_data[x_i_nn[nn_z], :] - x_i))
                newindex = newindex + 1
        y_new = np.array([minority_target] * np.sum(n_samples_generate))
        # combine
        X_resampled = np.vstack((X, ADASYNed))
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


