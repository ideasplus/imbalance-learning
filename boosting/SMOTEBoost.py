import logging
import numpy as np
from collections import Counter

from sklearn.base import clone
from sklearn import tree
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

from imblearn.metrics import geometric_mean_score

from SMOTE import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s : %(message)s', datefmt='%m%d%Y %I:%M%S %p ')


class SMOTE_(object):
    """Implementation of Synthetic Minority Over-Sampling Technique(SMOTE).

    SMOTE performs oversampling of the minority class by picking target
    minority class samples and their nearest minority class neighbours and
    generating new samples that linearly combine features of each target
    sample with features of its selected minority class neighbors [1].

    Parameters
    ----------
    k_neighbors: int, optional(default=5)
        Number of nearest neighbors
    random_state: int or None, optional(default=None)
        If int, random_state is the seed used by the random number generator
        If None, the random number generator is the RandomState instance used
        by np.random

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
       Synthetic Minority Over-Sampling Technique." Journal of Artificial
       Intelligence Research (JAIR), 2002.

    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

        self.X = None
        self.n_minority_samples = None
        self.n_features = None
        self.nn_k = None

    def sample(self, n_samples):
        """Generate samples

        Parameters
        ----------
        n_samples: int
            Number of new synthetic samples

        Returns
        -------
        S: array, shape = [n_samples, n_features]
            Return synthetic samples.
        """
        # np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])
            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.nn_k.kneighbors(self.X[j].reshape(1, -1),
                                      return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.

        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.nn_k = NearestNeighbors(n_neighbors=self.k + 1)
        self.nn_k.fit(self.X)

        return self


class SMOTEBoost(object):

    def __init__(self, rate, class_dist, k_neighbors=5,
                 weak_estimator=tree.DecisionTreeClassifier(max_depth=1),
                 n_estimators=10, random_state=None):
        """
        初始化
        :param rate:
        :param class_dist: true or false.
                           true indicates that the class distribution is maintained while doing weighted
                           resampling and before SMOTE is called at each iteration.
                           false indicates that the class distribution is not maintained while resampling.
        :param k_neighbors: SMOTE算法使用的k近邻数
        :param weak_estimator: Boost算法使用的弱分类器
        :param n_estimators: Boost算法使用的弱分类器数量
        :param random_state: 随机状态
        """
        self.rate = rate
        self.class_dist = class_dist
        self.k_neighbors = k_neighbors
        self.weak_estimator = weak_estimator
        self.n_estimator = n_estimators
        self.random_state = random_state
        np.random.seed(self.random_state)

        # self.smote = SMOTE_(k_neighbors=self.k_neighbors, random_state=self.random_state)
        self.majority_target = []
        self.minority_target = []

        self.pseudo_loss = []  # stores pseudo loss values
        self.estimator_weights_ = []  # stores (1/beta)values that is used as the weight of the hypothesis
        self.estimators_ = []  # stores hypothesis
        self.prediction = []

    def fit(self, X, y):
        # Determine the minority class label.
        stats_c_ = Counter(y)
        maj_c_ = max(stats_c_, key=stats_c_.get)
        self.majority_target = maj_c_
        min_c_ = min(stats_c_, key=stats_c_.get)
        self.minority_target = min_c_

        total_number = len(X)  # Total number of instances in the training set
        pos_data = X[y == self.minority_target]
        neg_data = X[y == self.majority_target]
        pos_size = len(pos_data)  # number of positive data
        neg_size = len(neg_data)  # number of negative data
        # Reorganize TRAIN by putting all the positive and negative exampels together, respectively
        X_train = np.vstack([pos_data, neg_data])
        y_train = np.array([self.minority_target] * pos_size + [self.majority_target] * neg_size)
        # weights stores the weights of the instances in each row for every iteration of boosting
        weights = np.zeros(shape=[self.n_estimator, X.shape[0]])
        # Weights for all the instances are initialized by 1/m for the first iteration
        weights[0] = 1 / X.shape[0]

        t = 0  # Loop counter
        count = 0  # Keeps counts of the number of times the same boosting iteration have been repeated
        while t < self.n_estimator:
            # log message
            # logger.debug('Boosting iteration # %d' % t)
            # print('Boosting iteration # %d' % t)
            if self.class_dist is True:
                # Resampling positive_data with weights of positive example
                sum_pos_weights = np.sum(weights[t][:pos_size])
                pos_weights = weights[t][:pos_size] / sum_pos_weights

                resample_pos = pos_data[np.random.choice(a=pos_size, size=pos_size,
                                                         replace=True, p=pos_weights)]

                # Resampling negative with weights of negative example
                sum_neg_weights = np.sum(weights[t][pos_size:total_number])
                neg_weights = weights[t][pos_size:total_number] / sum_neg_weights
                resample_neg = neg_data[np.random.choice(a=neg_size, size=neg_size,
                                                         replace=True, p=neg_weights)]
                # Resampled TRAIN is stored in RESAMPLED
                X_resampled = np.vstack([resample_pos, resample_neg])
                y_resampled = np.array([self.minority_target] * pos_size + [self.majority_target] * neg_size)

                # Calulating the number of boosting the positive class
                syn_size = pos_size * self.rate
            else:
                # indices of resampled train
                random_index = np.random.choice(a=total_number, size=total_number,
                                                replace=True, p=weights[t])
                # Resampled TRAIN is stored in RESAMPLED
                X_resampled = X_train[random_index]
                y_resampled = y_train[random_index]

                # Calulating the number of boosting the positive class
                pos_size = np.sum(y_resampled == self.minority_target)
                neg_size = np.sum(y_resampled == self.majority_target)
                syn_size = pos_size * self.rate

            # SMOTE step
            # self.smote.fit(X_resampled[y_resampled == self.minority_target])
            # X_syn = self.smote.sample(syn_size)
            # y_syn = np.array([self.minority_target] * syn_size)

            smote = SMOTE(N=self.rate, k_neighbors=5, random_state=self.random_state)
            X_res, y_res = smote.fit_sample(X_resampled, y_resampled)

            # train classifier
            model = clone(self.weak_estimator)
            # if self.weak_estimator == 'decision tree':
            #     model = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
            # elif self.weak_estimator == 'svm':
            #     model = svm.SVC(class_weight={1: 8})
            # else:
            #     pass
            model.fit(X_res, y_res)
            predict = model.predict(X_train)

            # Computing the pseudo loss of hypothesis 'model'
            incorrect = predict != y_train
            loss = np.mean(np.average(incorrect, weights=weights[t], axis=0))
            # print(loss)

            # If count exceeds a pre-defined threshold (5 in the current implementation),
            # the loop is broken and rolled back to the state where loss > 0.5 was not encountered
            if count > 5:
                self.pseudo_loss = self.pseudo_loss[:t]
                self.estimator_weights_ = self.estimator_weights_[:t]
                self.estimators_ = self.estimators_[:t]
                print('Too many iterations have loss > 0.5')
                print('Aborting boosting')
                break

            if loss > 0.5:
                count = count + 1
                continue
            else:
                count = 1

            self.pseudo_loss.append(loss)  # Pseudo-loss at each iteration
            self.estimators_.append(model)  # Hypothesis function
            beta = loss / (1 - loss)  # Setting weight update parameter 'beta'.
            self.estimator_weights_.append(np.log(1 / beta))  # Weight of the hypothesis

            # At the final iteration there is no need to update the weights any further
            if t == self.n_estimator - 1:
                break

            # Updating weight
            weights[t + 1][y_train == predict] = weights[t][y_train == predict] * beta
            weights[t + 1][y_train != predict] = weights[t][y_train != predict]

            # Normalizing the weight for the next iteration
            sum_weights = np.sum(weights[t + 1])
            weights[t + 1] /= sum_weights

            # Incrementing loop counter
            t = t + 1

    def predict(self, X):
        # Normalizing B
        # sum_B = np.sum(self.estimator_weights_)
        # self.estimator_weights_ /= sum_B

        prediction = np.zeros(shape=[len(X), 2])
        for i in range(len(X)):
            weight_pos = 0
            weigth_neg = 0
            for j in range(len(self.estimators_)):
                p = self.estimators_[j].predict(X[i].reshape(1, -1))
                if p == self.minority_target:
                    weight_pos += self.estimator_weights_[j]
                else:
                    weigth_neg += self.estimator_weights_[j]

            if weight_pos > weigth_neg:
                prediction[i] = [self.minority_target, weight_pos]
            else:
                prediction[i] = [self.majority_target, weight_pos]
        self.prediction = prediction

        return prediction[:, 0]

    def predict_proba(self, X):

        self.predict(X)

        probability = np.c_[1 - np.array(self.prediction[:, 1]), self.prediction[:, 1]]

        return probability


if __name__ == '__main__':

    import time
    import prettytable
    from collections import Counter
    from sklearn import metrics
    from sklearn import svm
    from sklearn import preprocessing
    from imblearn.datasets import fetch_datasets
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

    start_time = time.time()
    dataset = fetch_datasets()['satimage']
    X = dataset.data
    y = dataset.target

    # shuffle_indices = np.random.permutation(np.arange(len(X)))
    # X = X[shuffle_indices]
    # y = y[shuffle_indices]

    print(Counter(y))
    #
    # scaler = preprocessing.MinMaxScaler().fit(X)
    # X = scaler.transform(X)
    #
    # sb = SMOTEBoost(n_samples=626, weak_estimator='decision tree', random_state=42)
    # sb.fit(X, y, class_dist=False)
    # print(metrics.recall_score(y, sb.predict(X)))
    # print(metrics.classification_report(y, sb.predict(X)))

    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=10, random_state=42)
    dic = {'recall': [], 'precision': [], 'f1': [], 'auc': [], 'gmean': []}
    results = prettytable.PrettyTable(["Classifier", "Precision", 'Recall', 'F-measure', 'AUC', 'G-mean'])
    base = tree.DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42)
    for train, test in cv.split(X, y):
        # 预处理
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # 训练
        sb = SMOTEBoost(rate=100, n_estimators=40, weak_estimator=base, random_state=42, class_dist=False)
        sb.fit(X_train_minmax, y[train])
        # 预测
        predict = sb.predict(X_test_minmax)
        probability = sb.predict_proba(X_test_minmax)[:, 1]

        precision = metrics.precision_score(y[test], sb.predict(X_test_minmax))
        recall = metrics.recall_score(y[test], sb.predict(X_test_minmax))
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

    results.add_row(['SMOTEBoost',
                     np.mean(np.array(dic['precision'])),
                     np.mean(np.array(dic['recall'])),
                     np.mean(np.array(dic['f1'])),
                     np.mean(np.array(dic['auc'])),
                     np.mean(np.array(dic['gmean']))])
    print(results)
    print('SMOTEBoost building id transforming took %fs!' % (time.time() - start_time))
