import time
import numpy as np
from collections import Counter
from sklearn import preprocessing
from imblearn.datasets import fetch_datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV


class CGMOS(object):

    def __init__(self, ratio, sigmafactor, random_state):
        self.ratio = ratio
        self.sigmafactor = sigmafactor
        self.random_state = random_state
        self.minority_target = None
        self.majority_target = None
        np.random.seed(seed=self.random_state)

    def fit_sample(self, X, y):
        """
        Train model based on input data.
        :param X: array-like, shape = [n__samples, n_features]
        :return: X_res, y_res
        """
        stats_c_ = Counter(y)
        self.minority_target = min(stats_c_, key=stats_c_.get)
        self.majority_target = max(stats_c_, key=stats_c_.get)
        X_res, y_res = self._upsampling(X, y, self.sigmafactor)
        return X_res, y_res

    def _upsampling(self, X, y, sigmafactor):
        """
        :param X:
        :param y:
        :param sigmafactor:
        :return:
        """
        k = 10
        negative = X[y == self.majority_target]
        positive = X[y == self.minority_target]
        size0 = len(negative)
        size1 = len(positive)
        newDataNum = round(abs(size0 - size1) * self.ratio)
        # print(newDataNum)
        nbrs0 = NearestNeighbors(n_neighbors=k).fit(negative)
        distances0, indices0 = nbrs0.kneighbors(negative)
        nbrs1 = NearestNeighbors(n_neighbors=k).fit(positive)
        distances1, indices1 = nbrs1.kneighbors(positive)
        gsigma0 = np.mean(distances0, 1) * sigmafactor
        gsigma1 = np.mean(distances1, 1) * sigmafactor
        # negative
        pdf0from0, density0from0 = self.get_pdf_of_points(gsigma0, negative, negative)
        pdf0from1, density0from1 = self.get_pdf_of_points(gsigma1, positive, negative)
        # positive
        pdf1from1, density1from1 = self.get_pdf_of_points(gsigma1, positive, positive)
        # print(pdf1from1)
        pdf1from0, density1from0 = self.get_pdf_of_points(gsigma0, negative, positive)
        # Calculate Posterior Probability
        confidence0 = self.get_confidence(pdf0from0, pdf0from1, size0, size1)
        confidence1 = self.get_confidence(pdf1from1, pdf1from0, size1, size0)

        # search for seed in negative data
        # Compute confidence ratio for negative data upon new data added to negative data
        pdf0from0_mat = np.tile(pdf0from0.reshape(-1, 1), (1, size0))
        pdf0from0_mat = (pdf0from0_mat * size0 + density0from0) / (size0 + 1)
        pdf0from0_mat = np.r_[pdf0from0_mat, np.diag(pdf0from0_mat).reshape(1, -1)]

        pdf0from1_mat = np.tile(pdf0from1.reshape(-1, 1), (1, size0))
        pdf0from1_mat = np.r_[pdf0from1_mat, np.diag(pdf0from1_mat).reshape(1, -1)]

        confidence0_new = self.get_confidence(pdf0from0_mat, pdf0from1_mat, size0 + 1, size1)
        # Compute confidence ratio for positive data upon new data added to negative data
        pdf1from0_mat = np.tile(pdf1from0.reshape(-1, 1), (1, size0))
        pdf1from0_mat = (pdf1from0_mat * size0 + density1from0) / (size0 + 1)

        pdf1from1_mat = np.tile(pdf1from1.reshape(-1, 1), (1, size0))
        confidence1_new = self.get_confidence(pdf1from1_mat, pdf1from0_mat, size1, size0 + 1)

        confidence_new_0 = np.r_[confidence0_new, confidence1_new]
        confidence_old_0 = np.concatenate([np.r_[np.tile(confidence0.reshape(-1, 1), (1, size0)),
                                                 confidence0.reshape(1, -1)],
                                           np.tile(confidence1.reshape(-1, 1), (1, size0))], axis=0)
        # Relative Certainty Change
        confidence0_ratio = (confidence_new_0 - confidence_old_0) / confidence_old_0
        confidence0_ratio = 0.5 * (np.mean(confidence0_ratio[0:size0 + 1, :], axis=0) +
                                   np.mean(confidence0_ratio[size0 + 1:, :], axis=0))
        # Search for seed in positive data
        # Compute confidence ratio for positive data upon new data added to positive data
        pdf1from1_mat = np.tile(pdf1from1.reshape(-1, 1), (1, size1))
        pdf1from1_mat = (pdf1from1_mat * size1 + density1from1) / (size1 + 1)
        pdf1from1_mat = np.r_[pdf1from1_mat, np.diag(pdf1from1_mat).reshape(1, -1)]

        pdf1from0_mat = np.tile(pdf1from0.reshape(-1, 1), (1, size1))
        pdf1from0_mat = np.r_[pdf1from0_mat, np.diag(pdf1from0_mat).reshape(1, -1)]

        confidence1_new = self.get_confidence(pdf1from1_mat, pdf1from0_mat, size1 + 1, size0)
        # Compute confidence ratio for negative data upon new data added to positive data
        pdf0from1_mat = np.tile(pdf0from1.reshape(-1, 1), (1, size1))
        pdf0from1_mat = (pdf0from1_mat * size1 + density0from1) / (size1 + 1)

        pdf0from0_mat = np.tile(pdf0from0.reshape(-1, 1), (1, size1))
        confidence0_new = self.get_confidence(pdf0from0_mat, pdf0from1_mat, size0, size1 + 1)

        confidence_new_1 = np.r_[confidence0_new, confidence1_new]
        confidence_old_1 = np.concatenate([np.tile(confidence0.reshape(-1, 1), (1, size1)),
                                           np.r_[np.tile(confidence1.reshape(-1, 1), (1, size1)),
                                                 confidence1.reshape(1, -1)]], axis=0)
        # Relative Certainty Change
        confidence1_ratio = (confidence_new_1 - confidence_old_1) / confidence_old_1
        confidence1_ratio = 0.5 * (np.mean(confidence1_ratio[0:size0, :], axis=0) +
                                   np.mean(confidence1_ratio[size0:, :], axis=0))

        confidence = np.append(confidence0_ratio, confidence1_ratio)

        X_resampled, y_resampled = self.getNewDataByInterpolationRandomSimplex3(X, y, gsigma0, gsigma1, confidence,
                                                                                newDataNum)

        return X_resampled, y_resampled

    def getNewDataByInterpolationRandomSimplex3(self, X, y, gsigma0, gsigma1, cofidence, newDataNum):
        """
        生成样本
        :param X: training data
        :param y: label
        :param gsigma0: Sigma factor for Majority class
        :param gsigma1: Sigma factor for Minority class
        :param cofidence: Posterior probability
        :param newDataNum: The number of synthetic data
        :return:
        """
        negative = X[y == self.majority_target]
        positive = X[y == self.minority_target]
        size0 = len(negative)
        size1 = len(positive)
        k = min(5, size1 - 1)
        expansionRate = 1.0
        # normalizition
        cofidence = (cofidence - min(cofidence)) / (max(cofidence) - min(cofidence))
        # synthetic
        cofidence1 = cofidence[size0:]
        cofidence1 = (cofidence1 - min(cofidence1)) / (max(cofidence1) - min(cofidence1))
        cofidence1 = cofidence1 / sum(cofidence1)
        # weighted sampling
        maxConffidenceIdx = np.random.choice(size1, newDataNum, p=cofidence1)
        poiPos = positive[maxConffidenceIdx]
        posSigma = gsigma1[maxConffidenceIdx]
        nbrs1 = NearestNeighbors(n_neighbors=k + 1).fit(positive)
        distances1, indices1 = nbrs1.kneighbors(poiPos)

        newDataPos = np.zeros(shape=(newDataNum, positive.shape[1]))
        newSigma = np.zeros(shape=(newDataNum, 1))
        for i in range(len(indices1)):
            nbs = indices1[i]
            rperm = [0, np.random.randint(low=1, high=len(nbs))]
            nbCurPos = positive[nbs[rperm]]
            nbCurSigma = gsigma1[nbs[rperm]]
            w = [0, np.random.rand(1)[0], 1]
            w = sorted(w)
            tmpw1 = w[1:]
            tmpw2 = w[0:-1]
            w = np.array(tmpw1) - np.array(tmpw2)
            meanPos = np.mean(nbCurPos, axis=0)
            meanPosMat = np.tile(meanPos, (len(nbCurPos), 1))
            nbCurPos = nbCurPos - meanPosMat
            wmat = np.tile(w.reshape(-1, 1), (1, nbCurPos.shape[1]))
            newCurDataPos = np.sum(nbCurPos * wmat) * expansionRate + meanPos
            newCurSigma = sum(nbCurSigma * w)
            newDataPos[i] = newCurDataPos
            newSigma[i] = newCurSigma

        y_new = np.array([self.minority_target] * newDataNum)
        X_resampled = np.vstack((X, newDataPos))
        y_resampled = np.hstack((y, y_new))

        return X_resampled, y_resampled

    @staticmethod
    def get_pdf_of_points(src_sigma, source_points, target_points):
        """
        PDF Estimation(gauss kernel)
        :param src_sigma: Bandwidth factor for each sample
        :param source_points: Source the samples which usesd to estimate the dendity
        :param target_points: Target the samples which need to estimate the density
        :return: pdf_of_points: The pdf estimate of target samples
        :return: density: The pdf estimate matrix
        """
        lambda_factor = 1e-6
        src_sigma[src_sigma == 0] = lambda_factor
        # 高斯核函数前缀
        prefix_gauss = 1 / np.sqrt(2 * np.pi * np.square(src_sigma))
        # 将前缀沿行方向复制100次
        prefix_gauss_mat = np.tile(prefix_gauss, (len(target_points), 1))
        # ???
        src_sigma_mat = np.tile(src_sigma, (len(target_points), 1))
        source_poins_size = len(source_points)
        tar_tar = np.sum(np.square(target_points), 1).reshape(-1, 1)
        src_src = np.sum(np.square(source_points), 1).reshape(1, -1)
        difference = tar_tar + src_src - 2 * np.dot(target_points, source_points.T)
        difference[difference < 0] = 0
        distance = np.sqrt(difference)
        # print(distance.shape)
        # print(distance)
        density = prefix_gauss_mat * np.exp(-np.square(distance) / (2 * np.square(src_sigma_mat)))
        density[density == 0] = lambda_factor
        pdf_of_points = np.sum(density, 1) / source_poins_size
        return pdf_of_points, density

    @staticmethod
    def get_confidence(pdf_to_self, pdf_to_other, self_size, other_size):
        """
        Get posterior probability
        :param pdf_to_self: the pdf estimation by self class
        :param pdf_to_other: the pdf estimation by contrary class
        :param self_size: the number of samples of self class
        :param other_size: the number of samples of contrary class
        :return:
        """
        confidence = np.log(1 + (pdf_to_self / pdf_to_other) * (1.0 * self_size / other_size))
        return confidence


def main():
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
        # preprocessing
        scaler = preprocessing.MinMaxScaler().fit(X[train])
        X_train_minmax = scaler.transform(X[train])
        X_test_minmax = scaler.transform(X[test])
        # training
        sb = CGMOS(ratio=0.5, sigmafactor=1, random_state=42)
        # testing
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

    results.add_row(['CGMOS',
                     np.mean(np.array(dic['precision'])),
                     np.mean(np.array(dic['recall'])),
                     np.mean(np.array(dic['f1'])),
                     np.mean(np.array(dic['auc'])),
                     np.mean(np.array(dic['gmean']))])
    print(results)
    print('CGMOS building id transforming took %fs!' % (time.time() - start_time))


if __name__ == '__main__':
    main()
