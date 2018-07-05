import time
import math
import bisect
import random
import logging
from collections import Counter

import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from imblearn.datasets import fetch_datasets

logger = logging.getLogger(__name__)
logging.basicConfig(filename='MWMOTE_log.txt', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s : %(message)s', datefmt='%m%d%Y %I:%M%S %p ')


class Knn:

    def __init__(self):
        self.data = []
        self.real_indices = []
        self.dic = {}
        self.distances = []
        self.indices = []

    def fit(self, data):
        self.data = data
        self.real_indices = range(len(data))
        nbrs = NearestNeighbors(n_neighbors=len(data)).fit(data)
        self.distances, self.indices = nbrs.kneighbors(data)
        # self.dic = {(i, j): self.distances[i][np.where(self.indices[i] == i)[0][0]]
        #             for i in range(len(data)) for j in range(len(data))}

    def fit_subset(self, indices):
        self.real_indices = indices

    def get_dis(self, a, b):
        return self.distances[a][np.where(self.indices[a] == b)[0][0]]

    def find_neighbors(self, instance_index, n_neighbors, return_distance=False):
        result = []
        for i in self.real_indices:
            # distances = self.dic[(instance_index, i)]
            distances = self.distances[instance_index][np.where(self.indices[instance_index] == i)[0][0]]
            result.append((distances, i))
        result = sorted(result)[:n_neighbors]

        if return_distance:
            return [i[1] for i in result], [i[0] for i in result]
        else:
            return [i[1] for i in result]


class WeightedSampleRandomGenerator(object):
    def __init__(self, indices, weigths, random_state):
        self.totals = []
        self.indices = indices
        self.random_state = random_state
        random.seed(random_state)
        running_total = 0

        for w in weigths:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = self.random_state.rand() * self.totals[-1]
        return list(self.indices)[bisect.bisect_right(self.totals, rnd)]

    def __call__(self):
        return self.next()


def clus_dis(A, B, k_table):
    distance = 0.
    for i in A:
        for j in B:
            distance += k_table.get_dis(i, j)
    return distance / len(A) / len(B)


def MWMOTE(X, Y, N, random_state, k1=5, k2=3, k3=0.5,
           C_th=5., CMAX=2, C_p=3, return_mode='append'):
    random_state = check_random_state(random_state)
    print(Counter(Y))
    logger.debug('MWMOTE: Starting with %d instances' % len(Y))
    S_min = np.flatnonzero(Y == 1)
    S_maj = np.flatnonzero(Y == -1)
    if type(k3) == float:
        k3 = round(len(S_min) * k3)

    logger.debug(' Step   0: Computing Knn table')
    start_time = time.time()
    k_tabel = Knn()
    k_tabel.fit(X)
    print('building Knn table transforming took %fs!' % (time.time() - start_time))

    # Step 1~2: Generating S_minf
    S_minf = []
    start_time = time.time()
    for i in S_min:
        neighbors = k_tabel.find_neighbors(i, k1 + 1)
        neighbors.remove(i)  # remove itself from neighbors
        if not all(neighbor in S_maj for neighbor in neighbors):
            S_minf.append(i)
    logger.debug(' Step 1~2: %d in S_minf' % len(S_minf))
    print('building Step 1~2 transforming took %fs!' % (time.time() - start_time))

    # Step 3~4: Generating S_bmaj
    start_time = time.time()
    k_tabel.fit_subset(S_maj)
    S_bmaj = []
    for i in S_minf:
        neighbors = k_tabel.find_neighbors(i, k2)
        S_bmaj.extend(neighbors)
    S_bmaj = list(set(S_bmaj))
    logger.debug(' Step 3~4: %d in S_bmaj' % len(S_bmaj))
    print('building Step 3~4 transforming took %fs!' % (time.time() - start_time))

    # Step 5~6: Generating S_imin
    start_time = time.time()
    k_tabel.fit_subset(S_min)
    S_imin = []
    N_min = {}
    for i in S_bmaj:
        neighbors = k_tabel.find_neighbors(i, k3)
        S_imin.extend(neighbors)
        N_min[i] = neighbors
    S_imin = list(set(S_imin))
    logger.debug(' Step 5~6: %d in S_imin' % len(S_imin))
    print('building Step 5~6 transforming took %fs!' % (time.time() - start_time))

    # Step 7~9: Generating I_w, S_w, S_p
    # Step 7
    start_time = time.time()
    I_w = {}
    for y in S_bmaj:
        sum_C_f = 0.
        for x in S_imin:
            # closeness_factor
            if x not in N_min[y]:
                closeness_factor = 0.
            else:
                distance_n = np.linalg.norm(X[y] - X[x]) / len(X[y])
                closeness_factor = min(C_th, (1 / distance_n)) / C_th * CMAX
            I_w[(y, x)] = closeness_factor
            sum_C_f += I_w[(y, x)]
        for x in S_imin:
            closeness_factor = I_w[(y, x)]
            density_factor = closeness_factor / sum_C_f
            I_w[(y, x)] = closeness_factor * density_factor
    # Step 8
    S_w = {}
    for x in S_imin:
        S_w[x] = math.fsum((I_w[(y, x)]) for y in S_bmaj)
    # Step 9
    S_p = {}  # select probability
    WeightSum = math.fsum(S_w.values())
    for x in S_w:
        S_p[x] = float(S_w[x]) / WeightSum
    logger.debug(' Step 7~9: %d in I_w' % len(I_w))
    print('building Step 7~9 transforming took %fs!' % (time.time() - start_time))

    # Step 10:Generating L, clusters of S_min
    start_time = time.time()
    d_avg = 0.
    for i in S_minf:
        tmp = []
        for j in S_minf:
            if i == j:
                continue
            tmp.append(np.linalg.norm(X[i] - X[j]))
        d_avg += min(tmp)
    d_avg /= len(S_minf)
    T_h = d_avg * C_p

    L = {index: [i] for index, i in enumerate(S_min)}
    clusters_number = list(range(len(S_min)))
    # initialization
    distance_table = [[0 for i in clusters_number] for j in clusters_number]
    for i in clusters_number:
        for j in clusters_number:
            distance_table[i][j] = clus_dis(L[i], L[j], k_tabel)
    # set self's cluster to max
    MAX = max(max(j) for j in distance_table)
    for i in clusters_number:
        distance_table[i][i] = MAX

    for i in S_min:
        MIN = min(min(j) for j in distance_table)
        if MIN > T_h:
            break
        for j in clusters_number:
            if MIN in distance_table[j]:
                b = distance_table[j].index(MIN)
                a = j
                break
        L[a].extend(L[b])

        del L[b]
        clusters_number.remove(b)
        for j in clusters_number:
            tmp = clus_dis(L[a], L[j], k_tabel)
            distance_table[a][j] = tmp
            distance_table[j][a] = tmp
        distance_table[a][a] = MAX
        for j in clusters_number:
            distance_table[b][j] = MAX
            distance_table[j][b] = MAX

    which_cluster = {}
    for i, clu in L.items():
        for j in clu:
            which_cluster[j] = i
    logger.debug(' Step  10: %d clusters' % len(L))
    print('building Step 10 transforming took %fs!' % (time.time() - start_time))

    # Step 11: Generating X_gen, Y_gen
    start_time = time.time()
    X_gen = np.zeros(shape=(N, X.shape[1]))
    some_big_number = 10000000.
    sample = WeightedSampleRandomGenerator(S_w.keys(), S_w.values(), random_state=random_state)
    for z in range(N):
        x = sample()
        y = random_state.choice(L[which_cluster[x]])
        alpha = random_state.randint(0, some_big_number) / some_big_number
        X_gen[z] = X[x] + alpha * (X[y] - X[x])
    y_gen = [1 for z in range(N)]
    logger.debug(' Step  11: %d over-sample generated' % N)
    print('building Step 11 transforming took %fs!' % (time.time() - start_time))

    # return the desired data
    X_res = np.vstack([X, X_gen])
    y_res = np.hstack([Y, y_gen])

    if return_mode == 'append':
        return X_res, y_res
    elif return_mode == 'shuffled':
        Permutation = range(len(X_res))
        random_state.shuffle(Permutation)
        X_res = X_res[Permutation]
        y_res = y_res[Permutation]
        return X_res, y_res
    elif return_mode == 'only':
        return X_gen, y_gen
    else:
        pass


if __name__ == '__main__':
    dataset = fetch_datasets()['oil']
    X = dataset.data
    y = dataset.target
    X_gen, y_gen = MWMOTE(X, y, N=200, random_state=42)
    print(len(X_gen))
    print(len(y_gen))
    nbrs = NearestNeighbors(n_neighbors=len(X)).fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(distances)
