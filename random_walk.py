import numpy as np
import numpy.random as npr
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize

class walk_dic_featwalk:
    def __init__(self, Net, featur, num_paths, path_length, alpha):
        self.alpha = alpha
        self.num_paths = num_paths
        self.path_length = path_length
        self.n, self.m = featur.shape  # number of instance
        Net = normalize(csr_matrix(Net), norm='l1', axis=0)
        featur = normalize(csr_matrix(featur), norm='l1')
        self.path_list_Net = []
        self.qListrow = []  # for each instance
        self.JListrow = []
        self.idx = []
        net_featur = normalize(vstack([self.alpha * Net, (1 - self.alpha) * featur.T]), norm='l1', axis=0)
        net_featur.eliminate_zeros()

        for ni in range(self.n):  # for each instance select net or feature
            coli = net_featur.getcol(ni)
            self.path_list_Net.append(coli.nnz)
            J, q = alias_setup(coli.data)
            self.JListrow.append(J)
            self.qListrow.append(q)
            self.idx.append(coli.indices)
        featur = normalize(featur, norm='l1', axis=0)
        featur.eliminate_zeros()

        for ni in range(self.m):  # for each feature
            coli = featur.getcol(ni)
            J, q = alias_setup(coli.data)
            self.JListrow.append(J)
            self.qListrow.append(q)
            self.idx.append(coli.indices)


    def function(self):
        sentencedic = [[] for _ in range(self.n)]  # All the walks will be here
        sentnumdic = [[] for _ in range(self.n)]
        allidx = np.nonzero(self.path_list_Net)[0]

        if len(allidx) != self.n:  # initialize with Network
            for i in np.where(np.asarray(self.path_list_Net) == 0)[0]:
                sentencedic[i] = [i] * (self.path_length * self.num_paths)
        for i in allidx:
            sentence = []
            for j in range(self.num_paths):
                sentence.append(i)
                current = i
                for senidx in range(self.path_length - 1):
                    current = self.idx[current][alias_draw(self.JListrow[current], self.qListrow[current])]
                    sentence.append(current)

            sentencedic[i] = sentence
        return sentencedic, sentnumdic


#  Compute utility lists for non-uniform sampling from discrete distributions.
#  Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()
        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]