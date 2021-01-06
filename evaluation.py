from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np


def evaluation(X, Y, Kset):
    num = X.shape[0]
    # print("X.shape",X.shape)
    classN = np.max(Y)+1
    # print("classN",classN)
    kmax = np.max(Kset)
    # print("kmax",kmax)
    recallK = np.zeros(len(Kset))
    # print("recallk",recallK)
    #compute NMI
    kmeans = KMeans(n_clusters=classN).fit(X)
    # print("kmeans",kmeans)
    # nmi = normalized_mutual_info_score(Y, kmeans.labels_, average_method='arithmetic')
    nmi = normalized_mutual_info_score(Y, kmeans.labels_)
    # print("nmi:",nmi)

    #compute Recall@K
    sim = X.dot(X.T)
    # print("sim:",sim.shape)
    minval = np.min(sim) - 1.
    # print("minval:",minval)
    sim -= np.diag(np.diag(sim))
    # print("sim -= np.diag(sim) :",sim.shape)
    sim += np.diag(np.ones(num) * minval)
    # print("sim += np.diag(np.ones(sim)) :",sim.shape)

    indices = np.argsort(-sim, axis=1)[:, : kmax]
    # print("indices :",indices.shape)

    YNN = Y[indices]
    # print("YNN :",YNN.shape)

    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, num):
            if Y[j] in YNN[j, :Kset[i]]:
                pos += 1.
        recallK[i] = pos/num
    return nmi, recallK