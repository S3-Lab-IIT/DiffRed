import numpy as np
import numpy.linalg as LA
import math as m
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

def metric_trustworthiness(A,Z,k=7):
    return trustworthiness(A,Z,n_neighbors=k)

def distance_matrix(A:np.ndarray,sample:int) -> np.ndarray :
    m,n=A.shape
    examples= sample if sample else m
    D=np.zeros((examples,m))
    for i in range(examples):
        for j in range(m):
            D[i,j]=LA.norm(A[i,:]-A[j,:])
    return D

def continuity(A: np.ndarray ,Z: np.ndarray ,k: int ,sample: int ):
    DA=distance_matrix(A,sample)
    DZ=distance_matrix(Z,sample)
    m,n=A.shape
    examples= sample if sample else m
    nnA=DA.argsort()
    nnZ=DZ.argsort()
    knnA=nnA[:,1:k+1]
    knnZ=nnZ[:,1:k+1]
    sum_i=0
    for i in range(examples):
        V=np.setdiff1d(knnA[i],knnZ[i])
        sum_j=0
        for j in range(V.shape[0]):
            sum_j += np.where(nnZ[i] == V[j])[0] - k
        sum_i+=sum_j
    return float((1 - (2 / (examples * k * (2 * examples - 3 * k - 1)) * sum_i)).squeeze())

def neighborhood_hit(Z,yZ,k):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(Z,yZ)
    neighbors=knn.kneighbors(Z,return_distance=False)
    return np.mean(np.mean((yZ[neighbors]==np.tile(yZ.reshape((-1,1)),k)).astype('uint8'),axis=1))


def shepard_goodness(A,Z,sample=None):
    DA=distance_matrix(A,sample)
    DZ=distance_matrix(Z,sample)
    return stats.spearmanr(DA,DZ)[0]




