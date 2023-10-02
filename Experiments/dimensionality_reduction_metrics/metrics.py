import numpy as np
import numpy.linalg as LA
import math as m
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from tqdm import tqdm

def metric_trustworthiness(A,Z,k=7):
    return trustworthiness(A,Z,n_neighbors=k)

def distance_matrix(A:np.ndarray,sample:int, worker_id:int, dataset: str=None) -> np.ndarray :
    m,n=A.shape
    examples= sample if sample else m
    D=np.zeros((examples,m))
    progress_bar_id = f"Computing {dataset}"
    dataset_length = examples
    progress_bar = tqdm(total=dataset_length, desc=progress_bar_id, position=worker_id)
    for i in range(examples):
        for j in range(m):
            D[i,j]=LA.norm(A[i,:]-A[j,:])
        progress_bar.update(1)
    progress_bar.close()
    return D

def continuity(A: np.ndarray ,Z: np.ndarray ,k: int ,sample: int, dist_matrix=None ):
    if dist_matrix is None:
        DA=distance_matrix(A,sample, worker_id=0)
    else:
        DA=dist_matrix
    DZ=distance_matrix(Z,sample, worker_id=0,dataset='Embedding Dist Matrix')
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
    DA=distance_matrix(A,sample).flatten()
    DZ=distance_matrix(Z,sample).flatten()
    return stats.spearmanr(DA,DZ)[0]

def espadoto_stress(A,Z):
    scaler1,scaler2=MinMaxScaler(), MinMaxScaler()
    DA=scaler1.fit_transform(distance_matrix(A,None)).flatten()
    DZ=scaler2.fit_transform(distance_matrix(Z,None)).flatten()
    stress=0
    for i in range(len(DA)):
        stress+=(DA[i]-DZ[i])**2
    stress/=np.sum(DA**2)
    return stress






