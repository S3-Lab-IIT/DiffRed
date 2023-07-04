import numpy as np
import math as m
from sklearn.preprocessing import StandardScaler, normalize
import numpy.linalg as LA

def data_scaler(A):
    scaler=StandardScaler()
    A=scaler.fit_transform(A.T)
    A=A.T
    A=normalize(A,norm='l2')
    return A, scaler

def stress(A,Ad):
# Calculates Stress Metric Exactly
    n,N=A.shape
    d=Ad.shape[1]
    metric=0
    sum_sq=0
    p=n*(n-1)*0.5
    c=0
    for i in range(n):
        for j in range(i):
          d2ij=Ad[i,:]-Ad[j,:]
          dij=A[i,:]-A[j,:]
          d1=(LA.norm(dij)-LA.norm(d2ij))**2
          metric+=d1
          sum_sq+=LA.norm(dij)**2
#           d2=diff_norms[c]**2
#           sum_sq+=d2
          c+=1
          # if c%100000==0:
          #   print(c,end=' ')
    # sum_sq=np.sum(diff_norms**2)
    metric/=sum_sq
    return m.sqrt(metric)

def m1(A,Ad):
    fro_1=LA.norm(Ad,ord='fro')
    fro_2=LA.norm(A,ord='fro')
    metric=1-(fro_1/fro_2)**2
    return abs(metric)

def approx_stress(A,Ad, sample):
    n,N=A.shape
    d=Ad.shape[1]
    metric=0
    sum_sq=0
    for i in range(sample):
        for j in range(i):
            d2ij=Ad[i,:]-Ad[j,:]
            dij=A[i,:]-A[j,:]
            metric+=(LA.norm(dij)-LA.norm(d2ij))**2
            sum_sq+=LA.norm(dij)**2
    metric/=sum_sq
    return m.sqrt(metric)

def get_flag(A):
    n,D=A.shape
    if n<D:
        return 0
    else:
        return 1

def calculate_singular_values(A:np.ndarray)->np.ndarray:
    if get_flag(A)==0:
        aaT=A@A.T
    else:
        aaT=A.T@A
    eigv,eigvec=LA.eigh(aaT)
    for i in range(eigv.shape[0]):
        if eigv[i]<0.0:
            eigv[i]=0
    sigma=np.sqrt(eigv[::-1])
    return sigma

def opt_dimensions(A:np.ndarray, target_dimension: int, energy_threshold=0.98) -> tuple[int, int]:
    sigma=calculate_singular_values(A)
    sum_sq=np.sum(sigma**2)
    for k1 in range(len(sigma)):
        energy=np.sum(sigma[:k1]**2)/sum_sq
        if energy>=energy_threshold:
            if not target_dimension-k1<0:
                return k1, target_dimension-k1
            else:
                return k1,0