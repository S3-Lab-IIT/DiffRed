from .utils import data_scaler, get_flag,approx_stress
import math as m
import numpy as np
import sklearn.preprocessing as skp
import os
import numpy.linalg as LA

class DiffRed():
    def __init__(self,k1,k2,opt_metric='m1'):
        self.k1=k1
        self.k2=k2
        self.opt_metric=opt_metric
    def calculate_u_sigma(self,A):
        n,D=A.shape
        if self.flag:
            aaT=A.T@A
        else:
            aaT=A@A.T
        eigv,eigvec=np.linalg.eigh(aaT)
        for i in range(eigv.shape[0]):
            if eigv[i]<0.0:
                eigv[i]=0
                eigvec[:,i]=0
            eigv=eigv[::-1]
            eigvec=eigvec[:,::-1]
            sigma=np.sqrt(eigv)
            return eigvec,sigma
    def get_Ak(self,A):
        if self.flag:
            Ak=LA.multi_dot([A,self.U[:,0:self.k1],self.U[:,0:self.k1].T])
            return Ak
        else:
            Ak=LA.multi_dot([self.U[:,0:self.k1],self.U[:,0:self.k1].T,A])
            return Ak
    def get_X(self,A):
        if self.flag:
            X=A@self.U[:,0:self.k1]
            return X
        else:
            X=self.U[:,:self.k1]*self.sigma[:self.k1]
            return X
    def fit(self,A):
        self.n, self.D=A.shape
        self.flag=get_flag(A)
        A,self.scaler=data_scaler(A)
        self.U,self.sigma= self.calculate_u_sigma(A)
        self.Ak1=self.get_Ak(A)
        self.Ar=A-self.Ak1

    # def get_R(self):
    #     scale=m.sqrt(self.D/self.k2)
    #     R=np.random.normal(size=(self.D,self.k2),scale=m.sqrt(1/self.D))
    #     return scale,R

    def get_Y(self):
        scale=m.sqrt(self.D/self.k2)
        R=np.random.normal(size=(self.D,self.k2),scale=m.sqrt(1/self.D))
        Y=scale*(self.Ar@R)
        return Y,scale,R


    def monte_carlo_search(self,sample,max_iter):
        self.max_iter=max_iter
        self.sample=sample
        if self.opt_metric.lower()=='stress':
            minimum=float('inf')
            min_Y=None
            min_scale=None
            min_R=None
            stress=0
            for i in range(self.max_iter):
                Y,scale,R=self.get_Y()
                stress=approx_stress(self.Ar,Y,self.sample)
                if stress<minimum:
                    minimum=stress
                    min_Y=Y
                    min_scale=scale
                    min_R=R
            self.stress=minimum 
            self.opt_Y=min_Y
            self.opt_scale=min_scale
            self.opt_R=min_R    #optimal value of Y that minimizes stress
        elif self.opt_metric.lower()=='m1':
            minimum=float('inf')
            min_Y=None
            min_scale=None
            min_R=None
            residual_norm=LA.norm(self.Ar,ord='fro')
            for i in range(self.max_iter):
                Y,scale,R=self.get_Y()
                fro_Y=LA.norm(Y,ord='fro')
                fro_diff=abs(1-(fro_Y/residual_norm)**2)
                if fro_diff<minimum:
                    minimum=fro_diff
                    min_Y=Y
                    min_scale=scale
                    min_R=R
            self.opt_Y=min_Y
            self.m1=minimum
            self.opt_scale=min_scale
            self.opt_R=min_R
    
    def transform(self,A,monte_carlo=True,sample=100,max_iter=100):
        Ak1=self.get_Ak(A)
        Ar=A-Ak1
        self.X=self.get_X(A)
        if monte_carlo:
            self.monte_carlo_search(sample,max_iter)
            self.Y=self.opt_scale*(Ar@self.opt_R)
        else:
            scale=m.sqrt(self.D/self.k2)
            R=np.random.normal(size=(self.D,self.k2),scale=m.sqrt(1/self.D))
            self.Y=scale*(Ar@R)
        return np.concatenate((self.X,self.Y),axis=1)
    
    def fit_transform(self,A,monte_carlo=True,sample=100,max_iter=100):
        self.fit(A)
        return self.transform(A,monte_carlo,sample,max_iter)




                






