from .utils import data_scaler, get_flag,approx_stress,stress,m1
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
    
    # def calculate_u_sigma(self, A):
    #     if self.flag==0:
    #         aaT=A@A.T
    #     else:
    #         aaT=A.T@A
    #     eigv,eigvec=LA.eigh(aaT)
    #     for i in range(eigv.shape[0]):
    #         if eigv[i]<0.0:
    #             eigv[i]=0
    #             eigvec[:,i]=0
    #     eigv=eigv[::-1]
    #     eigvec=eigvec[:,::-1]
    #     sigma=np.sqrt(eigv)
    #     return eigvec, sigma
    def calculate_u_sigma(self,A):
    # Calculates U matrix and Sigma Vector
        if self.flag==0:
            aaT=A@A.T
        else:
            aaT=A.T@A
        eigv,eigvec=LA.eigh(aaT)
        for i in range(eigv.shape[0]):
            #print(eigv[i])
            if eigv[i]<0.0:
                eigv[i]=0
                eigvec[:,i]=0
        #need to check this code
        eigv=eigv[::-1] #eigh returns eigenvalues in ascending order so we need to reverse the order to have descending order 
        eigvec=eigvec[:,::-1] #we need to reverse the eigenvectors(columns of eigvec) as well
        sigma=np.sqrt(eigv) #take square root for singular values
        return eigvec,sigma 
    def get_X(self,A):
        if self.flag==0:
            Uk1=self.U[:,:self.k1]
            X=Uk1*self.sigma[:self.k1]
            return X
        else:
            print(self.U[:,0:self.k1].shape)
            X=A@self.U[:,0:self.k1]
            return X
    
    def get_Ak(self,A):
        if self.flag==0:
            Ak=LA.multi_dot([self.U[:,0:self.k1],self.U[:,0:self.k1].T,A])
            return Ak
        else:
            Ak=LA.multi_dot([A,self.U[:,0:self.k1],self.U[:,0:self.k1].T])
            return Ak
    
    def get_Y(self):
        scale=m.sqrt(self.D/self.k2)
        R=np.random.normal(size=(self.D,self.k2),scale=m.sqrt(1/self.D))
        second_term=self.Ar@R
        return scale*second_term,scale,R
    
    def monte_carlo_search(self,A,max_iter=100,sample=100):
        minimum=100000
        min_Y=None
        min_scale=None
        min_R=None
        if self.opt_metric=='m1':
            residual_norm=LA.norm(self.Ar,ord='fro')
            for i in range(max_iter):
                Y,scale,R=self.get_Y()
                fro_Y=LA.norm(Y,ord='fro')
                metric=abs(1-(fro_Y/residual_norm)**2)
                if metric<minimum:
                    minimum=metric
                    min_Y=Y
                    min_scale=scale
                    min_R=R
        elif self.opt_metric=='stress':
            for i in range(max_iter):
                Y,scale,R=self.get_Y()
                metric=approx_stress(self.Ar,Y,sample)
                if metric<minimum:
                    minimum=metric
                    min_Y=Y
                    min_R=R
                    min_scale=scale
        return minimum, min_Y, min_R, min_scale,

    def fit_transform(self,A,max_iter):
        self.n, self.D= A.shape
        self.flag=get_flag(A)
        # A,self.scaler=data_scaler(A)
        self.U,self.sigma=self.calculate_u_sigma(A)
        self.Ak1=self.get_Ak(A)
        self.X=self.get_X(A)
        self.Ar=A-self.Ak1
        self.metric,self.opt_Y,self.opt_R,self.opt_scale=self.monte_carlo_search(A,max_iter)
        self.embeddings=np.concatenate((self.X,self.opt_Y),axis=1)
        return self.embeddings

        





                






