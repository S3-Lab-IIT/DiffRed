import sys
sys.path.append('../../')
from DiffRed import DiffRed
import os
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, normalize
import math as m
from .utils import pseudoinverse, scale_data, inverse_norm, inverse_scale

class DiffCoder(DiffRed):

    def __init__(self, k1, k2, opt_metric='m1'):
        super().__init__(k1, k2, opt_metric)
    
    def encoder(self, A:np.ndarray):
        scaler, norms, A=scale_data(A)
        B=self.fit_transform(A)
        return B,scaler,norms
    
    
    def decoder(self, scaler:StandardScaler, norms:np.ndarray, B:np.ndarray):
        Z=B[:,:self.k1]
        R=B[:,self.k1:]

        self.Ak_res=self.Ak1
#         self.A_star_res=self.opt_Y@LA.pinv(self.opt_R)*(1/self.opt_scale)
        pseudo=pseudoinverse(self.opt_R)
        self.A_star_res=self.opt_Y@pseudo*(1/self.opt_scale)

        self.A_dec=self.Ak_res+self.A_star_res

        
#         return self.A_dec
    
    def reconstruct(self, A:np.ndarray):

        B,scaler,norms=self.encoder(A)
        self.decoder(scaler,norms,B)
        A_res=inverse_norm(self.A_dec, norms)
        A_res= inverse_scale(scaler,A_res)

        return A_res


