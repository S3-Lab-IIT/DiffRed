import math as m
import numpy as np
import sklearn.preprocessing as skp
import os
import numpy.linalg as LA
import pickle

def data_scaler(data):
    scaler=skp.StandardScaler()
    data=scaler.fit_transform(data.T)
    data=data.T
    data=skp.normalize(data,norm='l2')
    return data, scaler

def get_flag(data):
    n,D=data.shape
    if n<D:
        flag=0
    else:
        flag=1
    return flag

def approx_stress(A_r,Ad,sample):
    # Approximate calculation of Stress metric for a smaller sample
    n,N=A_r.shape
    d=Ad.shape[1]
    num=0 #numerator has sum of squares of distance pair distortions
    den=0 #denominator has sum of squares of distance pairs
    sum_sq=0
    p=n*(n-1)*0.5
    c=0
    # We limit the number of calculations done by using a sample
    for i in range(sample):
        for j in range(i):
          d2ij=Ad[i,:]-Ad[j,:]
          dij=A_r[i,:]-A_r[j,:]
          d1=(LA.norm(dij)-LA.norm(d2ij))**2
          d2=LA.norm(dij)**2  
          num+=d1
          den+=d2
   
    return m.sqrt(d1/d2)


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
          sum_sq+=LA.norm(dij)**2
          metric+=d1
#           d2=diff_norms[c]**2
#           sum_sq+=d2
          c+=1
          # if c%100000==0:
          #   print(c,end=' ')
    metric/=sum_sq
    return m.sqrt(metric)

def m1(A,Ad):
    return abs(1-(LA.norm(Ad,ord='fro')/LA.norm(A,ord='fro'))**2)

def print_in_color(input, color):
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m'
    }
    if color not in colors:
        print(f"Error: Invalid color '{color}'")
        return
    print(f"{colors[color]}{input}\033[0m")


def eval(A,Ad,colors=['red','blue']):
    print_in_color('Stress: {}'.format(stress(A,Ad)),colors[0])
    print_in_color('M1: {}'.format(m1(A,Ad)),colors[1])