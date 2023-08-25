import sys
sys.path.append('../../')
from DiffRed import DiffRed as dr 
from DiffRed.utils import stress,data_scaler
import numpy as np
from tensorflow.keras.datasets.fashion_mnist import load_data
from tqdm import tqdm
import matplotlib.pyplot as plt


(x_train,y_train), (x_test,y_test)=load_data()
x_train=x_train.reshape((60000,784))
del x_test,y_test
x_train,_=data_scaler(x_train)
sample_sizes=list(range(10,10000,100))
stress_list=[]
k1=5
k2=15
for sample in tqdm(sample_sizes,desc='Loading...'):
    A=x_train[:sample]
    DiffRed=dr(k1,k2,opt_metric='m1')
    Z=DiffRed.fit_transform(A)
    stress_list.append(stress(A,Z))

plt.plot(sample_sizes,stress_list,color='red')
plt.title('Dependence of Stress Estimate with Sample Size')
plt.xlabel('Sample Size')
plt.ylabel('Stress')
plt.savefig('./results/stress_vs_sample.png')