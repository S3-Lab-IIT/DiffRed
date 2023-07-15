import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from metrics import distance_matrix
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
from compute_theoretical_opt import calculate_stress_bound

lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Plot theoretical bound against experimental metric values')

    parser.add_argument('--dataset', '-d', help='Name of the dataset')

    parser.add_argument('--theory_dir', help='Directory where theoretical values are stored', default='./stress_bound_vals')

    parser.add_argument('--exp_result_file', '-e', help='Path to the file where experimental results are stored')

    parser.add_argument('--target_dims', nargs='+', help='Target Dimensions for which the plot is to be made')

    # parser.add_argument('--metric', '-m', help='Metric for which plot is to be made', choices=['stress', 'm1'], default='stress')

    parser.add_argument('--save_dir', help='Directory where plots are to be saved', default='./stress_plots')

    args=parser.parse_args()
    return args


def get_theory_vals(file_path:str):

    df=pd.read_excel(file_path)
    
    return [df['k1'].values,df['stress_bound'].values]

def get_exp_values(file_path:str, target_dim:int):

    df=pd.read_excel(file_path)

    subset=df[df['Target Dimension']==target_dim]

    return [subset['k1'].values, subset['Stress'].values]

def plot_stress(dataset:str, SAVE_DIR:str, TH_DIR:str, EXP_FILE:str, target_dim:int):

    global lock
    save_dir=os.path.join(SAVE_DIR,f'dataset')

    if not os.path.exists(save_dir):
        lock.acquire()
        os.mkdir(save_dir)
        lock.release()
    
    theory_vals=get_theory_vals(os.path.join(TH_DIR,f'{dataset}', 'stress', f'bound_{target_dim}.xlsx'))

    exp_vals=get_exp_values(EXP_FILE,target_dim)

    theory_x, theory_y=theory_vals

    exp_x, exp_y=exp_vals

    idx1=np.argsort(theory_x)
    idx2=np.argsort(exp_x)

    theory_x,theory_y=theory_x[idx1], theory_y[idx1]

    exp_x, exp_y = exp_x[idx2], exp_y[idx2]

    plt.plot(theory_x, theory_y, color='blue', label='Theoretical Stress bound')

    plt.plot(exp_x,exp_y, color='red', label='Experimental Stress')

    plt.xlabel('k1')
    plt.ylabel('Stress')
    plt.title(f'{dataset} Target Dimension: {target_dim}')
    plt.legend()

    theoretical_minima=np.min(theory_y)
    exp_minima=np.min(exp_y)

    plt.axvline(x=theory_x[np.where(theory_y==theoretical_minima)][0], linestyle='dashed', color='blue')
    plt.axvline(x=exp_x[np.where(exp_y==exp_minima)][0], linestyle='dashed', color='red')

    plt.text(theory_x[np.where(theory_y==theoretical_minima)][0],0.05, f'')
    plt.text(exp_x[np.where(exp_y==exp_minima)][0],0.05, f'')

    # plt.axhline(y=0, color='black')

    plt.savefig(os.path.join(save_dir,f'{dataset}_{target_dim}.png'))


def main():

    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(plot_stress, args=(args.dataset,args.save_dir,args.theory_dir,args.exp_result_file,target_dim)) for target_dim in target_dims]

    for result in results:
        result.wait()

    pool.close()
    pool.join()

if __name__=="__main__":
    main()
    
