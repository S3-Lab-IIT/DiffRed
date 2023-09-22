import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from sklearn.metrics.pairwise import pairwise_distances
from time import time


def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save distance matrices')
    parser.add_argument('--save_dir', '-s', help='Path to the directory where the distance matrices are saved', default='./distance_matrices')
    parser.add_argument('--dataset', '-d', help='Name of the dataset (same name as the folder which contains .npy files)')
    # parser.add_argument('--sample_sizes', nargs='+', help='List of sample sizes of each dataset')
    parser.add_argument('--data_dir', help='Directory of the dataset', default='./normalized_data')
    args=parser.parse_args()
    return args

def compute_and_dump(dataset:str,DATA_DIR:str,SAVE_DIR:str):
    X=np.load(os.path.join(DATA_DIR,dataset,'X.npy'),allow_pickle=True)
    D=pairwise_distances(X,metric='euclidean',n_jobs=-1)
    np.save(os.path.join(SAVE_DIR,f'{dataset}.npy'),D)
    print(f'Saved {dataset}')


def main():
    args=parse_arguments()
    start_time=time()
    compute_and_dump(args.dataset,args.data_dir,args.save_dir)
    print(f'Time taken: {time()-start_time}')

if __name__=="__main__":
    main()