import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from metrics import distance_matrix
from tqdm import tqdm
from multiprocessing import cpu_count,Pool
from sklearn.preprocessing import StandardScaler, normalize


def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save distance matrices')
    parser.add_argument('--save_dir', '-s', help='Path to the directory where the normalized datasets are to be saved', default='./normalized_data')
    parser.add_argument('--datasets', '-d', nargs='+', help='List of names of the dataset')
    # parser.add_argument('--sample_sizes', nargs='+', help='List of sample sizes of each dataset')
    parser.add_argument('--data_dir', help='Directory of the dataset', default='./datasets')
    args=parser.parse_args()
    return args


def normalize_and_save(dataset:str, DATA_DIR:str, SAVE_DIR:str):
    X=np.load(os.path.join(DATA_DIR,dataset,'X.npy'),allow_pickle=True)
    scaler=StandardScaler()
    X=scaler.fit_transform(X.T)
    X=X.T
    X=normalize(X,norm='l2')
    if not os.path.exists(os.path.join(SAVE_DIR,f'{dataset}')):
        os.mkdir(os.path.join(SAVE_DIR,f'{dataset}'))
    np.save(os.path.join(SAVE_DIR,f'{dataset}','X.npy'),X)
    print(f'Saved {dataset}')


def main():
    args=parse_arguments()
    
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(normalize_and_save, args=(dataset,args.data_dir,args.save_dir)) for dataset in args.datasets]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()

if __name__=="__main__":
    main()
