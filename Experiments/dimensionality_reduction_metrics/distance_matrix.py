import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from metrics import distance_matrix
from tqdm import tqdm
from multiprocessing import cpu_count,Pool


def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save distance matrices')
    parser.add_argument('--save_dir', '-s', help='Path to the directory where the distance matrices are saved', default='./distance_matrices')
    parser.add_argument('--datasets', '-d', nargs='+', help='List of names of the dataset (same name as the folder which contains .npy files)')
    # parser.add_argument('--sample_sizes', nargs='+', help='List of sample sizes of each dataset')
    parser.add_argument('--data_dir', help='Directory of the dataset', default='./normalized_data')
    args=parser.parse_args()
    return args


def compute_and_dump(dataset:str,DATA_DIR:str,SAVE_DIR:str,worker_id:int):
    X=np.load(os.path.join(DATA_DIR,dataset,'X.npy'),allow_pickle=True)
    D=distance_matrix(X,None,worker_id,dataset)
    np.save(os.path.join(SAVE_DIR,f'{dataset}.npy'),D)
    print(f'Saved {dataset}')


def main():
    args=parse_arguments()
    DATA_DIR= args.data_dir if args.data_dir else './datasets'
    SAVE_DIR=args.save_dir if args.save_dir else './distance_matrices'

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    progress_bars = []

    
    results=[pool.apply_async(compute_and_dump,args=(dataset,DATA_DIR,SAVE_DIR,i)) for i,dataset in enumerate(args.datasets)]

    for result in results:
        result.wait()
    # pool.starmap(compute_and_dump,[(dataset,DATA_DIR,SAVE_DIR) for dataset in args.datasets])
    pool.close()
    pool.join()

if __name__=="__main__":
    main()

    


