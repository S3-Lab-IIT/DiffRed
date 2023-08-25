import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
from sklearn.decomposition import PCA

lock = Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute PCA embeddings')

    parser.add_argument('--save_dir', '-s', help='Path to the directory where pca embeddings are to be saved', default='./pca_embeddings')
    parser.add_argument('--dataset', '-d', help='Dataset whose embeddings are to be computed')
    parser.add_argument('--data_dir',help='Directory where datasets are stored',default='../normalized_data')
    parser.add_argument('--target_dims', '-t', help='List of Target dimensions seperated by space', nargs='+')
    
    args=parser.parse_args()
    return args

def compute_embeddings(dataset:str, DATA_DIR:str,SAVE_DIR:str, target_dim:int):

    X=np.load(os.path.join(DATA_DIR,dataset,'X.npy'), allow_pickle=True)

    pca=PCA(n_components=target_dim)

    if os.path.exists(os.path.join(SAVE_DIR,dataset, f'{dataset}_{target_dim}_{pca}.npy')):
        print(f'Embeddings already saved for {dataset} for target dimension {target_dim}')

        return
    Z=pca.fit_transform(X)

    lock.acquire()
    if not os.path.exists(os.path.join(SAVE_DIR,dataset)):
        os.mkdir(os.path.join(SAVE_DIR,dataset))
    lock.release()

    np.save(os.path.join(SAVE_DIR, dataset,f'{dataset}_{target_dim}_pca.npy'), Z)


def main():
    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(compute_embeddings, args=(args.dataset,args.data_dir, args.save_dir,target_dim)) for target_dim in target_dims]

    for result in tqdm(results):
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()
