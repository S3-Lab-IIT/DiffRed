import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute RMap for a dataset')

    parser.add_argument('--save_dir', '-s', help='Path to the directory where pca embeddings are to be saved', default='./rmap_embeddings')
    parser.add_argument('--dataset', '-d', help='Dataset whose embeddings are to be computed')
    parser.add_argument('--data_dir',help='Directory where datasets are stored',default='../normalized_data')
    parser.add_argument('--target_dims', '-t', help='List of Target dimensions seperated by space', nargs='+')
    parser.add_argument('--eta', type=int, help='Number of instances of RMap to be computed for generating confidence interval', default=100)
    
    args=parser.parse_args()
    return args


def rmap(A:np.ndarray, target_dim:int):
    n,D=A.shape
    G=np.random.normal(0,1/m.sqrt(D), (D,target_dim))
    coeff=m.sqrt(D/target_dim)

    return coeff*G

def compute_jl_embeddings(dataset:str, DATA_DIR:str, SAVE_DIR:str, eta:int, target_dim:int, worker_id:int):
    global lock
    save_path=os.path.join(SAVE_DIR, dataset, str(target_dim))

    worker_desc=f'Dataset: {dataset} target dim: {target_dim}'
    lock.acquire()
    A=np.load(os.path.join(DATA_DIR, dataset, 'X.npy'))
    lock.release()
    lock.acquire()
    if not os.path.exists(os.path.join(SAVE_DIR, dataset)):
        os.mkdir(os.path.join(SAVE_DIR, dataset))
    
    if not os.path.exists(os.path.join(SAVE_DIR, dataset, str(target_dim))):
        os.mkdir(os.path.join(SAVE_DIR, dataset, str(target_dim)))
    
    lock.release()
    
    

    for i in tqdm(range(eta), desc=f'{worker_desc}', position=worker_id):

        if not os.path.exists(os.path.join(save_path, f'{i}.npy')):
            G=rmap(A, target_dim)
            Z=A@G

            np.save(os.path.join(save_path, f'{i}.npy'), Z)
        else:
            continue

def main():

    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(compute_jl_embeddings, args=(args.dataset, args.data_dir, args.save_dir, args.eta, target_dims[i], i )) for i in range(len(target_dims))]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()

if __name__=="__main__":
    main()
    

    




