import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count,Pool
import sys
sys.path.append('../..')
from DiffRed import DiffRed


def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save embedding matrices')
    parser.add_argument('--save_dir','-s', help='Path to the directory where the embeddings are to be saved', default='./embeddings')
    parser.add_argument('--dataset','-d',help='Name of the dataset whose embeddings are to be computed')
    parser.add_argument('--data_dir', help='Directory where the datasets are stored', default='./normalized_data' )
    parser.add_argument('--k1', help='List of k1 values seperated by space', nargs='+')
    parser.add_argument('--k2', help='List of corresponding k2 values seperated by space', nargs='+')
    parser.add_argument('--max_iter_list', help='True if max_iter is a list. If false then the single value provided is assumed to be for all the target dimensions.', default='True')
    parser.add_argument('--max_iter', help='List of max iter values corresponding to k1 and k2', nargs='+')

    args=parser.parse_args()
    return args

def compute_embeddings(dataset:str, DATA_DIR: str, SAVE_DIR: str, k1:int, k2:int, max_iter:int ):
    X=np.load(os.path.join(DATA_DIR,dataset,'X.npy'),allow_pickle=True)
    dr=DiffRed(k1,k2)
    if os.path.exists(os.path.join(SAVE_DIR,dataset, f'{k1}_{k2}_{max_iter}.npy')):
        print(f"Emebddings already saved for k1={k1}, k2={k2} max_iter={max_iter}")
        return
    Z=dr.fit_transform(X,max_iter)
    if not os.path.exists(os.path.join(SAVE_DIR,dataset)):
        os.mkdir(os.path.join(SAVE_DIR,dataset))
    np.save(os.path.join(SAVE_DIR,dataset,f'{k1}_{k2}_{max_iter}.npy'),Z)



def main():
    args=parse_arguments()
    k1=[int(x) for x in args.k1]
    k2=[int(x) for x in args.k2]
    if args.max_iter_list=='True':
        max_iter=[int(x) for x in args.max_iter]
    else:
        max_iter=[int(args.max_iter[0]) for x in range(len(k1))]
    dr_args=zip(k1,k2,max_iter)

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)
    
    # pbar=tqdm(dr_args, desc='Computing...')

    results=[pool.apply_async(compute_embeddings,args=(args.dataset,args.data_dir,args.save_dir, arg[0],arg[1],arg[2])) for arg in dr_args]

    for result in tqdm(results):
        result.wait()
    
    pool.close()
    pool.join()

    # for arg in dr_args:
    #     compute_embeddings(args.dataset,args.data_dir,args.save_dir, arg[0],arg[1],arg[2])



if __name__=="__main__":
    main()