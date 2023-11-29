# FOR DIFFRED:
# python3 renormalize_embeddings.py -d Bank --data_dir ../../Experiments/dimensionality_reduction_metrics/embeddings/ --embed_dir ./tsne_embeddings/ -t 3 5 6 7 8 10 --k1 2 4 5 6 7 7 --save_dir ./tsne2_embeddings --dr_tech DiffRed --max_iter 100 --max_iter_list False
# FOR PCA:
#  python3 renormalize_embeddings.py -d Bank --data_dir ../../Experiments/dimensionality_reduction_metrics/other_dr_techniques/pca_embeddings/ --embed_dir ./tsne_embeddings/ -t 3 5 6 7 8 10  --save_dir ./tsne2_embeddings --dr_tech PCA

import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count,Pool, Lock
from itertools import product
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
lock=Lock()

def parse_arguments():
    parser=argparse.ArgumentParser(description='Renormalize UMap for stress computation as described in the paper (UMap2,TSNE2). ')

    parser.add_argument('--dataset', '-d', help='Dataset for which renormalization is to be done' )
    parser.add_argument('--data_dir', help='Directory where the data whose frobenius norm is to be matched by the embeddings is stored')
    parser.add_argument('--embed_dir', help='Directory where UMap embeddings are stored')
    parser.add_argument('--target_dims', '-t',nargs='+', help='List of target dimensions (seperated by space) for which the renormalization is to be done')
    parser.add_argument('--k1', help='List of k1 values corresponding to each target dimension. Seperated by space. ( Argument required only for DiffRed, not for other techniques)', nargs='+')
    # parser.add_argument('--settings', '-s', nargs='+',help='List of settings(seperated by space) corresponding to target dimension for which the renormalization is to be done')
    # parser.add_argument('--setting_list', help='Whether a list of settings is provided or a single setting is provided which is to be used all target dimensions', default='True')
    parser.add_argument('--save_dir', help='Directory where to save the re-normalized embeddings', default='./embeddings')
    parser.add_argument('--dr_tech', help='Dimensionality Reduction technique which is used as preprocessing step. Choice: PCA, DiffRed')
    # parser.add_argument('--dr_tech2', help='Dimensionality Reduction technique which is used for downstream visualization. Choice: T-SNE, UMap')
    parser.add_argument('--max_iter', help='Max iter values corresponding to k1. (Required only for DiffRed)', nargs='+', default=100)
    parser.add_argument('--max_iter_list', help='True if max_iter is a list. If false then the single value provided is assumed to be for all the target dimensions.', default='True')

    args=parser.parse_args()
    return args

def renormalize_embeddings(dataset:str, embed_dir:str, dr_arg, save_dir:str, data_dir:str, dr_tech1:str):

    global lock
    parent_dir=os.path.join(embed_dir,dr_tech1,dataset)
    if dr_tech1=='PCA':
        X=np.load(os.path.join(parent_dir,f'{dataset}_{dr_arg[0]}.npy'))
    elif dr_tech1=='DiffRed':
        X=np.load(os.path.join(parent_dir, f'{dataset}_{dr_arg[0]+dr_arg[1]}_{dr_arg[0]}_{dr_arg[1]}.npy'))
    
    lock.acquire()
    if dr_tech1=='DiffRed':
        data_path=os.path.join(data_dir, dataset, f'{dr_arg[0]}_{dr_arg[1]}_{dr_arg[2]}.npy')
    elif dr_tech1=='PCA':
        data_path=os.path.join(data_dir,dataset,f'{dataset}_{dr_arg[0]}_pca.npy')

    A=np.load(data_path)
    lock.release()
    constant=LA.norm(A, ord='fro')/LA.norm(X, ord='fro')
    X=constant*X

    lock.acquire()
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(os.path.join(save_dir, dr_tech1)):
        os.mkdir(os.path.join(save_dir,dr_tech1))
    if not os.path.exists(os.path.join(save_dir,dr_tech1,dataset)):
        os.mkdir(os.path.join(save_dir,dr_tech1,dataset))
    lock.release()
    if dr_tech1=='PCA':
        save_path=os.path.join(save_dir, dr_tech1, dataset,f'{dataset}_{dr_arg[0]}.npy')
    elif dr_tech1=='DiffRed':
        save_path=os.path.join(save_dir, dr_tech1, dataset, f'{dataset}_{dr_arg[0]+dr_arg[1]}_{dr_arg[0]}_{dr_arg[1]}.npy')
    np.save(save_path, X)

def main():
    args=parse_arguments()
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)
    target_dims=[int(x) for x in args.target_dims]
    if args.dr_tech=='DiffRed':
        k1=[int(x) for x in args.k1]
        k2=[target_dims[x]-k1[x] for x in range(len(target_dims))]
        if args.max_iter_list=='True':
            max_iter=[int(x) for x in args.max_iter]
        else:
            max_iter=[int(args.max_iter[0]) for x in range(len(k1))]
        dr_args=list(zip(k1,k2,max_iter))

        results=[pool.apply_async(renormalize_embeddings, args=(args.dataset, args.embed_dir, dr_arg, args.save_dir, args.data_dir, args.dr_tech)) for dr_arg in dr_args]
    else:
        dr_args=target_dims
        results=[pool.apply_async(renormalize_embeddings, args=(args.dataset, args.embed_dir, [dr_arg], args.save_dir, args.data_dir, args.dr_tech)) for dr_arg in dr_args]
    # if args.setting_list=='True':
    #     settings=args.settings
    # else:
    #     settings=[args.settings[0] for i in range(len(target_dims))]
    
    # combinations =product(target_dims,settings)
    

    # results=[pool.apply_async(renormalize_embeddings, args=(args.dataset, args.embed_dir,target_dim, setting, args.save_dir, args.data_dir)) for target_dim,setting in combinations]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()