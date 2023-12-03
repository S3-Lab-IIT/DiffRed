# For T-SNE:
# Bank: python3 m1_matched_embeddings.py -d Bank --dr_tech T-SNE -t 2 --settings setting2
# FMnist: python3 m1_matched_embeddings.py -d FMnist --dr_tech T-SNE -t 2 --settings setting4
# Cifar10: python3 m1_matched_embeddings.py -d Cifar10 --dr_tech T-SNE -t 2 --settings setting4
# geneRNASeq: python3 m1_matched_embeddings.py -d geneRNASeq --dr_tech T-SNE -t 2 --settings setting4
# hatespeech: python3 m1_matched_embeddings.py -d hatespeech --dr_tech T-SNE -t 2 --settings setting5
# Reuters30k: python3 m1_matched_embeddings.py -d Reuters30k --dr_tech T-SNE -t 2 --settings setting4 
# For Reuters30k, the directory name is M-TSNE

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
from settings import SETTINGS
lock=Lock()

def parse_arguments():
    parser=argparse.ArgumentParser(description='Renormalize UMap/T-SNE for stress computation as described in the paper (UMap2/T-SNE2). ')

    parser.add_argument('--dataset', '-d', help='Dataset for which renormalization is to be done' )
    parser.add_argument('--data_dir', help='Directory where the normalized datasets are stored', default='../normalized_data/')
    parser.add_argument('--embed_dir', help='Directory where UMap/T-SNE embeddings are stored', default='./embeddings')
    parser.add_argument('--dr_tech', help='DR technique whose renormalization is to be done', choices=['T-SNE','UMap'])
    parser.add_argument('--target_dims', '-t',nargs='+', help='List of target dimensions (seperated by space) for which the renormalization is to be done')
    parser.add_argument('--settings', '-s', nargs='+',help='List of settings(seperated by space) corresponding to target dimension for which the renormalization is to be done. \'all\' for all settings')
    parser.add_argument('--setting_list', help='Whether a list of settings is provided or a single setting is provided which is to be used all target dimensions', default='True')
    parser.add_argument('--save_dir', help='Directory where to save the re-normalized embeddings', default='./embeddings')

    args=parser.parse_args()
    return args

def renormalize_embeddings(dataset:str, embed_dir:str, target_dim:int, setting:str, save_dir:str, data_dir:str, dr_tech:str):

    global lock
    X=np.load(os.path.join(embed_dir, dataset, dr_tech, f'{dataset}_{target_dim}_{setting}.npy'))
    
    lock.acquire()
    A=np.load(os.path.join(data_dir,dataset, 'X.npy'))
    lock.release()
    constant=LA.norm(A, ord='fro')/LA.norm(X, ord='fro')
    X=constant*X

    
    save_path=os.path.join(save_dir,dataset, f'{dr_tech}2')
    lock.acquire()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    lock.release()
    

    np.save(os.path.join(save_dir, dataset,f'{dr_tech}2', f'{dataset}_{target_dim}_{setting}.npy'),X)

def main():
    args=parse_arguments()

    target_dims=[int(x) for x in args.target_dims]

    if args.dataset=='Reuters30k' and args.dr_tech=='T-SNE':
        args.dr_tech='M-TSNE'
    if args.setting_list=='True':
        settings=args.settings
    else:
        settings=[args.settings[0] for i in range(len(target_dims))]
    
    combinations =product(target_dims,settings)
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    results=[pool.apply_async(renormalize_embeddings, args=(args.dataset, args.embed_dir,target_dim, setting, args.save_dir, args.data_dir,args.dr_tech)) for target_dim,setting in combinations]

    for result in results:
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()