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

from dr_techniques import DR_MAPS
from settings import SETTINGS

lock=Lock()

def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute and save embedding matrices')

    parser.add_argument('--save_dir', '-s', help='Path to the directory where embeddings are to be saved', default='./embeddings')
    parser.add_argument('--dataset', '-d', help='Dataset')
    parser.add_argument('--data_dir', help='Directory where datasets are stored', default='../normalized_data')
    parser.add_argument('--target_dims', '-t', help='List of target dimensions seperated by space', nargs='+')
    parser.add_argument('--dr_tech', help='Dimensionality Reduction technique to use', choices=['UMap', 'T-SNE','S-PCA', 'K-PCA', 'M-TSNE'])
    parser.add_argument('--setting', help='Which setting to use for the dr technique. Enter \'def\' for default or \'setting1\', \'setting2\' for custom settings stored in settings.py and \'all\' for computing on all settings', default='def')

    args=parser.parse_args()
    return args

def compute_embeddings(dataset:str, DATA_DIR:str,SAVE_DIR:str, dr_tech:str, target_dim:int, setting:dict):

    X=np.load(os.path.join(DATA_DIR, dataset, 'X.npy'), allow_pickle=True)

    
    if not os.path.exists(os.path.join(SAVE_DIR, dataset)):
       lock.acquire()
       os.mkdir(os.path.join(SAVE_DIR, dataset))
       lock.release()

    
    if not os.path.exists(os.path.join(SAVE_DIR, dataset, dr_tech)):
        lock.acquire()
        os.mkdir(os.path.join(SAVE_DIR, dataset,dr_tech))
        lock.release()

    if os.path.exists(os.path.join(SAVE_DIR, dataset, dr_tech, f'{dataset}_{target_dim}_{setting}.npy')):

        print(f'Embeddings already saved for dataset: {dataset}, dr_tech: {dr_tech}, target dimension: {target_dim}, setting: {setting}')
        return
    
    if not (dr_tech=='T-SNE' or dr_tech=='M-TSNE') :
        Z= DR_MAPS[dr_tech](X,target_dim,SETTINGS[dr_tech][setting])
    elif dr_tech=='T-SNE' or dr_tech=='M-TSNE':
        Z=DR_MAPS[dr_tech](X,SETTINGS[dr_tech][setting])
    else:
        print("Wrong DR Technique provided!")
        return

    np.save(os.path.join(SAVE_DIR, dataset, dr_tech, f'{dataset}_{target_dim}_{setting}.npy'), Z)


def main():

    args=parse_arguments()
    target_dims=[int(x) for x in args.target_dims]

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    if not args.setting=='all':
        results=[pool.apply_async(compute_embeddings, args=(args.dataset, args.data_dir, args.save_dir, args.dr_tech, target_dims[i], args.setting)) for i in range(len(target_dims))]

    else:
        all_settings=[k for k in SETTINGS[args.dr_tech].keys() ]
        combinations =product(target_dims,all_settings)
        results=[pool.apply_async(compute_embeddings, args=(args.dataset, args.data_dir,args.save_dir, args.dr_tech, target_dim, setting)) for target_dim, setting in combinations]

    for result in tqdm(results, desc=f'Computing {args.dataset} {args.dr_tech}'):
        result.wait()
    
    pool.close()
    pool.join()

if __name__=="__main__":
    main()
