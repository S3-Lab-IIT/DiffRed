import numpy as np
import numpy.linalg as LA
import math as m
import os
import sys
sys.path.append('../..')
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm 
from DiffRed.utils import calculate_singular_values
from multiprocessing import cpu_count, Pool

def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save the spectral plots of the datasets')
    parser.add_argument('--save_dir', '-s', help='Directory where the plots are to be saved', default='./spectral_plots')
    parser.add_argument('--datasets','-d', nargs='+', help='Datasets whose plots are to be computed', default='all')
    parser.add_argument('--data_dir', help='Directory where datasets are stored', default='./datasets')
    parser.add_argument('--sample_percentage', help='Percentage of data to be sampled for computation')
    parser.add_argument('--singular_dir', help='Directory to save singular values', default='./singular_values')
    args=parser.parse_args()
    return args

def plot_data(dataset:str, DATA_DIR:str, SAVE_DIR:str, SING_DIR:str):
    X=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'),allow_pickle=True)
    sigma=calculate_singular_values(X)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Plot sigma
    axes[0, 0].plot(sigma)
    axes[0, 0].set_title('All')

    # Plot sigma[:50]
    axes[0, 1].plot(sigma[:25])
    axes[0, 1].set_title('Top 25')

    # Plot sigma[:25]
    axes[0, 2].plot(sigma[:50])
    axes[0, 2].set_title('Top 50')

    # Plot sigma[:100]
    axes[1, 0].plot(sigma[:100])
    axes[1, 0].set_title('Top 100')

    # Plot sigma[:200]
    axes[1, 1].plot(sigma[:150])
    axes[1, 1].set_title('Top 150')

    axes[1,2].plot(sigma[:200])
    axes[1,1].set_title('Top 200')

    plt.suptitle(f'Singular Values of {dataset}')

    # sigmadjust spacing between subplots
    fig.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'{dataset}.png'))
    np.save(os.path.join(SING_DIR,f'{dataset}.npy'),sigma)


def main():
    args=parse_arguments()

    if args.datasets[0].lower()=='all':
        datasets=os.listdir(args.data_dir)
    else:
        datasets=args.datasets
    pbar=tqdm(datasets)
    for dataset in pbar:
        pbar.set_description(f'Plotting {dataset}...')
        plot_data(dataset, args.data_dir, args.save_dir,args.singular_dir)

if __name__=="__main__":
    main()


    
    