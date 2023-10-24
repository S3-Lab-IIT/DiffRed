import numpy as np
import numpy.linalg as LA
import math as m
import os
import sys
sys.path.append('../..')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from tqdm import tqdm 
from DiffRed.utils import calculate_singular_values
from multiprocessing import cpu_count, Pool

def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save the spectral plots of the datasets')
    parser.add_argument('--save_dir', '-s', help='Directory where the plots are to be saved', default='./spectral_plots')
    parser.add_argument('--datasets','-d', nargs='+', help='Datasets whose plots are to be computed', default='all')
    parser.add_argument('--data_dir', help='Directory where datasets are stored', default='./normalized_data')
    parser.add_argument('--sample_percentage', help='Percentage of data to be sampled for computation')
    parser.add_argument('--energy_threshold', '-e', help='List of energy thresholds to annotate seperated by space', default='0.90 0.95 0.98 0.99')
    parser.add_argument('--singular_dir', help='Directory to save singular values/Directory where singular values are saved', default='./norm_singular_values')
    parser.add_argument('--clip_at', nargs='+', help='List of values seperated by space specifying the value of x-axis for each dataset at which to clip the plot. Only to be used when energy threshold=None')

    args=parser.parse_args()
    return args

def opt_k1(sigma:np.ndarray, energy_threshold:int)-> int:
    sum_sq=np.sum(sigma**2)
    for k1 in range(len(sigma)):
        energy=np.sum(sigma[:k1]**2)/sum_sq
        if energy>=energy_threshold:
            return k1
    return len(sigma)

# def opt_dimensions(sigma:np.ndarray, target_dimension: int, energy_threshold=0.98) -> tuple[int, int]:
#     # sigma=calculate_singular_values(A)
#     sum_sq=np.sum(sigma**2)
#     for k1 in range(len(sigma)):
#         energy=np.sum(sigma[:k1]**2)/sum_sq
#         if energy>=energy_threshold:
#             if not target_dimension-k1<0:
#                 return k1, target_dimension-k1
#             else:
#                 return k1,0

def plot_data(dataset:str, DATA_DIR:str, SAVE_DIR:str, SING_DIR:str,energy_threshold: list[float],clip:int):
    X=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'),allow_pickle=True)
    
    if energy_threshold is not None:
        X=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'),allow_pickle=True)
        
        # print(os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')))

        if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
            sigma=calculate_singular_values(X)
        else:
            sigma=np.load(os.path.join(SING_DIR,f'{dataset}.npy'))

        k1_values=[opt_k1(sigma, x) for x in energy_threshold]
        # k1_values=[opt_dimensions(sigma,40,x)[0] for x in energy_threshold]

        # print(f'k1 values for {dataset}: ',k1_values)

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
        axes[1,2].set_title('Top 200')

        colors = cm.get_cmap('tab10', len(k1_values))

        for i,x in enumerate(k1_values):
            for ax in axes.flatten():
                ax.axvline(x=x,color=colors(i), linestyle='--')

                ax.text(x,0,f'{x} ',color='black',ha='right', va='bottom', weight='bold')
        
        plt.subplots_adjust(top=0.9)

        plt.suptitle(f'Singular Values of {dataset}')

        # sigmadjust spacing between subplots
        fig.tight_layout()
        legend_title='Energy Threshold'
        legend_labels=[f'{e}' for e in energy_threshold ]
        plt.figlegend(handles=[plt.Line2D([0], [0], color=colors(i), linestyle='--') for i in range(len(k1_values))],
                labels=legend_labels, title=legend_title, loc='upper left', ncol=2)
        plt.savefig(os.path.join(SAVE_DIR, f'{dataset}.png'))

        if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
            np.save(os.path.join(SING_DIR,f'{dataset}.npy'),sigma)

    else:
        X=np.load(os.path.join(DATA_DIR,dataset, 'X.npy'),allow_pickle=True)
        
        # print(os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')))

        if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
            sigma=calculate_singular_values(X)
        else:
            sigma=np.load(os.path.join(SING_DIR,f'{dataset}.npy'))

        if clip:
            plt.plot(sigma[:clip])
        else:
            plt.plot(sigma)
        plt.title(f'Spectral Plot of {dataset}')
        plt.xlabel('Principal Components')
        plt.ylabel('Singular Values')
        plt.savefig(os.path.join(SAVE_DIR, f'{dataset}.png'))

        if not os.path.exists(os.path.join(SING_DIR,f'{dataset}.npy')):
            np.save(os.path.join(SING_DIR,f'{dataset}.npy'),sigma)


def main():
    args=parse_arguments()

    clips=[]
    if args.clip_at:
        for i in args.clip_at:
            if i=='None':
                clips.append(None)
            else:
                clips.append(int(i))
    if args.energy_threshold=='None':
        energy_threshold=None
    else:
        energy_threshold=[float(x) for x in args.energy_threshold.split()]

    if args.datasets[0].lower()=='all':
        datasets=os.listdir(args.data_dir)
    else:
        datasets=args.datasets
    pbar=tqdm(range(len(datasets)), desc='Plotting')
    # for dataset in pbar:
    #     pbar.set_description(f'Plotting {dataset}...')
    #     plot_data(dataset, args.data_dir, args.save_dir,args.singular_dir,energy_threshold)

    num_cores=cpu_count()
    pool=Pool(processes=num_cores)

    if len(clips)>0:
        results=[pool.apply_async(plot_data, args=(datasets[i],args.data_dir,args.save_dir,args.singular_dir,energy_threshold,clips[i])) for i in pbar]
    else:
         results=[pool.apply_async(plot_data, args=(datasets[i],args.data_dir,args.save_dir,args.singular_dir,energy_threshold,None)) for i in pbar]

    for result in results:
        result.wait()
    pool.close()
    pool.join()


if __name__=="__main__":
    main()