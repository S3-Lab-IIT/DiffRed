import numpy as np
import numpy.linalg as LA
import math as m
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from tqdm import tqdm 
from compute_stable_rank import stable_rank
from multiprocessing import cpu_count, Pool


def parse_arguments():

    parser=argparse.ArgumentParser(description='Plot the residual stable rank against k1')

    parser.add_argument('--save_dir', '-s', help='Directory where the plots are to be saved', default='./stable_rank_plots')
    parser.add_argument('--datasets','-d', nargs='+', help='Datasets whose plots are to be computed', default='all')
    parser.add_argument('--data_dir', help='Directory where datasets are stored', default='./normalized_data')
    parser.add_argument('--singular_dir', help='Directory to save singular values/Directory where singular values are saved', default='./norm_singular_values')
    parser.add_argument('--clip_at', '-c', help='Value of k1 at which plot must be clipped', type=int, default=None)

    parser.add_argument('--all_in_one', help='True if you want all plots in a single png', default="False")

    args=parser.parse_args()

    return args

def set_size(width, fraction=1):
    """
    COURTESY: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    
    Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_stable_rank(dataset:str, SAVE_DIR:str, SING_DIR:str, clip:int=None):

    sing_path=os.path.join(SING_DIR, f'{dataset}.npy')

    if clip is None:
        if not os.path.exists(sing_path):
            print(f'Singular values for {dataset} not computed')
        else:
            sigma=np.load(sing_path)
            k1_vals=[x for x in range(len(sigma))]
            stable_ranks=[]

            for k1 in k1_vals:
                stable_ranks.append(stable_rank(sigma[k1:]))
    else:
        if not os.path.exists(sing_path):
            print(f'Singular values for {dataset} not computed')
        else:
            sigma=np.load(sing_path)
            k1_vals=[x for x in range(clip)]
            stable_ranks=[]

            for k1 in k1_vals:
                stable_ranks.append(stable_rank(sigma[k1:]))
    
    plt.plot(k1_vals, stable_ranks)
    plt.xlabel('k1')
    plt.ylabel('stable rank')
    plt.title(f'Stable rank vs k1 for {dataset}')

    plt_name= f'{dataset}_clip={clip}.png' if not clip is None else f'{dataset}.png'
    plt.savefig(os.path.join(SAVE_DIR, plt_name))

def all_in_one(dataset:str, SAVE_DIR:str, SING_DIR:str, clip:int):
    sing_path=os.path.join(SING_DIR, f'{dataset}.npy')

    if clip is None:
        if not os.path.exists(sing_path):
            print(f'Singular values for {dataset} not computed')
        else:
            sigma=np.load(sing_path)
            k1_vals=[x for x in range(len(sigma))]
            stable_ranks=[]

            for k1 in k1_vals:
                stable_ranks.append(stable_rank(sigma[k1:]))
        return k1_vals, stable_ranks
    else:
        if not os.path.exists(sing_path):
            print(f'Singular values for {dataset} not computed')
        else:
            sigma=np.load(sing_path)
            k1_vals=[x for x in range(clip)]
            stable_ranks=[]

            for k1 in k1_vals:
                stable_ranks.append(stable_rank(sigma[k1:]))
        return k1_vals, stable_ranks

def main():

    args=parse_arguments()

    if args.datasets[0].lower()=='all':
        datasets=os.listdir(args.data_dir)
    else:
        datasets=args.datasets
    
    
    if args.all_in_one=="False":
        num_cores=cpu_count()
        pool=Pool(processes=num_cores)

        results=[pool.apply_async(plot_stable_rank, args=(dataset, args.save_dir, args.singular_dir,args.clip_at)) for dataset in datasets]

        for result in results:
            result.wait()
        
        pool.close()
        pool.join()
    
    else:
        # plt.rcParams['text.usetex'] = True
        num_cores=cpu_count()
        pool=Pool(processes=num_cores)

        results=[pool.apply_async(all_in_one, args=(dataset, args.save_dir, args.singular_dir,args.clip_at)) for dataset in datasets]

        for result in results:
            result.wait()
        
        plot_data={}
        for i in range(len(results)):
            
            k1_vals, stable_ranks=results[i].get()

            plot_data[datasets[i]]=(k1_vals, stable_ranks)
        
        cols=2
        rows= len(datasets)//2 if len(datasets)%2==0 else len(datasets)//2 +1
        fig, axes= plt.subplots(nrows=rows, ncols=cols, figsize=(5,6))
        for i, dataset_name in enumerate(datasets):
            row = i // cols
            col = i % cols
            x, y = plot_data[dataset_name]

            if row>1:
                ax = axes[row, col]
            else:
                ax=axes[col]
            ax.plot(x, y)
            # max_y = max(y)
            # max_x = x[y.index(max_y)]

            # Annotate the maximum value
            # ax.annotate(
            #     f"Max:({max_x:.2f}, {max_y:.2f})",
            #     xy=(max_x, max_y),
            #     xytext=(max_x, max_y + 0.1),  # Adjust text position for visibility
            #     ha='left',
            #     fontsize=8,
            # )
            ax.set_xlim(min(x), 1.2*max(x))
            ax.set_ylim(min(y), 1.5*max(y))
            ymax = max(y)
            xmax= x[y.index(ymax)]
            text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
            if not ax:
                ax=plt.gca()
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
            kw = dict(xycoords='data',textcoords="axes fraction",
                    arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
            ax.set_xlabel('k1')
            ax.set_ylabel(r'Stable Rank')
            ax.set_title(dataset_name)
        fig.tight_layout()
        for i in range(len(datasets), rows * cols):
            fig.delaxes(axes.flatten()[i])
        plt.suptitle(r'Stable rank vs k1')
        plt.subplots_adjust(top=0.85)
        
        plt_name=f'all_in_one_clip={args.clip_at}'
        plt.savefig(os.path.join(args.save_dir, plt_name))

if __name__=="__main__":
    main()

