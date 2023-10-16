import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from tqdm import tqdm



def parse_arguments():

    parser=argparse.ArgumentParser(description='For figures used in the paper')

    parser.add_argument('--metric', '-m', help='Metric whose results are to be compiled', default='stress', choices=['Stress', 'M1'])

    parser.add_argument('--result_dir', '-r', help='Directory where DiffRed results are stored', default='./results/')

    parser.add_argument('--other_res', '-o', help='Directory where results of other dr techniques are stored', default='./results/other_dr_techniques/')

    parser.add_argument('--dr_tech', help='Which dr technique to compile', choices=['PCA', 'RMap', 'K-PCA', 'S-PCA', 'UMap', 'T-SNE', 'DiffRed', 'UMap2', 'all'], default='all')

    parser.add_argument('--save_dir', help='Directory where results are to be saved', default='./results/compiled_results/')

    parser.add_argument('--dataset', '-d', help='Dataset for which results have to be compiled')

    parser.add_argument('--target_dims', nargs='+', help='(FOR RMAP ONLY) Target Dimensions for which the results are to be compiled')

    parser.add_argument('--setting', '-s', help='Which setting to use', default='def')

    args=parser.parse_args()

    return args

def min_row(group, col:str):
    return group.loc[group[col].idxmin()]

def compile(metric:str, RES_DIR:str, OTHER_RES_DIR:str, SAVE_DIR:str, dataset:str, setting:str, dr_tech:str, target_dims:list[int]):

    if setting=='def':

        if not os.path.exists(os.path.join(SAVE_DIR, metric)):
            os.mkdir(os.path.join(SAVE_DIR, metric))
        
        save_path=os.path.join(SAVE_DIR, metric, dr_tech)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        
        if dr_tech=='DiffRed':
            if not os.path.exists(os.path.join(save_path,f'{dataset}.xlsx')):
                
                diffred_df=pd.read_excel(os.path.join(RES_DIR, f'{metric}_results',f'{dataset}.xlsx'))

                compiled_df=diffred_df.groupby('Target Dimension', group_keys=False).apply(min_row, [metric])

                compiled_df.sort_values(by='Target Dimension', ascending=True, inplace=True)

                compiled_df[compiled_df.columns[3:]].to_excel(os.path.join(save_path,f'{dataset}.xlsx'), index=False)
            else:
                print("Already saved for dataset: ", dataset)
        
        elif dr_tech=='RMap':
            if not os.path.exists(os.path.join(save_path,f'{dataset}.xlsx')):

                metric1='stress' if metric=='Stress' else 'm1'
                stress_vals={}
                for target_dim in target_dims:

                    rmap_df=pd.read_excel(os.path.join(OTHER_RES_DIR, dr_tech, dataset, f'{metric1}_results_{dr_tech}_{target_dim}.xlsx'))

                    stress_vals[target_dim]={'mean': rmap_df[metric].mean(), 'stdev': rmap_df[metric].std()}
                
                flat_data = [{'Target Dimension': dim, metric: stats['mean'], 'stdev': stats['stdev']} for dim, stats in stress_vals.items()]

                compiled_df=pd.DataFrame(flat_data)
                compiled_df.sort_values(by='Target Dimension', ascending=True, inplace=True)

                compiled_df.to_excel(os.path.join(save_path,f'{dataset}.xlsx'), index=False)
            else:
                print("Already saved for dataset: ", dataset)
        
        else:

            if not os.path.exists(os.path.join(save_path,f'{dataset}.xlsx')):
                metric1='stress' if metric=='Stress' else 'm1'

                metric_df=pd.read_excel(os.path.join(OTHER_RES_DIR, dr_tech, f'{metric1}_results_{dr_tech}.xlsx'))

                metric_df=metric_df[metric_df['Dataset']==dataset]

                compiled_df=metric_df.groupby('Target Dimension', group_keys=False).apply(min_row, [metric])
                compiled_df.sort_values(by='Target Dimension', ascending=True, inplace=True)
                compiled_df[compiled_df.columns[3:]].to_excel(os.path.join(save_path,f'{dataset}.xlsx'), index=False)
            else:
                print("Already saved for dataset: ", dataset)


def main():
    args=parse_arguments()
    target_dims=[int(x) for x in args.target_dims]

    if not args.dr_tech=='all':
        compile(args.metric,args.result_dir, args.other_res,args.save_dir,args.dataset,args.setting,args.dr_tech, target_dims)
    else:

        dr_techs=['PCA', 'RMap', 'K-PCA', 'S-PCA', 'UMap', 'T-SNE', 'DiffRed', 'UMap2']

        for dr_tech in dr_techs:
            compile(args.metric,args.result_dir, args.other_res,args.save_dir,args.dataset,args.setting,dr_tech, target_dims)



if __name__=="__main__":
    main()







    