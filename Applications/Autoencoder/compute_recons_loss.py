import argparse
from multiprocessing import Pool, cpu_count, Lock
import pandas as pd
from datetime import datetime
import os
from DiffCoder import DiffCoder
from DiffCoder.utils import load_dataset, mse


lock=Lock()



def parse_arguments():

    parser=argparse.ArgumentParser(description='Compute the reconstruction loss using DiffCoder, PCA, other autoencoders')
    parser.add_argument('--dataset', '-d', help='Name of the dataset to be used')
    parser.add_argument('--ae', help='Name of the autoencoder to be used', choices=['DiffCoder', 'PCA', 'NN'])
    parser.add_argument('--data_dir', help='Name of the directory where datasets(original, non-normalized) are stored', default='../../Experiments/dimensionality_reduction_metrics/datasets/')
    parser.add_argument('--target_dims', '-t', nargs='+', help='List of target dimensions to use')
    parser.add_argument('--k1', nargs='+', help='(FOR DIFFCODER ONLY) List of k1 values corresponding to each target dimension')
    parser.add_argument('--save_dir', help='Directory where results are to be saved', default='./results')
    parser.add_argument('--file_name', '-f', help='Name of the results file')

    args=parser.parse_args()

    return args

def calculate_mse(dataset:str, ae:str, data_dir:str, target_dim:int, save_dir:str, file_name:str, k1:int=None):
    global lock
    if ae=='DiffCoder':
        A=load_dataset(data_dir, dataset)

        dc=DiffCoder(k1=k1, k2=target_dim-k1)

        A_res=dc.reconstruct(A)

        metric=mse(A,A_res)

        lock.acquire()
        # if not os.path.exists(os.path.join(save_dir, dataset)):
        #     os.mkdir(os.path.join(save_dir, dataset))
        columns=['Timestamp', 'Autoencoder', 'Dataset', 'Target Dimension','k1','k2','MSE Reconstruction Loss']
        if not os.path.exists(os.path.join(save_dir, f'{file_name}.xlsx')):
            df=pd.DataFrame(columns=columns)
            df.to_excel(os.path.join(save_dir,f'{file_name}.xlsx'), index=False, sheet_name=dataset)
        
        new_row=[datetime.now().strftime("%Y-%m-%d %H:%M:%S"),ae,dataset, target_dim,k1,target_dim-k1,metric]
        excel=pd.ExcelFile(os.path.join(save_dir, f'{file_name}.xlsx'))
        if dataset not in excel.sheet_names:
            df=pd.DataFrame(columns=columns)

            with pd.ExcelWriter(excel,mode='a', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=dataset,index=False)
        
        result_sheet=pd.read_excel(os.path.join(save_dir, f'{file_name}.xlsx'),sheet_name=dataset)

        result_sheet.loc[len(result_sheet.index)]=new_row
        result_sheet.to_excel(os.path.join(save_dir, f'{file_name}.xlsx'), index=False, sheet_name=dataset)
        lock.release()


def main():
    args=parse_arguments()
    num_cores=cpu_count()
    pool=Pool(processes=num_cores)
    target_dims=[int(x) for x in args.target_dims]

    if args.ae=='DiffCoder':
        k1=[int(x) for x in args.k1]
        results=[pool.apply(calculate_mse, args=(args.dataset, args.ae, args.data_dir, target_dims[i], args.save_dir, args.file_name,k1[i])) for i in range(len(target_dims))]
    else:
        results=[pool.apply_async(calculate_mse, args=(args.dataset, args.ae, args.data_dir, target_dims[i], args.save_dir, args.file_name,None)) for i in range(len(target_dims))]
    
    for result in results:
        result.wait()
    
    pool.close()
    pool.join()


if __name__=="__main__":
    main()
    

    


