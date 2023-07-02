# This python script can download and save all the datasets in

import argparse
import numpy as np
import numpy.linalg as LA
import zipfile
import os
import requests
import inspect
from tqdm import tqdm
from datasets import *
import datasets
import argparse


CACHE_DIR='./temp_data'
DATASET_DIR='./datasets'

def parse_arguments():
    parser=argparse.ArgumentParser(description='Download and save datasets')
    parser.add_argument('--save_dir','-s',help='Directory where datasets are stored')
    parser.add_argument('--datasets','-d', nargs='+',help='List of names of the dataset')
    args=parser.parse_args()
    return args

# def dataset_create(dataset_obj):
#     global DATASET_DIR, CACHE_DIR
#     dataset_obj.download()
#     dataset_obj.preprocess()
#     dataset_obj.save_as_numpy()

def create_datasets(objs,datasets):

    global DATASET_DIR, CACHE_DIR

    for dataset in tqdm(datasets):
        obj=objs[dataset.lower()]
        obj.download()
        obj.preprocess()
        obj.save_as_numpy()

def main():
    args=parse_arguments()

    global DATASET_DIR, CACHE_DIR
    url={
        'Bank':'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
        'geneRNASeq': 'https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip',
        'ElectricityLoadDiagrams':'https://archive.ics.uci.ed/static/public/321/electricityloaddiagrams20112014.zip',
        'DIV2k': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'

    }

    objs={'bank':Bank(url['Bank']),'fmnist': FMNIST(), 'cifar10': Cifar10(),'reuters30k':Reuters30k(),'genernaseq':geneRNASeq(url['geneRNASeq']),'electricityloaddiagrams': ElectricityLoadDiagrams(url['ElectricityLoadDiagrams']),'div2k':DIV2K(url['DIV2k'])}

    create_datasets(objs,args.datasets)    
    # for obj in tqdm(objs):
    #     dataset_create(obj)
    
    # div2k=DIV2K(None)
    # div2k.preprocess()
    # div2k.save_as_numpy()


if __name__=="__main__":
    main()



    # module_classes=inspect.getmembers(datasets,inspect.isclass)

    # dataset_instances=[]
    
    # for class_name,class_obj in module_classes:
    #     if class_name!='Dataset' and class_name!='DIV2K':
    #         instance=class_obj()
    #         dataset_instances.append(instance)
    
    # for dataset_obj in dataset_instances:
    #     dataset_obj.download()
    #     dataset_obj.






        



