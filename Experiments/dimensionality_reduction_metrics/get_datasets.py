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

CACHE_DIR='./temp_data'
DATASET_DIR='./datasets'

def dataset_create(dataset_obj):
    global DATASET_DIR, CACHE_DIR
    dataset_obj.download()
    dataset_obj.preprocess()
    dataset_obj.save_as_numpy()

def main():

    global DATASET_DIR, CACHE_DIR
    url={
        'Bank':'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
        'geneRNASeq': 'https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip',
        'ElectricityLoadDiagrams':'https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip'
    }

    objs=[]
    objs.extend([Bank(url['Bank']), FMNIST(),Cifar10(),Reuters100k(),Reuters30k(),geneRNASeq(url['geneRNASeq']),ElectricityLoadDiagrams(url['ElectricityLoadDiagrams'])])

    for obj in tqdm(objs):
        dataset_create(obj)
    
    div2k=DIV2K(None)
    div2k.preprocess()
    div2k.save_as_numpy()


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






        



