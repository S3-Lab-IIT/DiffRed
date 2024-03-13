# This python script can download and save all the datasets in

import argparse
import numpy as np
import numpy.linalg as LA
import zipfile
import os
import requests
import inspect
from tqdm import tqdm
from Datasets import *
import Datasets
import argparse


CACHE_DIR='./temp_data'
DATASET_DIR='./datasets'

def parse_arguments():
    parser=argparse.ArgumentParser(description='Download and save datasets')
    parser.add_argument('--save_dir','-s',help='Directory where datasets are stored', default='./datasets/')
    parser.add_argument('--datasets','-d', nargs='+',help='List of names of the dataset')
    args=parser.parse_args()
    return args

# def dataset_create(dataset_obj):
#     global DATASET_DIR, CACHE_DIR
#     dataset_obj.download()
#     dataset_obj.preprocess()
#     dataset_obj.save_as_numpy()

def create_datasets(objs:dict[str,Dataset],datasets):

    global DATASET_DIR, CACHE_DIR

    for dataset in tqdm(datasets):
        obj=objs[dataset.lower()]
        obj.download()
        obj.preprocess()
        # if dataset=='hatespeech':
        #     pass
        # else:
        #     obj.save_as_numpy()
        obj.save_as_numpy()

def main():
    args=parse_arguments()

    global DATASET_DIR, CACHE_DIR
    url={
        'Bank':'https://archive.ics.uci.edu/static/public/222/bank+marketing.zip',
        'geneRNASeq': 'https://archive.ics.uci.edu/static/public/401/gene+expression+cancer+rna+seq.zip',
        'DIV2k': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'hatespeech':{'X': 'https://mespadoto.github.io/proj-quant-eval/data/hatespeech/X.npy', 'y':'https://mespadoto.github.io/proj-quant-eval/data/hatespeech/y.npy'}

    }

    objs={'bank':Bank(url['Bank']),'fmnist': FMNIST(), 'cifar10': Cifar10(),'reuters30k':Reuters30k(),'genernaseq':geneRNASeq(url['geneRNASeq']),'div2k':DIV2K(url['DIV2k']), 'hatespeech':hatespeech('hatespeech',url['hatespeech'])}


    create_datasets(objs, args.datasets)


if __name__=="__main__":
    main()







        



