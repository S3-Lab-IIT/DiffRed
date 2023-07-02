import numpy as np
import numpy.linalg as LA
import math as m
import os
import argparse
from metrics import distance_matrix
from tqdm import tqdm
from multiprocessing import cpu_count,Pool


def parse_arguments():
    parser=argparse.ArgumentParser(description='Compute and save embedding matrices')
    parser.add_argument('--save_dir','-s', help='Path to the directory where the embeddings are to be saved', default='./embeddings')
    parser.add_argument('--datasets','-d',nargs='+',help='List of names of the dataset ')
    parser.add_argument('--data_dir', help='Directory where the datasets are stored', default='./datasets' )
    parser.add_argument('--target_dims', help='List of target dimensions at which embeddings are to computed', nargs='+')

    args=parser.parse_args()
    return args

def compute_embeddings(dataset:str, ):
    pass