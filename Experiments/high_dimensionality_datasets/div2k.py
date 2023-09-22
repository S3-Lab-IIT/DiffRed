import argparse
import numpy as np
import numpy.linalg as LA
import zipfile
import argparse
import numpy as np
import numpy.linalg as LA
import zipfile
import os
import requests
import urllib
import tarfile
from urllib.parse import urlparse
from urllib.request import urlretrieve
import pandas as pd
from sklearn.preprocessing import LabelEncoder,MultiLabelBinarizer
# from tensorflow.keras.datasets import fashion_mnist, cifar10
# import nltk
# from nltk.corpus import reuters
# from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2

def parse_arguments():
    parser=argparse.ArgumentParser(description='Download and preprocess the DIV2K dataset')

    parser.add_argument('-d', '--dataset', help='Name of the dataset', choices=['DIV2k', 'APTOS'])

    parser.add_argument('--data_dir', help='Directory where final preprocessed data is to be stored', default='./datasets')

    parser.add_argument('--cache_dir', help='Directory where downloaded file is to be stored', default='./temp_dir')

    args=parser.parse_args()

    return args

class Dataset:

    global DATASET_DIR, CACHE_DIR
    def __init__(self, name, url):
        self.name=name
        self.url=url
        self.path=os.path.join(DATASET_DIR,self.name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
    
    def download(self):
        raise NotImplementedError("Download method not implemented")
    
    
    def save_as_numpy(self):
        np.save(os.path.join(self.path,'X.npy'),self.X)
        np.save(os.path.join(self.path,'y.npy'),self.y)

    def set_data(self,X,y):
        self.X=X
        self.y=y

class DIV2K(Dataset):

    global DATASET_DIR, CACHE_DIR

    def __init__(self,url):
        super().__init__('DIV2k',url)
        self.downloaded=False
    
    def download(self):

        self.cache_path=os.path.join(CACHE_DIR, self.name)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        
        parsed_url=urlparse(self.url)
        file_name=os.path.basename(parsed_url.path)
        self.zip_path=os.path.join(self.cache_path, file_name)

        try:
            urlretrieve(self.url, self.zip_path)
            self.downloaded=True
        except urllib.error.URLError as e:
            print("Failed to download the file ",e)
            self.downloaded=False
        
    def preprocess(self,width=2048 , height=1080):

        if not self.downloaded:
            self.cache_path=os.path.join(CACHE_DIR, self.name)
            if os.path.exists(os.path.join(self.cache_path,'DIV2K_train_HR.zip')):

                self.zip_path=os.path.join(self.cache_path,'DIV2K_train_HR.zip')
                with zipfile.ZipFile(self.zip_path,'r') as zip_file:
                    zip_file.extractall(self.cache_path)
                os.remove(self.zip_path)

        self.img_dir=os.path.join(self.cache_path, 'DIV2K_train_HR')

        X=[]

        for img in os.listdir(self.img_dir):
            img_path=os.path.join(self.img_dir,img)
            image=cv2.imread(img_path)
            resized_image = cv2.resize(image, (width, height))
            image_vector = resized_image.flatten()
            X.append(image_vector)
        
        X=np.array(X)

        self.set_data(X,None)
    

if __name__=="__main__":

    args=parse_arguments()

    DATASET_DIR, CACHE_DIR= args.data_dir, args.cache_dir
    if args.dataset=='DIV2k':

        if not os.path.exists(os.path.join(DATASET_DIR, args.dataset, 'X.npy')):
            div2k=DIV2K(url='http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip')

            div2k.download()
            div2k.preprocess()
            div2k.save_as_numpy()
        else:
            print('Dataset already exists, clear the dataset directory to download again')
    elif args.dataset=='APTOS':
        print("Code not implemented")
    
    else:
        print("Dataset not valid")

