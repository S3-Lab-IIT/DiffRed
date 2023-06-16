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
from tensorflow.keras.datasets import fashion_mnist, cifar10
import nltk
from nltk.corpus import reuters
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2


CACHE_DIR='./temp_data'
DATASET_DIR='./datasets'
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

class Bank(Dataset):

    global DATASET_DIR, CACHE_DIR

    def __init__(self,url):
        super().__init__('Bank',url)


    def download(self):
        self.cache_path=os.path.join(CACHE_DIR, self.name)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        parsed_url=urlparse(self.url)
        file_name=os.path.basename(parsed_url.path)
        self.zip_path=os.path.join(self.cache_path, file_name)
        try: 
            urlretrieve(self.url,self.zip_path)
            # print('Downloaded Bank!')
        except urllib.error.URLError as e:
            print("Failed to download the file: ",e)
    
    def preprocess(self):
        with zipfile.ZipFile(self.zip_path,'r') as zip_file:
            zip_file.extract('bank.zip',self.cache_path)
        with zipfile.ZipFile(os.path.join(self.cache_path,'bank.zip'),'r') as zip_file:
            zip_file.extract('bank.csv',self.cache_path)
        os.remove(os.path.join(self.cache_path,'bank.zip'))
        os.remove(os.path.join(self.cache_path,'bank+marketing.zip'))
        
        data=pd.read_csv(os.path.join(self.cache_path,'bank.csv'),delimiter=';')
        label_encoder=LabelEncoder()

        cols_to_encode=['job','marital','education','default','housing','loan','contact','month','poutcome','y']
        for col in cols_to_encode:
            data[col]=label_encoder.fit_transform(data[col])
        
        y=data['y'].values
        X=data.drop('y',axis=1).values
        self.set_data(X,y)

class FMNIST(Dataset):

    global DATASET_DIR,CACHE_DIR

    def __init__(self):
        super().__init__('FMnist',None)
    
    def download(self):
        (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
        self.set_data(x_train.reshape((60000,784)),y_train)

    def preprocess(self):
        pass
    
class Cifar10(Dataset):

    global DATASET_DIR, CACHE_DIR

    def __init__(self):
        super().__init__('Cifar10',None)
    
    def download(self):
        (x_train,y_train),(x_test,y_test)=cifar10.load_data()
        self.set_data(x_train.reshape((50000,3072)),y_train)
    
    def preprocess(self):
        pass
        
class Reuters100k(Dataset):
    
    global DATASET_DIR, CACHE_DIR

    def __init__(self):
        super().__init__('Reuters10k',None)
    
    def download(self):
        nltk.download('reuters')
    
    def preprocess(self):
        documents=reuters.fileids()
        preprocessed_documents=[reuters.raw(doc_id) for doc_id in documents]
        model = Word2Vec(preprocessed_documents, vector_size=100000, window=5, min_count=1, workers=32)
        X = np.array([np.mean([model.wv[word] for word in doc], axis=0) for doc in preprocessed_documents])
        label_binarizer=MultiLabelBinarizer()
        y=label_binarizer.fit_transform([reuters.categories(doc_id) for doc_id in documents])
        self.set_data(X,y)

class Reuters30k(Dataset):

    global DATASET_DIR, CACHE_DIR

    def __init__(self):
        super().__init__('Reuters80k',None)
    
    def download(self):
        nltk.download('reuters')

    def preprocess(self):
        documents=reuters.fileids()
        vectorizer=TfidfVectorizer()
        X = vectorizer.fit_transform(reuters.raw(doc_id) for doc_id in documents)
        label_binarizer=MultiLabelBinarizer()
        y=label_binarizer.fit_transform([reuters.categories(doc_id) for doc_id in documents])
        self.set_data(X,y)


class geneRNASeq(Dataset):\

    global DATASET_DIR, CACHE_DIR

    def __init__(self,url):
        super().__init__('geneRNASeq', url)
    
    def download(self):
        self.cache_path=os.path.join(CACHE_DIR, self.name)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        parsed_url=urlparse(self.url)
        file_name=os.path.basename(parsed_url.path)
        self.zip_path=os.path.join(self.cache_path, file_name)
        try: 
            urlretrieve(self.url,self.zip_path)
            # print('Downloaded Bank!')
        except urllib.error.URLError as e:
            print("Failed to download the file: ",e)

    def preprocess(self):

        with zipfile.ZipFile(self.zip_path,'r') as zip_file:
            zip_file.extractall(self.cache_path)
        with tarfile.open(os.path.join(self.cache_path,'TCGA-PANCAN-HiSeq-801x20531.tar.gz'),'r:gz') as tar_file:
            tar_file.extractall(self.cache_path)
        
        os.remove(self.zip_path)
        os.remove(os.path.join(self.cache_path,'TCGA-PANCAN-HiSeq-801x20531.tar.gz'))

        data=pd.read_csv(os.path.join(self.cache_path,'TCGA-PANCAN-HiSeq-801x20531','data.csv'))
        X=data.iloc[:,1:].values
        labels=pd.read_csv(os.path.join(self.cache_path,'TCGA-PANCAN-HiSeq-801x20531','labels.csv'))
        label_encoder=LabelEncoder()
        y=label_encoder.fit_transform(labels['Class'])
        self.set_data(X,y)

class ElectricityLoadDiagrams(Dataset):

    global DATASET_DIR,CACHE_DIR

    def __init__(self,url):
        super().__init__('ElectrictiyLoadDiagrams',url)
    
    def download(self):
        self.cache_path=os.path.join(CACHE_DIR, self.name)
        if not os.path.exists(self.cache_path):
            os.mkdir(self.cache_path)
        
        parsed_url=urlparse(self.url)
        file_name=os.path.basename(parsed_url.path)
        self.zip_path=os.path.join(self.cache_path, file_name)

        try:
            urlretrieve(self.url, self.zip_path)
        except urllib.error.URLError as e:
            print("Failed to download the file ",e)
    
    def preprocess(self):
        with zipfile.ZipFile(self.zip_path,'r') as zip_file:
            zip_file.extract('LD2011_2014.txt',self.cache_path)
        
        os.remove(self.zip_path)

        data=pd.read_csv(os.path.join(self.cache_path,'LD2011_2014.txt'),delimiter=';')

        X=data.iloc[:,1:].values.T
        y=None
        self.set_data(X,y)


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













    



        



        

