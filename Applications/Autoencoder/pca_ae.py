import numpy as np
from sklearn.decomposition import PCA


class pca_ae:

    def __init__(self, target_dim:int) -> None:
        self.target_dim=target_dim  
    def encoder(self, A:np.ndarray)->None:
        self.pca=PCA(n_components=self.target_dim)
        B=self.pca.fit_transform(A)
        self.enc=B
    def decoder(self)->np.ndarray:

        return self.pca.inverse_transform(self.enc)   
    def reconstruct(self, A:np.ndarray)->np.ndarray:

        self.encoder(A)
        return self.decoder()
