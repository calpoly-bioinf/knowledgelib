from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import StandardScaler
import pandas as pd

from joblib import Parallel, delayed

from . import common
from . import distance

class PassTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit( self, X, y = None  ):
        return self
    
    def transform(self, X, y = None ):
        return X

class ScaleTransformer( BaseEstimator, TransformerMixin ):
    distributions = pd.DataFrame([
            ('Unscaled data', lambda X: PassTransformer().fit(X)),
            ('Data after standard scaling',lambda X: StandardScaler().fit(X)),
            ('Data after min-max scaling',lambda X: MinMaxScaler().fit(X)),
            ('Data after max-abs scaling',lambda X: MaxAbsScaler().fit(X)),
            ('Data after robust scaling',lambda X: RobustScaler(quantile_range=(25, 75)).fit(X)),
            ('Data after power transformation (Yeo-Johnson)',lambda X: PowerTransformer(method='yeo-johnson').fit(X)),
            ('Data after power transformation (Box-Cox)',lambda X: PowerTransformer(method='box-cox').fit(X)),
            ('Data after quantile transformation (uniform pdf)',lambda X: QuantileTransformer(output_distribution='uniform').fit(X)),
            ('Data after quantile transformation (gaussian pdf)',lambda X: QuantileTransformer(output_distribution='normal').fit(X)),
            ('Data after sample-wise L2 normalizing',lambda X: Normalizer().fit(X))],columns=['method','scaler'])
    
    def __init__(self,method='Unscaled data'):
        self.method = method
        self.distribution_func = self.distributions.set_index('method').loc[method,'scaler']
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        self.transformer = self.distribution_func(X)
        return self
    
    def transform(self, X, y = None ):
        return pd.DataFrame(self.transformer.transform(X), columns = X.columns,index=X.index)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return pd.DataFrame(self.transform(X), columns = X.columns)
    
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]
    
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=3, classes = None):
        self.k = k
        self.classes = classes

    def fit(self, X, y=None):
        self.neigh = KNeighborsClassifier(n_neighbors=self.k)
        self.neigh.fit(X, y)
        self.y = y
        return self

    def transform(self, X, y=None):
        distances, indices = self.neigh.kneighbors(X, n_neighbors=self.k+1, return_distance=True)
        if np.sum(distances[:,0]) == 0:
            distances = distances[:,1:]
            indices = indices[:,1:]
        else:
            distances = distances[:,:-1]
            indices = indices[:,:-1]
        labels = self.y.iloc[indices.flat].values.reshape(indices.shape)
        return distances, indices, labels

class DistributionTransformer(BaseEstimator, TransformerMixin):
    distance_funcs = {
        'L1':distance.L1,
        'L2':distance.L2,
        'FSIGN':distance.FSIGN
    }
    def __init__(self,gene_cols=[],ncores=6,distance='L1'):
        self.ncores = ncores
        self.gene_cols = list(gene_cols)
        self.distance = distance
        self.distance_func = self.distance_funcs[distance]
    
    def fit( self, X, y = None  ):
        if len(self.gene_cols) > 0:
            X = X[self.gene_cols]
        return self
    
    def transform(self, X, y = None ):
        if len(self.gene_cols) > 0:
            X = X[self.gene_cols]
        
        pair_ixs = common.get_pair_inxs(X.shape[0])
        chunks_pair_ixs = list(common.divide_chunks(pair_ixs,common.calc_len_chunk(len(pair_ixs),self.ncores)))
        return pd.concat(
            Parallel(n_jobs=self.ncores)(delayed(distance.chunk_distance)(X,chunk,self.distance_func) for chunk in chunks_pair_ixs))
    
    def fit_transform(self,X,y=None):
        if len(self.gene_cols) > 0:
            X = X[self.gene_cols]
            
        self.fit(X)
        return self.transform(X)

def standard_scale(genes_df):
    
    scaler = StandardScaler()
    scaler.fit(genes_df)

    genes_df_scaled = pd.DataFrame(scaler.transform(genes_df),index=genes_df.index,columns=genes_df.columns).fillna(0)
    return genes_df_scaled


def arctan_scale(genes_df):
    
    scaler = StandardScaler()
    scaler.fit(genes_df)

    genes_df_scaled = pd.DataFrame(scaler.transform(genes_df),index=genes_df.index,columns=genes_df.columns).fillna(0)
    return np.arctan(genes_df_scaled)


